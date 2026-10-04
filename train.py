"""Train a tiny MLP to imitate closed-form inverse kinematics.

Setup: planar 2-link arm, unit links (L1 = L2 = 1). The teacher is the
law-of-cosines solution with the elbow-down branch fixed; the student is
a 2-16-16-4 tanh MLP that maps a normalized target (x, y) to
(cos t1, sin t1, cos t2, sin t2). Angles are predicted on the unit
circle (never raw radians) so the +/-pi wrap is not a discontinuity.

Capacity is deliberately small (~700 params): the point is what a tiny
student learns and where it gives up, not leaderboard error.

Outputs: runs/arm-ik-nn-1.pt (state dict) and runs/report.json (metrics).
"""

import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

SEED = 7
L1 = 1.0
L2 = 1.0
SNAPSHOT = "arm-ik-nn/1"

N_UNIFORM = 60_000
N_RIM = 0  # no oversample: joint-uniform sampling lets difficulty follow
# physics monotonically (interior < rim < near-singular)
RIM_T2 = 0.45  # |theta2| below this is near-singular territory
EPOCHS = 10
BATCH = 2048
LR = 3e-3
LBFGS_STEPS = 120
MM_PER_UNIT = 300.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fk(t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics for unit links. Angles in radians."""
    x = L1 * torch.cos(t1) + L2 * torch.cos(t1 + t2)
    y = L1 * torch.sin(t1) + L2 * torch.sin(t1 + t2)
    return x, y


def sample_data(n_uniform: int, n_rim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (theta1, theta2) elbow-down (theta2 <= 0), labels exact by FK.

    Uniform-over-joints sampling covers the workspace the way the arm
    actually moves. The rim oversample stays available as a knob, but it
    ships at zero: an early run showed extra rim density outweighed the
    steep mapping there and flattened the difficulty curve. Natural
    sampling lets difficulty follow physics.
    """
    t1_u = torch.rand(n_uniform) * 2 * math.pi - math.pi
    t2_u = -torch.rand(n_uniform) * math.pi
    t1_r = torch.rand(n_rim) * 2 * math.pi - math.pi
    t2_r = -torch.rand(n_rim) * RIM_T2
    t1 = torch.cat([t1_u, t1_r])
    t2 = torch.cat([t2_u, t2_r])
    x, y = fk(t1, t2)
    inputs = torch.stack([x / 2.0, y / 2.0], dim=1)
    targets = torch.stack(
        [torch.cos(t1), torch.sin(t1), torch.cos(t2), torch.sin(t2)], dim=1
    )
    return inputs, targets


class IkNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def decode(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-circle outputs back to angles via atan2 (wrap-safe)."""
    t1 = torch.atan2(out[:, 1], out[:, 0])
    t2 = torch.atan2(out[:, 3], out[:, 2])
    return t1, t2


def angle_rmse_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = torch.atan2(torch.sin(a - b), torch.cos(a - b))
    return float(torch.sqrt((diff**2).mean()) * 180.0 / math.pi)


def band_mask(r: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (r >= lo) & (r < hi)


def evaluate(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    pred_t1, pred_t2 = decode(out)
    true_t1 = torch.atan2(targets[:, 1], targets[:, 0])
    true_t2 = torch.atan2(targets[:, 3], targets[:, 2])
    # Recover target positions from the normalized inputs.
    tx, ty = inputs[:, 0] * 2.0, inputs[:, 1] * 2.0
    px, py = fk(pred_t1, pred_t2)
    ee = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)
    r = torch.sqrt(tx**2 + ty**2)

    def dist_stats(v: torch.Tensor) -> dict:
        return {
            "mean": float(v.mean()),
            "p50": float(v.median()),
            "p95": float(v.quantile(0.95)),
            "max": float(v.max()),
        }

    bands = {
        "interior": band_mask(r, 0.0, 1.6),
        "rim": band_mask(r, 1.6, 1.95),
        "near_singular": band_mask(r, 1.95, 10.0),
    }
    report: dict = {
        "angle_rmse_deg": angle_rmse_deg(
            torch.cat([pred_t1, pred_t2]), torch.cat([true_t1, true_t2])
        ),
        "ee_rmse_units": float(torch.sqrt((ee**2).mean())),
        "ee_rmse_mm": float(torch.sqrt((ee**2).mean()) * MM_PER_UNIT),
        "bands": {},
    }
    for name, mask in bands.items():
        n = int(mask.sum())
        if n == 0:
            continue
        band_ee_mm = ee[mask] * MM_PER_UNIT
        report["bands"][name] = {
            "n": n,
            "ee_rmse_mm": float(band_ee_mm.pow(2).mean().sqrt()),
            "ee_mm": dist_stats(band_ee_mm),
        }
    return report


def load_split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild the exact seeded train/test split (no training)."""
    seed_all(SEED)
    inputs, targets = sample_data(N_UNIFORM, N_RIM)
    n = inputs.shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    return (
        inputs[perm[:split]],
        targets[perm[:split]],
        inputs[perm[split:]],
        targets[perm[split:]],
    )


def write_report(model: nn.Module, test_in: torch.Tensor, test_tg: torch.Tensor) -> dict:
    report = evaluate(model, test_in, test_tg)
    report.update(
        {
            "snapshot": SNAPSHOT,
            "seed": SEED,
            "links": [L1, L2],
            "branch": "elbow-down",
            "net": "2-16-16-4 tanh MLP, unit-circle outputs",
            "mm_per_unit": MM_PER_UNIT,
        }
    )
    return report


def main() -> None:
    if "--eval-only" in sys.argv:
        # Recompute the report from the committed weights. Deterministic:
        # same seed rebuilds the identical split, no training involved.
        _, _, test_in, test_tg = load_split()
        model = IkNet()
        model.load_state_dict(torch.load("runs/arm-ik-nn-1.pt"))
        report = write_report(model, test_in, test_tg)
        report["n_train"] = int(load_split()[0].shape[0])
        report["n_test"] = int(test_in.shape[0])
        with open("runs/report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    seed_all(SEED)
    os.makedirs("runs", exist_ok=True)

    train_in, train_tg, test_in, test_tg = load_split()

    model = IkNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_in, train_tg),
        batch_size=BATCH,
        shuffle=True,
    )
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    # Second-order finish: tiny net, smooth loss, full-batch LBFGS
    # converges superlinearly where Adam crawls. Deterministic.
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=20,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    for _ in range(LBFGS_STEPS):
        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss = loss_fn(model(train_in), train_tg)
            loss.backward()
            return loss

        lbfgs.step(closure)

    report = write_report(model, test_in, test_tg)
    report["n_train"] = int(train_in.shape[0])
    report["n_test"] = int(test_in.shape[0])
    torch.save(model.state_dict(), "runs/arm-ik-nn-1.pt")
    with open("runs/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
