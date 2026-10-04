# elbow-grease

A tiny student learns what cosine already knows: a 2-16-16-4 tanh MLP
imitates closed-form inverse kinematics for a planar 2-link arm with
unit links (L1 = L2 = 1). The point is what a ~400-parameter net learns
and where it gives up — not leaderboard error.

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt
```
CPU-only; no GPU needed anywhere.

## Reproduce

```bash
.venv/bin/python train.py     # ~4 min: trains, writes runs/
.venv/bin/python train.py --eval-only   # seconds: recompute report only
.venv/bin/python export.py    # writes snapshots/arm-ik-nn-1.json
```

Everything is seeded (`SEED = 7`); the eval-only pass reproduces the
committed report bit-for-bit.

## Method in brief

- **Teacher:** law-of-cosines IK with the **elbow-down branch fixed**.
  The map (x, y) → θ is one-to-many, and training on mixed branches
  teaches the mean — a broken arm. One branch, always.
- **Sampling:** joint-uniform θ1 ∈ [-π, π], θ2 ∈ [-π, 0]; labels exact
  by forward kinematics. No workspace oversampling, so difficulty
  follows physics.
- **Outputs on the unit circle:** the net predicts
  (cos θ1, sin θ1, cos θ2, sin θ2), decoded with atan2. Raw-angle
  regression breaks at the ±π wrap; this does not.
- **Optimizer:** 10 Adam epochs to break symmetry, then full-batch
  LBFGS (strong Wolfe) to convergence. Deterministic.

## Results (`arm-ik-nn/1`, held-out n = 12,000, 1 unit = 300 mm)

| Band | n | EE RMSE | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- |
| Interior (r < 1.6) | 7,154 | 10.0 mm | 8.0 mm | 17.9 mm | 26.4 mm |
| Rim (1.6–1.95) | 3,223 | 9.8 mm | 6.7 mm | 19.9 mm | 34.6 mm |
| Near-singular (r > 1.95) | 1,623 | 12.6 mm | 7.1 mm | **27.2 mm** | 34.6 mm |

Overall: 10.3 mm end-effector RMSE, 2.63° joint RMSE. Medians sit near
7 mm everywhere — the student shadows the teacher — while the 95th
percentile nearly doubles past r = 1.95, where the Jacobian goes
singular and small Cartesian steps demand large joint swings.

## Files

| File | Role |
| --- | --- |
| `train.py` | Data, model, two-phase training, banded evaluation |
| `export.py` | Snapshot → portable JSON handoff (`snapshots/`) |
| `runs/` | Trained weights (`.pt`) + evaluation `report.json` |
| `runs/report-64-64.json` | Archived 64-64-capacity run (9.1 mm overall, flat bands — capacity without character, not shipped) |
| `snapshots/arm-ik-nn-1.json` | The handoff: weights, normalization, spec, frozen report |

## Snapshot contract

`snapshots/arm-ik-nn-1.json` is the full interface: 3 layers in
`[out][in]` row order, `normalization.in_scale = 0.5`, tanh hidden
activations, linear head. Any consumer that forwards tanh correctly
and decodes with atan2 reproduces these numbers.
