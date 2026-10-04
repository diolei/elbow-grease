"""Export the committed snapshot to a portable JSON handoff.

Reads runs/arm-ik-nn-1.pt plus runs/report.json, writes
snapshots/arm-ik-nn-1.json: layer weights in [out][in] row order,
normalization, net spec, and the frozen evaluation report.
The handoff is the full contract — nothing else leaves this repo.
"""

import json
import math
import os

import torch

from train import IkNet, SNAPSHOT


def main() -> None:
    os.makedirs("snapshots", exist_ok=True)
    model = IkNet()
    model.load_state_dict(torch.load("runs/arm-ik-nn-1.pt"))
    model.eval()

    layers = []
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    for lin in linears:
        layers.append(
            {
                "weights": lin.weight.detach().tolist(),
                "bias": lin.bias.detach().tolist(),
            }
        )

    with open("runs/report.json", encoding="utf-8") as f:
        report = json.load(f)

    handoff = {
        "snapshot": SNAPSHOT,
        "activation": "tanh",
        "output": "unit-circle (cos t1, sin t1, cos t2, sin t2), atan2 decode",
        "normalization": {"in_scale": 0.5},
        "layers": layers,
        "report": report,
    }
    out = "snapshots/arm-ik-nn-1.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(handoff, f, indent=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"wrote {out}: {len(layers)} layers, {params} params")

    # Parity probes: fixed targets decoded by the torch model. Downstream
    # consumers paste these vectors into their own tests; any transcription
    # error in the weights fails loudly there.
    probes = [(0.5, 0.5), (1.5, 1.2), (1.9, 0.3)]
    with torch.no_grad():
        pin = torch.tensor(
            [[x * 0.5, y * 0.5] for x, y in probes], dtype=torch.float32
        )
        pout = model(pin)
    parity = []
    for (x, y), row in zip(probes, pout.tolist()):
        t1 = math.atan2(row[1], row[0])
        t2 = math.atan2(row[3], row[2])
        parity.append({"x": x, "y": y, "t1": t1, "t2": t2})
    with open("snapshots/parity.json", "w", encoding="utf-8") as f:
        json.dump(parity, f, indent=2)
    print(f"wrote snapshots/parity.json: {len(parity)} probes")


if __name__ == "__main__":
    main()
