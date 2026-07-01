"""
Loads a magnetization sweep produced by `plot_magnetization_vs_h.py` (saved under
scripts/reports/data/) and renders a 3D surface of |magnetization| vs. h, swept out
across training step.

Run from the repo root:

    uv run python scripts/reports/plot_magnetization_3d.py \\
        scripts/reports/data/20260701_120000_magnetization.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DPI = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Tensor file written by plot_magnetization_vs_h.py.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: figures/<data_path stem>_3d.png).",
    )
    args = parser.parse_args()

    data = torch.load(args.data_path, weights_only=False)
    h_values = data["h_values"].numpy()
    steps = data["steps"].numpy()
    means = data["means"].numpy()  # (n_steps, n_h)
    L = data["L"]

    H, STEP = np.meshgrid(h_values, steps, indexing="xy")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(H, STEP, means, cmap="viridis", edgecolor="none", antialiased=True)

    ax.set_xlabel("h")
    ax.set_ylabel("Training step")
    ax.set_zlabel("|magnetization|")
    ax.set_title(f"Magnetization vs. h across training — L={L}")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="|magnetization|")
    fig.tight_layout()

    out = args.out if args.out is not None else Path(__file__).parent / "figures" / f"{args.data_path.stem}_3d.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
