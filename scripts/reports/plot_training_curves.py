"""
Plots training curves logged to Weights & Biases by `main.py`:

  1. energy_variance.png  — energy per site and variance vs. training step.
  2. loss_components.png — three-panel plot of sym_loss, energy_loss, and total loss
                            vs. training step (sym_loss panel is empty if the run
                            didn't have symmetrization enabled).

Run from the repo root (requires `wandb login` once beforehand):

    uv run python scripts/reports/plot_training_curves.py <run_name_or_id_or_path>
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

sys.path.insert(0, str(Path(__file__).parent))

from wandb_utils import DEFAULT_PROJECT, extract_series, fetch_history, resolve_run

DPI = 300
AXIS_GREY = "#595959"
LINE_COLOR = "#c0504d"

plt.rcParams.update(
    {
        "axes.edgecolor": AXIS_GREY,
        "axes.labelcolor": AXIS_GREY,
        "xtick.color": AXIS_GREY,
        "ytick.color": AXIS_GREY,
        "text.color": AXIS_GREY,
        "grid.color": AXIS_GREY,
    }
)


def plot_energy_variance(rows: list[dict], out_dir: Path, label: str) -> Path:
    steps_e, energy = extract_series(rows, "energy_per_site")
    steps_v, variance = extract_series(rows, "variance")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(steps_e, energy, color=LINE_COLOR)
    axes[0].set_ylabel("Energy / site")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps_v, variance, color=LINE_COLOR)
    axes[1].set_ylabel("Cross-Sample Energy Variance")
    axes[1].set_xlabel("Step")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    out_path = out_dir / f"{label}_energy_variance.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_loss_components(rows: list[dict], out_dir: Path, label: str) -> Path:
    panels = [
        ("sym_loss", "Symmetrization loss"),
        ("energy_loss", "Energy loss"),
        ("loss", "Total loss"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, (key, panel_label) in zip(axes, panels):
        steps, values = extract_series(rows, key)
        if not values:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(steps, values, color=LINE_COLOR)
        ax.set_ylabel(panel_label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Step")

    fig.tight_layout()

    out_path = out_dir / f"{label}_loss_components.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run",
        help="Run name, run id, or full 'entity/project/run_id' path (e.g. 'decent-lake-4').",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"W&B project (default: '{DEFAULT_PROJECT}').")
    parser.add_argument("--entity", default=None, help="W&B entity (default: your logged-in default entity).")
    parser.add_argument(
        "--out-dir",
        default=Path(__file__).parent / "figures",
        type=Path,
        help="Directory to write PNGs into (default: scripts/reports/figures).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    api = wandb.Api()
    run = resolve_run(api, args.run, args.project, args.entity)
    rows = fetch_history(run)

    label = f"{timestamp}_{run.name}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    energy_path = plot_energy_variance(rows, args.out_dir, label)
    loss_path = plot_loss_components(rows, args.out_dir, label)

    print(f"Run: {run.name} ({run.id})")
    print(f"Wrote {energy_path}")
    print(f"Wrote {loss_path}")


if __name__ == "__main__":
    main()
