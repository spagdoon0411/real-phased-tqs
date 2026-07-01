"""
Sweeps the transverse-field strength h and plots |magnetization| vs. h for every
checkpoint saved during a training run.

Mirrors yuanhangzhang98/transformer_quantum_state's test.py: for a fixed system size L,
each h value is read out `--ensemble-size` times by resampling the model from scratch,
and the order parameter for a single reading is

    m(x) = | mean_i (2 x_i - 1) |

averaged over sample-weight-weighted unique configurations. The per-h mean and std
across ensemble readings are then plotted (DMRG comparison, present in the original
script, is intentionally omitted — this only reports the model's own estimate).

Model hyperparameters (architecture, L range, h range, J, periodicity) are recovered
from the `run_summary.json` that `main.py` writes alongside its checkpoints, so only
the sweep granularity needs to be supplied on the command line.

Run from the repo root:

    uv run python scripts/reports/plot_magnetization_vs_h.py checkpoints/20260701_120000 \\
        --h-min 0.0 --h-max 2.0 --n-h 21
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hamiltonian.transverse_field_ising import TransverseFieldIsing
from hamiltonian.transverse_field_ising_y import TransverseFieldIsingY
from model.tqs import TransformerQuantumState

DPI = 300


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_config(ckpt_dir: Path) -> dict:
    summary_path = ckpt_dir / "run_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No run_summary.json found in '{ckpt_dir}'; can't recover model hyperparameters.")
    with open(summary_path) as f:
        return json.load(f)


def _build_model(config: dict, device: torch.device) -> TransformerQuantumState:
    ham_cls = TransverseFieldIsing if config["hamiltonian"] == "TFI-X" else TransverseFieldIsingY
    L_min, L_max = config["L_range"]
    h_min, h_max = config["h_range"]
    hamiltonian = ham_cls(
        system_dim_range=np.array([L_min, L_max]),
        static_params=np.array([config["J"]]),
        ranged_params=np.array([[h_min, h_max]]),
        periodic=config["periodic"],
        device=device,
    )
    return TransformerQuantumState(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dim_feedforward=config["dim_feedforward"],
        max_len=L_max,
        hamiltonian=hamiltonian,
        device=device,
    )


def _list_checkpoints(ckpt_dir: Path, stride: int) -> list[Path]:
    ckpts = sorted(ckpt_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    if not ckpts:
        raise SystemExit(f"No checkpoints (NNNNNN.pt) found in '{ckpt_dir}'.")
    return ckpts[::stride]


@torch.no_grad()
def _magnetization_reading(
    model: TransformerQuantumState,
    L: int,
    num_walkers: int,
    microbatch_size: int,
    sample_buffer_size: int,
) -> float:
    """
    One sampling readout of the |magnetization| order parameter. Chains are always
    sampled at the model's fixed `max_len`; only the first `L` sites correspond to the
    physical system currently selected via `set_prefix`, so the rest are discarded here
    the same way `hamiltonian.observables()` ignores them for energy evaluation.
    """
    samples, sample_weight = model.sample_iid_microbatches(
        num_walkers=num_walkers,
        microbatch_size=microbatch_size,
        sample_buffer_size=sample_buffer_size,
    )
    spins_pm = 2 * samples[:L].to(torch.get_default_dtype()) - 1
    per_config = spins_pm.mean(dim=0).abs()
    return (per_config * sample_weight).sum().item()


def sweep_checkpoint(
    model: TransformerQuantumState,
    ckpt_path: Path,
    device: torch.device,
    L: int,
    J: float,
    h_values: np.ndarray,
    ensemble_size: int,
    num_walkers: int,
    microbatch_size: int,
    sample_buffer_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    system_dim = torch.tensor([float(L)], device=device)
    means = np.zeros(len(h_values))
    stds = np.zeros(len(h_values))
    for i, h in enumerate(tqdm(h_values, desc=f"{ckpt_path.stem}: h sweep", leave=False)):
        phys_params = torch.tensor([float(h), J], device=device)
        model.set_prefix(phys_params, system_dim)
        readings = [
            _magnetization_reading(model, L, num_walkers, microbatch_size, sample_buffer_size)
            for _ in range(ensemble_size)
        ]
        means[i] = float(np.mean(readings))
        stds[i] = float(np.std(readings))
    return means, stds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "ckpt_dir", type=Path, help="Directory with checkpoint NNNNNN.pt files and run_summary.json."
    )
    parser.add_argument("--h-min", type=float, required=True, help="Lower bound of the h sweep.")
    parser.add_argument("--h-max", type=float, required=True, help="Upper bound of the h sweep.")
    parser.add_argument("--n-h", type=int, required=True, help="Number of h values to sweep (granularity).")
    parser.add_argument(
        "--L", type=int, default=None, help="System size to evaluate at (default: the run's trained L_max)."
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=10,
        help="Independent sampling readings per h (default: 10, matching transformer_quantum_state).",
    )
    parser.add_argument("--num-walkers", type=int, default=2048, help="Walkers per reading (default: 2048).")
    parser.add_argument(
        "--microbatch-size", type=int, default=2048, help="Autoregressive microbatch size (default: 2048)."
    )
    parser.add_argument(
        "--sample-buffer-size", type=int, default=2048, help="Dedup buffer width (default: 2048)."
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Only plot every Nth checkpoint, oldest-first (default: 1, i.e. all)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "figures" / "magnetization_vs_h.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to save the raw swept tensors into (default: scripts/reports/data).",
    )
    args = parser.parse_args()

    config = _load_config(args.ckpt_dir)
    device = _select_device()
    model = _build_model(config, device)

    L_min, L_max = config["L_range"]
    L = args.L if args.L is not None else L_max
    if not (L_min <= L <= L_max):
        raise SystemExit(f"--L={L} is outside the trained range [{L_min}, {L_max}].")

    h_min_trained, h_max_trained = config["h_range"]
    if args.h_min < h_min_trained or args.h_max > h_max_trained:
        raise SystemExit(
            f"Requested h range [{args.h_min}, {args.h_max}] falls outside the trained range "
            f"[{h_min_trained}, {h_max_trained}]; the model was never shown those field strengths."
        )
    h_values = np.linspace(args.h_min, args.h_max, args.n_h)

    ckpts = _list_checkpoints(args.ckpt_dir, args.stride)
    steps = [int(p.stem) for p in ckpts]

    means_all = np.zeros((len(ckpts), args.n_h))
    stds_all = np.zeros((len(ckpts), args.n_h))

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(steps), vmax=max(steps))

    for row, (ckpt_path, step) in enumerate(tqdm(list(zip(ckpts, steps)), desc="Checkpoints")):
        means, stds = sweep_checkpoint(
            model,
            ckpt_path,
            device,
            L,
            config["J"],
            h_values,
            args.ensemble_size,
            args.num_walkers,
            args.microbatch_size,
            args.sample_buffer_size,
        )
        means_all[row] = means
        stds_all[row] = stds
        color = cmap(norm(step))
        ax.plot(h_values, means, color=color)
        ax.fill_between(h_values, means - stds, means + stds, color=color, alpha=0.15)
        tqdm.write(f"step {step:6d}: done")

    ax.set_xlabel("h")
    ax.set_ylabel("|magnetization|")
    ax.set_title(f"Magnetization vs. h — L={L}")
    ax.grid(alpha=0.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Training step")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {args.out}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / f"{args.ckpt_dir.name}_magnetization.pt"
    torch.save(
        {
            "h_values": torch.tensor(h_values),
            "steps": torch.tensor(steps),
            "means": torch.tensor(means_all),
            "stds": torch.tensor(stds_all),
            "L": L,
        },
        data_path,
    )
    print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
