import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from tabulate import tabulate

from hamiltonian.symmetries import Reflection, SpinFlip, Translation
from hamiltonian.transverse_field_ising import TransverseFieldIsing
from hamiltonian.transverse_field_ising_y import TransverseFieldIsingY
from model.tqs import TransformerQuantumState
from training.training_loop import train

# Physical parameters
hamiltonian_id = "x"
L_min, L_max = (10, 30)
h_min, h_max = (0.5, 1.5)
J = 1.0
periodic = True

# Training parameters
sampler_id = "iid"
n_steps = 2000
num_walkers = 2048
microbatch_size = 2048  # Ignored for the tree sampler
sample_buffer_size = 2048
warmup_steps = 500

# Symmetry regularization parameters
sym_beta_max = 0.05
sym_tau_frac = 0.1 * n_steps
sym_batch_size = 512
sym_phase_weight = 1.0

# Model parameters
d_model = 32
n_layers = 8
n_heads = 8
dim_feedforward = 4 * d_model

# Logging / checkpointing
wandb_project = "real-phased-tqs"
checkpoint_every = 200


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_run_config(device_str: str) -> dict:
    return {
        "hamiltonian": "TFI-X" if hamiltonian_id == "x" else "TFI-Y",
        "L_range": [L_min, L_max],
        "h_range": [h_min, h_max],
        "J": J,
        "periodic": periodic,
        "sampler": sampler_id,
        "n_steps": n_steps,
        "warmup_steps": warmup_steps,
        "num_walkers": num_walkers,
        "microbatch_size": microbatch_size,
        "sample_buffer_size": sample_buffer_size,
        "sym_beta_max": sym_beta_max,
        "sym_tau_frac": sym_tau_frac,
        "sym_batch_size": sym_batch_size,
        "sym_phase_weight": sym_phase_weight,
        "d_model": d_model,
        "dim_feedforward": dim_feedforward,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "device": device_str,
    }


def _print_summary(config: dict) -> None:
    rows = [
        ["Hamiltonian", config["hamiltonian"]],
        ["L range", config["L_range"]],
        ["h range", config["h_range"]],
        ["J (static)", config["J"]],
        ["Periodic", config["periodic"]],
        ["Sampler", config["sampler"]],
        ["Steps", config["n_steps"]],
        ["Warmup steps", config["warmup_steps"]],
        ["Walkers", config["num_walkers"]],
        ["Microbatch size", config["microbatch_size"]],
        ["Sample buffer size", config["sample_buffer_size"]],
        ["Sym beta_max", config["sym_beta_max"]],
        ["Sym tau_frac", config["sym_tau_frac"]],
        ["Sym batch size", config["sym_batch_size"]],
        ["Sym phase weight", config["sym_phase_weight"]],
        ["d_model", config["d_model"]],
        ["Feedforward dim", config["dim_feedforward"]],
        ["Layers", config["n_layers"]],
        ["Heads", config["n_heads"]],
        ["Device", config["device"]],
    ]
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="rounded_outline"))


def main() -> None:
    device = _select_device()
    device_str = str(device)
    if device.type == "cuda":
        device_str = f"cuda ({torch.cuda.get_device_name(device)})"

    config = _build_run_config(device_str)
    _print_summary(config)

    wandb.init(project=wandb_project, config=config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path("checkpoints") / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "run_summary.json", "w") as f:
        json.dump(config, f, indent=2)

    ham_cls = TransverseFieldIsing if hamiltonian_id == "x" else TransverseFieldIsingY
    sym = [SpinFlip(), Reflection(), Translation()] if hamiltonian_id == "x" else []
    hamiltonian = ham_cls(
        system_dim_range=np.array([L_min, L_max]),
        static_params=np.array([J]),
        ranged_params=np.array([[h_min, h_max]]),
        periodic=periodic,
        device=device,
        symmetries=sym,
    )

    model = TransformerQuantumState(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        max_len=L_max,
        hamiltonian=hamiltonian,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    def noam_lambda(step: int) -> float:
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

    tau = sym_tau_frac * n_steps

    def beta_lambda(step: int) -> float:
        return sym_beta_max * (1.0 - math.exp(-step / tau))

    def log(step: int, diagnostics: dict) -> None:
        lr = optimizer.param_groups[0]["lr"]
        sym_loss = diagnostics.get("sym_loss")
        dedup = 1 - diagnostics["n_unique"] / num_walkers
        p = diagnostics["phys_params"]

        line1 = f"step {step:4d}  {hamiltonian.param_str()}"
        line2 = (
            f"  energy {diagnostics['energy']:+.6f}"
            f"  /site {diagnostics['energy_per_site']:+.6f}"
            f"  variance {diagnostics['variance']:.6f}"
        )
        if sym_loss is not None:
            line3 = f"  lr {lr:.2e}  e_loss {diagnostics['energy_loss']:+.6f}"
            line4 = (
                f"  β {diagnostics['beta']:.4f}"
                f"  sym_loss {sym_loss:.6f}"
                f"  total {diagnostics['loss']:+.6f}"
                f"  dedup {dedup:.1%}  it {diagnostics['iter_time']:.3f}s"
            )
            print(line1, line2, line3, line4, sep="\n")
        else:
            line3 = (
                f"  lr {lr:.2e}  e_loss {diagnostics['energy_loss']:+.6f}"
                f"  dedup {dedup:.1%}  it {diagnostics['iter_time']:.3f}s"
            )
            print(line1, line2, line3, sep="\n")

        metrics = {
            "energy": diagnostics["energy"],
            "energy_per_site": diagnostics["energy_per_site"],
            "variance": diagnostics["variance"],
            "energy_loss": diagnostics["energy_loss"],
            "loss": diagnostics["loss"],
            "lr": lr,
            "h": p[0],
            "system_dim": diagnostics["system_dim"],
            "dedup": dedup,
            "iter_time": diagnostics["iter_time"],
        }
        if sym_loss is not None:
            metrics["sym_loss"] = sym_loss
            metrics["beta"] = diagnostics["beta"]
        wandb.log(metrics, step=step)

        if step % checkpoint_every == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                ckpt_dir / f"{step:06d}.pt",
            )

    match sampler_id:
        case "iid":

            def sampler() -> tuple[torch.Tensor, torch.Tensor]:
                return model.sample_iid_microbatches(
                    num_walkers=num_walkers,
                    microbatch_size=microbatch_size,
                    sample_buffer_size=sample_buffer_size,
                )

        case "tree":

            def sampler() -> tuple[torch.Tensor, torch.Tensor]:
                return model.sample_states(
                    num_walkers=num_walkers,
                    sample_buffer_size=sample_buffer_size,
                )

    train(
        model=model,
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        n_steps=n_steps,
        sampler=sampler,
        on_step=log,
        scheduler=scheduler,
        beta_schedule=beta_lambda,
        sym_batch_size=sym_batch_size,
        sym_phase_weight=sym_phase_weight,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
