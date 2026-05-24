import torch

from hamiltonian.transverse_field_ising import TransverseFieldIsing
from hamiltonian.transverse_field_ising_y import TransverseFieldIsingY
from model.tqs import TransformerQuantumState
from training.training_loop import train

# Physical parameters
hamiltonian_id = "x"
L = 8
h = 0.25
J = 1.0
periodic = False

# Training parameters
device = torch.device("cpu")
sampler_id = "iid"
n_steps = 2000
num_walkers = 1024
microbatch_size = 128  # Ignored for the tree sampler
sample_buffer_size = 1024
warmup_steps = 4000

# Model parameters
d_model = 32
n_layers = 8
n_heads = 8
dim_feedforward = 4 * d_model


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def main() -> None:
    device = _select_device()

    ham_cls = TransverseFieldIsing if hamiltonian_id == "x" else TransverseFieldIsingY
    hamiltonian = ham_cls(
        system_dim=torch.tensor([float(L)]),
        phys_params=torch.tensor([h]),
        coupling=J,
        periodic=periodic,
        device=device,
    )

    model = TransformerQuantumState(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        max_len=L,
        hamiltonian=hamiltonian,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    def noam_lambda(step: int) -> float:
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

    def log(step: int, diagnostics: dict) -> None:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"step {step:4d}  "
            f"energy {diagnostics['energy']:+.6f}  "
            f"energy/site {diagnostics['energy_per_site']:+.6f}  "
            f"variance {diagnostics['variance']:.6f}  "
            f"loss {diagnostics['loss']:+.6f}  "
            f"lr {lr:.2e}  "
            f"it {diagnostics['iter_time']:.3f}s"
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
    )


if __name__ == "__main__":
    main()
