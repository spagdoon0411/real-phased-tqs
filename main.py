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
lr = 1e-4
num_walkers = 1024
microbatch_size = 128  # Ignored for the tree sampler
sample_buffer_size = 1024

# Model parameters
d_model = 32
n_layers = 8
n_heads = 8


def main() -> None:
    ham_cls = TransverseFieldIsing if hamiltonian_id == "x" else TransverseFieldIsingY
    hamiltonian = ham_cls(
        system_dim=torch.tensor([float(L)]),
        phys_params=torch.tensor([h]),
        coupling=J,
        periodic=periodic,
    )

    model = TransformerQuantumState(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=L,
        hamiltonian=hamiltonian,
        device=device,
    )

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def log(step: int, diagnostics: dict) -> None:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"step {step:4d}  "
            f"energy {diagnostics['energy']:+.6f}  "
            f"energy/site {diagnostics['energy_per_site']:+.6f}  "
            f"variance {diagnostics['variance']:.6f}  "
            f"loss {diagnostics['loss']:+.6f}  "
            f"lr {lr:.2e}"
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
    )


if __name__ == "__main__":
    main()
