import torch

from hamiltonian.transverse_field_ising import TransverseFieldIsing
from model.tqs import TransformerQuantumState
from training.training_loop import train


def main() -> None:
    device = torch.device("cpu")

    L = 6
    h = 0.5
    J = 1.0

    hamiltonian = TransverseFieldIsing(
        system_dim=torch.tensor([float(L)]),
        phys_params=torch.tensor([h]),
        coupling=J,
    )

    model = TransformerQuantumState(
        d_model=32,
        max_len=L,
        batch_size=1 << L,
        hamiltonian=hamiltonian,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def log(step: int, diagnostics: dict) -> None:
        print(
            f"step {step:4d}  "
            f"energy {diagnostics['energy']:+.6f}  "
            f"variance {diagnostics['variance']:.6f}  "
            f"loss {diagnostics['loss']:+.6f}"
        )

    train(
        model=model,
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        n_steps=200,
        num_samples_each_step=100,
        on_step=log,
    )


if __name__ == "__main__":
    main()
