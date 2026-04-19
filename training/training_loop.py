from typing import Callable

import torch

from hamiltonian.hamiltonian import Hamiltonian
from model.pauli_observables import compute_grad, compute_observable
from model.tqs import TransformerQuantumState


def _samples_from_buffer(model: TransformerQuantumState, buffer: torch.Tensor) -> torch.Tensor:
    """
    Extracts binary `(n, batch)` spin configurations from the one-hot spin slots of `buffer`
    populated by `model.sample_states`.
    """
    spin_one_hot = buffer[model.prefix_dim : model.prefix_dim + model.max_len, :, model.prefix_dim :]
    return spin_one_hot.argmax(dim=-1)


def _local_energy(
    model: TransformerQuantumState,
    hamiltonian: Hamiltonian,
    samples: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluates E_loc(x) = sum over every operator in every observable tuple, returning a
    complex tensor of shape `(batch,)`.
    """
    batch = samples.shape[1]
    Eloc = torch.zeros(batch, dtype=torch.complex64, device=model.device)
    for obs in hamiltonian.observables():
        per_string = compute_observable(model, samples, sample_weight, obs, batch_mean=False)
        for value in per_string:
            Eloc = Eloc + value.sum(dim=0)
    return Eloc


def train(
    model: TransformerQuantumState,
    hamiltonian: Hamiltonian,
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    num_samples_each_step: int,
    on_step: Callable[[int, dict], None] | None = None,
) -> None:
    """
    Runs a variational Monte Carlo training loop against `hamiltonian` using the model's
    tree-expansion sampler. Each step:

        1. Fills `model.init_spin_buffer()` via `model.sample_states`, yielding a batch of
           unique basis states together with per-leaf multinomial counts `freq`.
        2. Computes E_loc(x) as the sum of `compute_observable` values across every
           observable tuple returned by `hamiltonian.observables()`.
        3. Builds the REINFORCE-style surrogate loss via `compute_grad` and steps the
           optimizer.

    `on_step`, if provided, is called after each step with the step index and a dict of
    diagnostics (`energy`, `loss`, `variance`). This is the seam for injecting loggers.
    """

    for step in range(n_steps):
        buffer = model.init_spin_buffer()
        buffer, freq = model.sample_states(buffer, num_samples_each_step)

        samples = _samples_from_buffer(model, buffer)
        sample_weight = freq.to(torch.get_default_dtype())
        sample_weight = sample_weight / sample_weight.sum()

        with torch.no_grad():
            Eloc = _local_energy(model, hamiltonian, samples, sample_weight)

        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = compute_grad(model, samples, sample_weight, Eloc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if on_step is not None:
            energy = (Eloc * sample_weight).sum().real
            variance = ((Eloc - energy) * (Eloc - energy).conj() * sample_weight).sum().real
            on_step(
                step,
                {
                    "energy": energy.item(),
                    "loss": loss.detach().item(),
                    "variance": variance.item(),
                },
            )
