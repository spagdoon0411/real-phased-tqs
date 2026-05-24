from typing import Callable

import torch

from hamiltonian.hamiltonian import Hamiltonian
from model.pauli_observables import compute_grad, compute_observable
from model.tqs import TransformerQuantumState


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
    sampler: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    on_step: Callable[[int, dict], None] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """
    Runs a variational Monte Carlo training loop with energies computed via `hamiltonian`
    using the supplied `sampler` callable. Each step:

        1. Calls `sampler()` to obtain `(samples, sample_weight)` of shapes `(n, batch)` and
           `(batch,)`, where weights are already normalized.
        2. Computes E_loc(x) as the sum of `compute_observable` values across every
           observable tuple returned by `hamiltonian.observables()`.
        3. Builds the REINFORCE-style surrogate loss via `compute_grad` and steps the
           optimizer.

    `on_step`, if provided, is called after each step with the step index and a dict of
    diagnostics (`energy`, `loss`, `variance`). This is the seam for injecting loggers.
    """

    for step in range(n_steps):
        samples, sample_weight = sampler()

        with torch.no_grad():
            Eloc = _local_energy(model, hamiltonian, samples, sample_weight)

        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = compute_grad(model, samples, sample_weight, Eloc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if on_step is not None:
            energy = (Eloc * sample_weight).sum().real
            variance = ((Eloc - energy) * (Eloc - energy).conj() * sample_weight).sum().real
            n_sites = int(hamiltonian.system_dim.prod().item())
            on_step(
                step,
                {
                    "energy": energy.item(),
                    "energy_per_site": energy.item() / n_sites,
                    "loss": loss.detach().item(),
                    "variance": variance.item(),
                },
            )
