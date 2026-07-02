import math
import time
from typing import Callable

import torch

from hamiltonian.hamiltonian import Hamiltonian
from hamiltonian.symmetries import Symmetry1D
from model.loss_utils import _draw_sym_samples, _local_energy, _symmetry_loss, compute_grad
from model.tqs import TransformerQuantumState


def train(
    model: TransformerQuantumState,
    hamiltonian: Hamiltonian,
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    sampler: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    on_step: Callable[[int, dict], None] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    beta_schedule: Callable[[int], float] | None = None,
    sym_batch_size: int | None = None,
    sym_phase_weight: float = 1.0,
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

    `on_step` is called after each step with the step index and a dict of diagnostics
    (`energy`, `loss`, `variance`, and `sym_loss` when symmetries are active).
    `beta_schedule` maps the step index to a symmetry-loss coefficient β, analogous to
    `lr_lambda` in `LambdaLR`. Active only when `hamiltonian.symmetries` is non-empty.
    `sym_batch_size` is the number of randomly-drawn configurations used for the symmetry
    loss each step.
    `sym_phase_weight` weights the phase penalty relative to the amplitude penalty.
    """
    symmetries: list[Symmetry1D] = getattr(hamiltonian, "symmetries", [])

    try:
        for step in range(n_steps):
            t0 = time.perf_counter()

            hamiltonian.cycle_system_dim()
            hamiltonian.cycle_params()
            model.set_prefix(hamiltonian.phys_params, hamiltonian.system_dim)

            # Draw samples from quantum state via chosen sampling routine
            samples, sample_weight = sampler()

            # Define target datatypes
            autocast = torch.autocast(device_type=model.device.type, dtype=torch.bfloat16)

            # AUTOGRAD: Compute local energy samples, detached from computation graph in line with a
            # REINFORCE-style surrogate loss paradigm
            with torch.no_grad(), autocast:
                Eloc = _local_energy(model, hamiltonian, samples, sample_weight)

            # Clear .grad (adjoint) fields across reverse-mode graph
            optimizer.zero_grad(set_to_none=True)

            with autocast:
                # Compute base VMC energy loss
                energy_loss, _, _ = compute_grad(model, samples, sample_weight, Eloc)
                loss = energy_loss

                # Compute symmetrization penalties if symmetries must be enforced
                if symmetries and beta_schedule is not None:
                    # Compute contributions to the total loss from the symmetrization term
                    beta = beta_schedule(step)
                    sym_samples, sym_weight = _draw_sym_samples(samples, sample_weight, sym_batch_size)
                    sym_loss_val = beta * _symmetry_loss(model, sym_samples, sym_weight, symmetries, sym_phase_weight)
                    loss = energy_loss + sym_loss_val
                else:
                    # No symmetries; neither compute nor report a symmetrization loss value
                    sym_loss_val = None

            # AUTOGRAD: Populate adjoints upstream, limit grad norms, apply gradient updates to input
            # parameters, and perform scheduler bookkeeping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Report diagnostics
            if on_step is not None:
                tf = time.perf_counter()
                energy = (Eloc * sample_weight).sum().real
                variance = ((Eloc - energy) * (Eloc - energy).conj() * sample_weight).sum().real
                n_sites = int(hamiltonian.system_dim.prod().item())
                diagnostics: dict = {
                    "energy": energy.item(),
                    "energy_per_site": energy.item() / n_sites,
                    "energy_loss": energy_loss.detach().item(),
                    "loss": loss.detach().item(),
                    "variance": variance.item(),
                    "n_unique": samples.shape[1],
                    "iter_time": tf - t0,
                    "system_dim": hamiltonian.system_dim.tolist(),
                    "phys_params": hamiltonian.phys_params.tolist(),
                }
                if sym_loss_val is not None:
                    diagnostics["sym_loss"] = sym_loss_val.detach().item()
                    diagnostics["beta"] = beta
                on_step(step, diagnostics)
    except KeyboardInterrupt:
        # Avoid aborting any post-run cleanup.
        print(f"\nTraining interrupted at step {step}.")
