import math
import time
from typing import Callable

import torch

from hamiltonian.hamiltonian import Hamiltonian
from hamiltonian.symmetries import Symmetry1D
from model.pauli_observables import _psi_along_samples, compute_grad, compute_observable
from model.tqs import TransformerQuantumState


def _draw_sym_samples(samples: torch.Tensor, sym_batch_size: int | None) -> torch.Tensor:
    """
    Draws a random subset of columns (configurations) from `samples` (L, batch) for
    use in the symmetry loss. If `sym_batch_size` is None or >= batch, returns all samples.
    """
    batch = samples.shape[1]
    if sym_batch_size is None or sym_batch_size >= batch:
        return samples
    idx = torch.randperm(batch, device=samples.device)[:sym_batch_size]
    return samples[:, idx]


def _symmetry_loss(
    model: TransformerQuantumState,
    samples: torch.Tensor,
    symmetries: list[Symmetry1D],
    phase_weight: float = 1.0,
) -> torch.Tensor:
    """
    Computes the generator-penalty symmetry loss:

        L_sym = (1/|S|) sum_{s in S} w_s * E_b[(ΔA_s)^2 + w_phi * (1 - cos Δφ_s)]

    where:
        ΔA_s(b)   = log|ψ(sb)| - log|ψ(b)|        (log-amplitude residual)
        Δφ_s(b)   = arg ψ(sb) - arg ψ(b) - angle_s (phase residual, wrapped via cosine)

    samples : (L, batch)  integer spin chains drawn for this loss term.
    """
    log_p, phases, _ = _psi_along_samples(model, samples)

    total = torch.zeros(1, device=model.device)
    weight_sum = 0.0
    for sym in symmetries:
        samples_g = sym.apply(samples)
        log_p_g, phases_g, _ = _psi_along_samples(model, samples_g)

        d_amp = 0.5 * (log_p_g - log_p)
        d_phase = phases_g - phases - sym.angle

        amp_loss = (d_amp ** 2).mean()
        phase_loss = (1.0 - torch.cos(d_phase)).mean()

        total = total + sym.weight * (amp_loss + phase_weight * phase_loss)
        weight_sum += sym.weight

    return total / weight_sum


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

    for step in range(n_steps):
        t0 = time.perf_counter()

        samples, sample_weight = sampler()

        autocast = torch.autocast(device_type=model.device.type, dtype=torch.bfloat16)

        with torch.no_grad(), autocast:
            Eloc = _local_energy(model, hamiltonian, samples, sample_weight)

        optimizer.zero_grad(set_to_none=True)
        sym_loss_val: torch.Tensor | None = None
        with autocast:
            energy_loss, _, _ = compute_grad(model, samples, sample_weight, Eloc)
            loss = energy_loss

            if symmetries and beta_schedule is not None:
                beta = beta_schedule(step)
                sym_samples = _draw_sym_samples(samples, sym_batch_size)
                sym_loss_val = beta * _symmetry_loss(model, sym_samples, symmetries, sym_phase_weight)
                loss = energy_loss + sym_loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if on_step is not None:
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
                "iter_time": time.perf_counter() - t0,
            }
            if sym_loss_val is not None:
                diagnostics["sym_loss"] = sym_loss_val.detach().item()
                diagnostics["beta"] = beta
            on_step(step, diagnostics)
