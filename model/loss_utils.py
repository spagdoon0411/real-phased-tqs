import torch

from hamiltonian.hamiltonian import Hamiltonian
from hamiltonian.symmetries import Symmetry1D
from model.pauli_observables import _psi_along_samples, compute_observable
from model.tqs import TransformerQuantumState


def compute_grad(model, samples, sample_weight, Eloc):
    """
    Builds the REINFORCE-style surrogate loss for the variational energy.

    Computes Gk = <<2Re[(Eloc-<<Eloc>>) Dk*]>>
    where Dk = d log Psi / d pk, pk is the NN parameter.

    Since the transformer wavefunction is normalized, <<Dk>> = 0 and Gk simplifies to
    Gk = <<2Re[Eloc Dk*]>>.

    Adapted from yuanhangzhang98/transformer_quantum_state.
    """
    log_probs, phases, _ = _psi_along_samples(model, samples)
    E_model = (Eloc * sample_weight).sum().detach()
    scale = torch.clamp(1 / E_model.abs(), max=5)
    E = Eloc - E_model

    loss = ((E.real * log_probs + E.imag * phases) * sample_weight).sum() * scale
    return loss, log_probs, phases


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


def _draw_sym_samples(
    samples: torch.Tensor,
    sample_weight: torch.Tensor,
    sym_batch_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draws a random subset of columns (configurations) from `samples` (L, batch), along with
    the corresponding entries of `sample_weight`, for use in the symmetry loss. The subset's
    weights are renormalized to sum to 1 so they remain a valid probability distribution over
    the reduced batch. If `sym_batch_size` is None or >= batch, returns the inputs unchanged.
    """
    batch = samples.shape[1]
    if sym_batch_size is None or sym_batch_size >= batch:
        return samples, sample_weight
    idx = torch.randperm(batch, device=samples.device)[:sym_batch_size]
    sub_weight = sample_weight[idx]
    sub_weight = sub_weight / sub_weight.sum()
    return samples[:, idx], sub_weight


def _symmetry_loss(
    model: TransformerQuantumState,
    samples: torch.Tensor,
    sample_weight: torch.Tensor,
    symmetries: list[Symmetry1D],
    phase_weight: float = 1.0,
) -> torch.Tensor:
    """
    Computes the generator-penalty symmetry loss:

        L_sym = (1/|S|) sum_{s in S} w_s * E_b[(ΔA_s/L)^2 + w_phi * (1 - cos(Δφ_s/L))]

    where:
        ΔA_s(b)   = log|ψ(sb)| - log|ψ(b)|        (log-amplitude residual)
        Δφ_s(b)   = arg ψ(sb) - arg ψ(b) - angle_s (phase residual, wrapped via cosine)

    Both residuals are extensive (summed over the L sites of the chain via `_psi_along_samples`),
    so they are divided by L to make the loss intensive, i.e. comparable in scale to
    `energy_loss` (which `compute_grad` normalizes by the extensive `E_model`) across the
    randomized system sizes drawn by `hamiltonian.cycle_system_dim()`.

    E_b[...] is taken over `samples` weighted by `sample_weight`, matching the weighting used
    elsewhere in the training loop (the samplers return deduplicated configurations with
    non-uniform multiplicity weights, so a plain average would misweight the residual).

    samples : (L, batch)  integer spin chains drawn for this loss term.
    sample_weight : (batch,)  normalized weight of each configuration.
    """
    L = samples.shape[0]
    log_p, phases, _ = _psi_along_samples(model, samples)

    total = torch.zeros(1, device=model.device)
    weight_sum = 0.0
    for sym in symmetries:
        samples_g = sym.apply(samples)
        log_p_g, phases_g, _ = _psi_along_samples(model, samples_g)

        d_amp = 0.5 * (log_p_g - log_p) / L
        d_phase = (phases_g - phases - sym.angle) / L

        amp_loss = (d_amp ** 2 * sample_weight).sum()
        phase_loss = ((1.0 - torch.cos(d_phase)) * sample_weight).sum()

        total = total + sym.weight * (amp_loss + phase_weight * phase_loss)
        weight_sum += sym.weight

    return total / weight_sum
