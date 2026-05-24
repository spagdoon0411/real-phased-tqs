import torch
import torch.nn.functional as F

"""
These functions are taken from yuanhangzhang98/transformer_quantum_state.
"""


def _psi_along_samples(model, samples):
    """
    Runs the model's forward pass over a buffer built from binary `samples` of shape (n, batch),
    selects the next-token log-probabilities and phases corresponding to the sampled spins,
    sums them over the chain, and reconstructs psi via `model.construct_wavefunction`.

    Returns (log_probs, phases, psi), each with batch dimension (batch,).
    """
    n, batch = samples.shape
    spin_idx = samples.to(torch.int64)
    spin_tokens = F.one_hot(spin_idx, num_classes=model.spin_dim).to(torch.get_default_dtype())

    buffer = torch.zeros(
        n + model.prefix_dim,
        batch,
        model.prefix_dim + model.spin_dim,
        device=model.device,
    )
    buffer[: model.prefix_dim, :, : model.prefix_dim] = model.prefix.unsqueeze(1)
    buffer[model.prefix_dim : model.prefix_dim + n, :, model.prefix_dim :] = spin_tokens

    log_probs, phases = model.forward(buffer, compute_phases=True)
    log_probs = log_probs[model.prefix_dim - 1 : model.prefix_dim - 1 + n]
    phases = phases[model.prefix_dim - 1 : model.prefix_dim - 1 + n]

    n_idx = torch.arange(n, device=model.device).unsqueeze(1)
    batch_idx = torch.arange(batch, device=model.device).unsqueeze(0)
    log_probs = log_probs[n_idx, batch_idx, spin_idx].sum(dim=0)
    phases = phases[n_idx, batch_idx, spin_idx].sum(dim=0)

    psi = model.construct_wavefunction(log_probs, phases)
    return log_probs, phases, psi


def compute_grad(model, samples, sample_weight, Eloc):
    """
    Parameters
    ----------
    model : The transformer model
    samples : (n, batch)
        batched sample from the transformer distribution
    sample_weight: (batch, )
        weight for each sample
    Eloc : (batch, ), complex tensor
        local energy estimator

    Returns
    -------
    None.

    Computes Gk = <<2Re[(Eloc-<<Eloc>>) Dk*]>>
    where Dk = d log Psi / d pk, pk is the NN parameter

    Note: since the transformer wavefunction is normalized, we should have
    <<Dk>> = 0, and Gk has the simplified expression
    Gk = <<2Re[Eloc Dk*]>>
    TODO: Check this

    """

    log_probs, phases, _ = _psi_along_samples(model, samples)
    E_model = (Eloc * sample_weight).sum().detach()  # (1, )
    scale = torch.clamp(1 / E_model.abs(), max=5)
    E = Eloc - E_model  # (batch, )

    loss = ((E.real * log_probs + E.imag * phases) * sample_weight).sum() * scale
    return loss, log_probs, phases


@torch.no_grad()
def compute_observable(model, samples, sample_weight, observable, batch_mean=True):
    """
    Parameters
    ----------
    model : The transformer model
    samples : (n_param+n, batch, input_dim)
        samples drawn from the wave function
    sample_weight: (batch, )
        weight for each sample
    observable: tuple,
        (['XX', 'YY', 'ZZ'], [coef_XX, coef_YY, coef_ZZ], spin_idx)
        grouping up operators that act on the same indices to speed up
        (e.g., interaction in the Heisenberg model)
        pauli_str: string made up of 'X', 'Y', or 'Z', Pauli matrices
        coef: (1, ), (n_op, ) or (n_op, batch), coefficient of operator
        spin_idx: (n_op, n_site), indices that the Pauli operators act on
    batch_mean: bool, whether return the mean value over the batch or not

    Returns
    -------
    O: list, [value_XX, value_YY, value_ZZ], values of computed observables
        value:   (n_op, ) if batch_mean is True
            else (n_op, batch)

    Computes the expectation of observables, specified with Pauli strings
    """
    pauli_strs, coefs, spin_idx = observable
    n_type = len(pauli_strs)
    # ord('X')=88, maps 'X' to 0, 'Y' to 1, 'Z' to 2
    pauli_num = torch.tensor([[ord(c) - 88 for c in str_i] for str_i in pauli_strs], device=model.device)  # (n_type, n_site)
    X_sites = pauli_num == 0
    Y_sites = pauli_num == 1
    Z_sites = pauli_num == 2
    flip_sites = X_sites | Y_sites  # (n_type, n_site)
    phase_sites = Y_sites | Z_sites  # (n_type, n_site)

    # group up the computations that can be done at the same time
    # example: XX and YY share flip, while YY and ZZ share phase up to a constant
    flip_sites, inv_flip_idx = torch.unique(flip_sites, dim=0, return_inverse=True)  # (n_unique, n_site)
    phase_sites, inv_phase_idx = torch.unique(phase_sites, dim=0, return_inverse=True)  # (n_unique, n_site)
    flip_results = []
    phase_results = []

    # Y = -i Z X
    # compute phase like Z, flip like X, then account for the additional -i
    Y_count = Y_sites.sum(dim=1)  # (n_type, )
    Y_phase = torch.tensor([1, -1j, -1, 1j], device=model.device)[Y_count % 4]

    if flip_sites.any():
        log_amp, log_phase, _ = _psi_along_samples(model, samples)

    if phase_sites.any():
        spin_pm = (1 - 2 * samples).to(torch.get_default_dtype())  # +-1, (n, batch)

    for flip_sites_i in flip_sites:
        if flip_sites_i.any():
            flip_idx = spin_idx.T[flip_sites_i].T  # (n_op, n_flip)
            psixp_over_psix = compute_flip(model, samples, flip_idx, log_amp, log_phase)  # (n_op, batch)
            flip_results.append(psixp_over_psix)
        else:
            flip_results.append(torch.ones(1, device=model.device))

    for phase_sites_i in phase_sites:
        if phase_sites_i.any():
            phase_idx = (spin_idx.T[phase_sites_i]).T
            Oxxp = compute_phase(spin_pm, phase_idx)  # (n_op, batch)
            phase_results.append(Oxxp)
        else:
            phase_results.append(torch.ones(1, device=model.device))

    results = []
    for i in range(n_type):
        coef = coefs[i]
        if not isinstance(coef, torch.Tensor):
            coef = torch.tensor(coef)
        if len(coef.shape) < 2:
            coef = coef.reshape(-1, 1)
        result_i = Y_phase[i] * phase_results[inv_phase_idx[i]] * flip_results[inv_flip_idx[i]]  # (n_op, batch)
        result_i = coef * result_i  # (n_op, batch)
        results.append(result_i)

    if batch_mean:
        results = [(sample_weight * result_i).mean(dim=1) for result_i in results]

    return results


def compute_flip(model, samples, flip_idx, log_amp, log_phase):
    """
    Parameters
    ----------
    model: the transformer model
    samples : Tensor, (seq, n_samples)
        samples drawn from the wave function
    flip_idx : Tensor, (n_op, n_flip)
        indices with either X or Y acting on it
    log_amp : Tensor, (n_samples,)
        sum of log-amplitudes for the original samples
    log_phase : Tensor, (n_samples,)
        sum of phases for the original samples

    Returns
    -------
    psi(x') / psi(x) : (n_op, n_samples)
    """

    seq, n_samples = samples.shape
    n_op, n_flip = flip_idx.shape

    samples_flipped = samples.expand(n_op, -1, -1).transpose(0, 1).clone()  # (seq, n_op, n_samples)
    flip_mask = torch.zeros_like(samples_flipped, dtype=torch.bool)
    #         (n_op, n_flip)        (n_op, 1)  indices selected: (n_op, n_flip, n_samples)
    flip_mask[flip_idx, torch.arange(n_op, device=samples.device).unsqueeze(1), :] = 1
    samples_flipped[flip_mask] = 1 - samples_flipped[flip_mask]

    log_amp_1, log_phase_1, _ = _psi_along_samples(model, samples_flipped.reshape(seq, n_op * n_samples))
    log_amp_1 = log_amp_1.reshape(n_op, n_samples)
    log_phase_1 = log_phase_1.reshape(n_op, n_samples)

    return (((log_amp_1 - log_amp) + 1j * (log_phase_1 - log_phase)) / 2).exp()


def compute_phase(spin_pm, phase_idx):
    """
    Parameters
    ----------
    spin_pm : Tensor, (seq, n_samples)
        +-1, sampled spin configurations
    phase_idx : Tensor, (n_op, n_phase)
        indices with either Y or Z acting on it
        additional -i and spin flip for Y are computed outside this function

    Returns
    -------
    O_{x, x'} : (n_op, n_samples)
        where x is given
        O_loc(x) = O_{x, x'} psi(x') / psi(x)
    """
    seq, n_samples = spin_pm.shape
    spin_pm_relevant = spin_pm[phase_idx.unsqueeze(-1), torch.arange(n_samples)]  # (n_op, n_phase, n_samples), +-1
    return spin_pm_relevant.prod(dim=1)  # (n_op, n_samples)
