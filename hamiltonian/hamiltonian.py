import numpy as np
import scipy.sparse as sp
import torch


_PAULI = {
    "I": sp.csr_array(np.array([[1, 0], [0, 1]], dtype=np.complex64)),
    "X": sp.csr_array(np.array([[0, 1], [1, 0]], dtype=np.complex64)),
    "Y": sp.csr_array(np.array([[0, -1j], [1j, 0]], dtype=np.complex64)),
    "Z": sp.csr_array(np.array([[1, 0], [0, -1]], dtype=np.complex64)),
}


class Hamiltonian:
    """
    Base class for lattice Hamiltonians.

    System size and physical parameters are both specified as ranges. The model prefix
    encodes the current sampled values:

        [log(L) | ranged_param_values | static_params]

    `cycle_system_dim` draws a new integer L uniformly from [L_min, L_max].
    `cycle_params` draws new values uniformly from each parameter's [low, high] interval.
    `static_params` are fixed at construction and never randomized by `cycle_params`.
    """

    def __init__(
        self,
        system_dim_range: np.ndarray,
        static_params: np.ndarray,
        ranged_params: np.ndarray,
        periodic: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """
        system_dim_range : 1-D numpy array [L_min, L_max] (inclusive integers). Initialized
                           to the midpoint; cycle_system_dim samples uniformly from this range
                           and set_system_dim validates against it.
        static_params    : 1-D numpy array of fixed parameter values, shape (n_static,).
        ranged_params    : 2-D numpy array of shape (n_ranged, 2); each row is [low, high].
        """
        if system_dim_range.shape != (2,):
            raise ValueError(f"system_dim_range must have shape (2,), got {system_dim_range.shape}")
        if ranged_params.ndim != 2 or (ranged_params.shape[0] > 0 and ranged_params.shape[1] != 2):
            raise ValueError(f"ranged_params must have shape (n_ranged, 2), got {ranged_params.shape}")
        if static_params.ndim != 1:
            raise ValueError(f"static_params must be 1-D, got {static_params.shape}")

        self.device = device
        self._system_dim_range: np.ndarray = system_dim_range
        L_init = int((int(system_dim_range[0]) + int(system_dim_range[1])) // 2)
        self.system_dim = torch.tensor([float(L_init)], device=device)
        self.periodic = bool(periodic)

        self._static_params = torch.tensor(static_params, dtype=torch.float32, device=device)
        self._ranged_param_ranges: np.ndarray = ranged_params

        if ranged_params.shape[0] > 0:
            midpoints = (ranged_params[:, 0] + ranged_params[:, 1]) / 2.0
            self._current_ranged_params = torch.tensor(midpoints, dtype=torch.float32, device=device)
        else:
            self._current_ranged_params = torch.empty(0, dtype=torch.float32, device=device)

        self.n_params: int = ranged_params.shape[0] + static_params.shape[0]

    @property
    def phys_params(self) -> torch.Tensor:
        """Returns all current parameter values as [ranged_values | static_values]."""
        return torch.cat([self._current_ranged_params, self._static_params])

    def cycle_params(self) -> None:
        """Resample each ranged parameter uniformly from its [low, high] interval."""
        ranges = self._ranged_param_ranges
        if ranges.shape[0] == 0:
            return
        new_vals = np.random.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)
        self._current_ranged_params = torch.tensor(new_vals, device=self.device)

    def set_ranged_params(self, values: np.ndarray) -> None:
        """
        Explicitly set current ranged parameter values.

        Raises ValueError if any value falls outside its declared [low, high] range.
        """
        ranges = self._ranged_param_ranges
        n_ranged = ranges.shape[0]
        if values.shape != (n_ranged,):
            raise ValueError(f"Expected shape ({n_ranged},), got {values.shape}")
        out_of_range = (values < ranges[:, 0]) | (values > ranges[:, 1])
        if out_of_range.any():
            bad = np.where(out_of_range)[0]
            raise ValueError(
                f"Parameter(s) at index {bad.tolist()} out of declared range: "
                f"values={values[bad].tolist()}, ranges={ranges[bad].tolist()}"
            )
        self._current_ranged_params = torch.tensor(values.astype(np.float32), device=self.device)

    def set_phys_params(self, phys_params: torch.Tensor) -> None:
        """
        Set all physical parameters from a combined tensor [ranged_values | static_values].

        The ranged portion is validated against the declared ranges.
        """
        if phys_params.ndim > 1:
            raise ValueError(f"Physical parameters must be 1-D, got {phys_params.shape}")
        if phys_params.shape[0] != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {phys_params.shape[0]}")

        n_ranged = self._ranged_param_ranges.shape[0]
        if n_ranged > 0:
            self.set_ranged_params(phys_params[:n_ranged].detach().cpu().numpy())
        if self._static_params.shape[0] > 0:
            self._static_params = phys_params[n_ranged:].to(self.device)

    def cycle_system_dim(self) -> None:
        """Sample a new system size uniformly at random from the integer range [L_min, L_max]."""
        lo, hi = int(self._system_dim_range[0]), int(self._system_dim_range[1])
        L = int(np.random.randint(lo, hi + 1))
        self.system_dim = torch.tensor([float(L)], device=self.device)

    def set_system_dim(self, system_dim: torch.Tensor) -> None:
        if system_dim.ndim > 1:
            raise ValueError(f"System dimensions must be 1-dimensional, got {system_dim.shape}")
        if system_dim.shape[0] != 1:
            raise NotImplementedError("Systems with more than one dimension are not yet supported")
        L = int(system_dim[0].item())
        lo, hi = int(self._system_dim_range[0]), int(self._system_dim_range[1])
        if not (lo <= L <= hi):
            raise ValueError(f"system_dim {L} is outside the declared range [{lo}, {hi}]")
        self.system_dim = system_dim.to(self.device)

    def param_str(self) -> str:
        """Human-readable summary of the current system size and parameter values."""
        L = int(self.system_dim[0].item())
        params = [f"{v:.4f}" for v in self.phys_params.tolist()]
        return f"n={L}  params=[{', '.join(params)}]"

    def observables(self) -> list[tuple[list[str], list[torch.Tensor], torch.Tensor]]:
        """
        Returns the Hamiltonian as a list of observable tuples in the convention used by
        `model.pauli_observables.compute_observable`. Each tuple is
        `(pauli_strs, coefs, spin_idx)` with `spin_idx` of shape `(n_op, n_site)`.
        """
        raise NotImplementedError

    def sparse_matrix(self) -> sp.csr_array:
        """
        Materializes H on the full 2^L-dimensional computational basis by summing Kronecker
        products of Pauli matrices against the observable tuples returned by `observables()`.
        Returns a scipy CSR array of shape (2^L, 2^L).
        """
        L = int(self.system_dim[0].item())
        N = 1 << L

        H = sp.csr_array((N, N), dtype=np.complex64)

        for pauli_strs, coefs, spin_idx in self.observables():
            n_op = spin_idx.shape[0]
            for pauli_str, coef in zip(pauli_strs, coefs):
                if not isinstance(coef, torch.Tensor):
                    coef = torch.tensor(coef)
                coef = coef.reshape(-1).to(torch.complex64).numpy()
                for op_idx in range(n_op):
                    acting = dict(zip(spin_idx[op_idx].tolist(), pauli_str))
                    term = _PAULI[acting.get(0, "I")]
                    for site in range(1, L):
                        term = sp.kron(term, _PAULI[acting.get(site, "I")], format="csr")
                    H = H + coef[op_idx] * term

        return H
