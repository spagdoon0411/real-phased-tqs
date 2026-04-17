import torch


_PAULI = {
    "I": torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
}


class Hamiltonian:
    def __init__(self, n_params: int, system_dim: torch.Tensor, phys_params: torch.Tensor):
        self.n_params = n_params
        self.system_dim = system_dim
        self.phys_params = phys_params

    def set_system_dim(self, system_dim: torch.Tensor) -> None:
        if system_dim.ndim > 1:
            raise ValueError(f"System dimensions must be 1-dimensional, got {system_dim.shape}")

        if system_dim.shape[0] != 1:
            raise NotImplementedError(f"Systems with more than one dimension are not yet supported")

        self.system_dim = system_dim

    def set_phys_params(self, phys_params: torch.Tensor) -> None:
        if phys_params.ndim > 1:
            raise ValueError(f"Physical parameters must be 1-dimensional, got {phys_params.shape}")

        if phys_params.shape[0] != self.n_params:
            raise ValueError(f"Expected {self.n_params} physical parameters, got {phys_params.shape[0]}")

        self.phys_params = phys_params

    def observables(self) -> list[tuple[list[str], list[torch.Tensor], torch.Tensor]]:
        """
        Returns the Hamiltonian as a list of observable tuples in the convention used by
        `model.pauli_observables.compute_observable`. Each tuple is
        `(pauli_strs, coefs, spin_idx)` with `spin_idx` of shape `(n_op, n_site)`.
        """
        raise NotImplementedError

    def sparse_matrix(self) -> torch.Tensor:
        """
        Materializes H on the full 2^L-dimensional computational basis by summing Kronecker
        products of Pauli matrices against the observable tuples returned by `observables()`.
        Returns a coalesced sparse COO tensor of shape (2^L, 2^L).
        """
        L = int(self.system_dim[0].item())
        N = 1 << L

        H = torch.zeros((N, N), dtype=torch.complex64)

        for pauli_strs, coefs, spin_idx in self.observables():
            n_op = spin_idx.shape[0]
            for pauli_str, coef in zip(pauli_strs, coefs):
                if not isinstance(coef, torch.Tensor):
                    coef = torch.tensor(coef)
                coef = coef.reshape(-1).to(torch.complex64)
                for op_idx in range(n_op):
                    acting = dict(zip(spin_idx[op_idx].tolist(), pauli_str))
                    mats = [_PAULI[acting.get(site, "I")] for site in range(L)]
                    term = mats[0]
                    for m in mats[1:]:
                        term = torch.kron(term, m)
                    H = H + coef[op_idx] * term

        return H.to_sparse().coalesce()
