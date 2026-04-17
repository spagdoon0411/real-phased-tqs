import torch

from hamiltonian.hamiltonian import Hamiltonian


class TransverseFieldIsing(Hamiltonian):
    """
    Transverse-field Ising Hamiltonian on a 1D open chain:

        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    system_dim: (L,) chain length.
    phys_params: (h,) transverse-field strength.
    """

    def __init__(
        self,
        system_dim: torch.Tensor,
        phys_params: torch.Tensor,
        coupling: float = 1.0,
    ):
        super().__init__(n_params=1, system_dim=system_dim, phys_params=phys_params)
        self.coupling = coupling

    def observables(self) -> list[tuple[list[str], list[torch.Tensor], torch.Tensor]]:
        """
        Returns the Hamiltonian as a list of observable tuples in the convention used by
        `model.pauli_observables.compute_observable`. Each tuple is
        `(pauli_strs, coefs, spin_idx)`; multiple tuples are required here because `ZZ`
        and `X` act on different numbers of sites.
        """
        L = int(self.system_dim[0].item())
        h = self.phys_params[0]

        bond_idx = torch.stack([torch.arange(L - 1), torch.arange(1, L)], dim=1)
        site_idx = torch.arange(L).unsqueeze(1)

        zz_coefs = -self.coupling * torch.ones(L - 1)
        x_coefs = -h * torch.ones(L)

        return [
            (["ZZ"], [zz_coefs], bond_idx),
            (["X"], [x_coefs], site_idx),
        ]
