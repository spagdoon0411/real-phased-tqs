import torch

from hamiltonian.hamiltonian import Hamiltonian


class TransverseFieldIsingY(Hamiltonian):
    """
    Transverse-field Ising Hamiltonian on a 1D open chain with a Y-direction field:

        H = -J sum_i Z_i Z_{i+1} - h sum_i Y_i

    system_dim: (L,) chain length.
    phys_params: (h,) transverse-field strength.
    """

    def __init__(
        self,
        system_dim: torch.Tensor,
        phys_params: torch.Tensor,
        coupling: float = 1.0,
        periodic: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            n_params=1,
            system_dim=system_dim,
            phys_params=phys_params,
            periodic=periodic,
            device=device,
        )
        self.coupling = coupling

    def observables(self) -> list[tuple[list[str], list[torch.Tensor], torch.Tensor]]:
        L = int(self.system_dim[0].item())
        h = self.phys_params[0]

        left = torch.arange(L - 1, device=self.device)
        right = torch.arange(1, L, device=self.device)
        if self.periodic and L > 1:
            left = torch.cat([left, torch.tensor([L - 1], device=self.device)])
            right = torch.cat([right, torch.tensor([0], device=self.device)])
        bond_idx = torch.stack([left, right], dim=1)
        site_idx = torch.arange(L, device=self.device).unsqueeze(1)

        zz_coefs = -self.coupling * torch.ones(bond_idx.shape[0], device=self.device)
        y_coefs = -h * torch.ones(L, device=self.device)

        return [
            (["ZZ"], [zz_coefs], bond_idx),
            (["Y"], [y_coefs], site_idx),
        ]
