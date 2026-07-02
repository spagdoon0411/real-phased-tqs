import numpy as np
import torch

from hamiltonian.hamiltonian import Hamiltonian
from hamiltonian.symmetries import Symmetry1D


class TransverseFieldIsing(Hamiltonian):
    """
    Transverse-field Ising Hamiltonian on a 1D chain:

        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    Parameter layout (ranged first, then static):
        phys_params[0]  h  transverse field strength  ranged_params[0]
        phys_params[1]  J  ZZ coupling                static_params[0]
    """

    name = "TFI-X"
    ranged_param_names = ("h",)
    static_param_names = ("J",)

    def __init__(
        self,
        system_dim_range: np.ndarray,
        static_params: np.ndarray,
        ranged_params: np.ndarray,
        periodic: bool = False,
        device: torch.device = torch.device("cpu"),
        symmetries: list[Symmetry1D] | None = None,
    ):
        super().__init__(
            system_dim_range=system_dim_range,
            static_params=static_params,
            ranged_params=ranged_params,
            periodic=periodic,
            device=device,
        )
        self.symmetries: list[Symmetry1D] = symmetries or []

    def observables(self) -> list[tuple[list[str], list[torch.Tensor], torch.Tensor]]:
        L = int(self.system_dim[0].item())
        h = self.phys_params[0]
        J = self.phys_params[1]

        left = torch.arange(L - 1, device=self.device)
        right = torch.arange(1, L, device=self.device)
        if self.periodic and L > 1:
            left = torch.cat([left, torch.tensor([L - 1], device=self.device)])
            right = torch.cat([right, torch.tensor([0], device=self.device)])
        bond_idx = torch.stack([left, right], dim=1)
        site_idx = torch.arange(L, device=self.device).unsqueeze(1)

        zz_coefs = -J * torch.ones(bond_idx.shape[0], device=self.device)
        x_coefs = -h * torch.ones(L, device=self.device)

        return [
            (["ZZ"], [zz_coefs], bond_idx),
            (["X"], [x_coefs], site_idx),
        ]
