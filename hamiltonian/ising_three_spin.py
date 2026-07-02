import numpy as np
import torch

from hamiltonian.hamiltonian import Hamiltonian
from hamiltonian.symmetries import Symmetry1D


class IsingThreeSpin(Hamiltonian):
    """
    Cluster-Ising chain on a 1D ring with a 3-spin Z-X-Z interaction:

        H = -(J2/2) sum_i Z_i Z_{i+1} - (J3/2) sum_i Z_{i-1} X_i Z_{i+1} - (h/2) sum_i X_i

    with periodic boundary conditions (Z_{i+L} = Z_i, X_{i+L} = X_i).

    Parameter layout (ranged first, then static):
        phys_params[0]  h    transverse field strength     ranged_params[0]
        phys_params[1]  J2   ZZ coupling                    static_params[0]
        phys_params[2]  J3   ZXZ (cluster) coupling          static_params[1]
    """

    name = "IsingThreeSpin"
    ranged_param_names = ("h",)
    static_param_names = ("J2", "J3")

    def __init__(
        self,
        system_dim_range: np.ndarray,
        static_params: np.ndarray,
        ranged_params: np.ndarray,
        periodic: bool = True,
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
        J2 = self.phys_params[1]
        J3 = self.phys_params[2]

        left = torch.arange(L - 1, device=self.device)
        right = torch.arange(1, L, device=self.device)
        if self.periodic and L > 1:
            left = torch.cat([left, torch.tensor([L - 1], device=self.device)])
            right = torch.cat([right, torch.tensor([0], device=self.device)])
        bond_idx = torch.stack([left, right], dim=1)

        center = torch.arange(L, device=self.device)
        prev = (center - 1) % L
        nxt = (center + 1) % L
        if not self.periodic:
            interior = (center > 0) & (center < L - 1)
            center, prev, nxt = center[interior], prev[interior], nxt[interior]
        triple_idx = torch.stack([prev, center, nxt], dim=1)

        site_idx = torch.arange(L, device=self.device).unsqueeze(1)

        zz_coefs = -(J2 / 2) * torch.ones(bond_idx.shape[0], device=self.device)
        zxz_coefs = -(J3 / 2) * torch.ones(triple_idx.shape[0], device=self.device)
        x_coefs = -(h / 2) * torch.ones(L, device=self.device)

        return [
            (["ZZ"], [zz_coefs], bond_idx),
            (["ZXZ"], [zxz_coefs], triple_idx),
            (["X"], [x_coefs], site_idx),
        ]
