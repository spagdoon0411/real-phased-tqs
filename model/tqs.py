from hamiltonian.hamiltonian import Hamiltonian
import torch
from torch import nn


class TransformerQuantumState(nn.Module):
    def __init__(
        self,
        expected_max_system_size: int,
        device: torch.device,
        hamiltonian: Hamiltonian,
    ):
        self.hamiltonian = hamiltonian
        self.device = device
        self.set_prefix(self.hamiltonian.phys_params, self.hamiltonian.system_dim)

        # TODO: init max transformer mask on device here

    def _arrange_prefix(
        self,
        phys_params: torch.Tensor,
        system_dim: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produces the tensor copied into the spin buffer
        """
        diagonal = torch.cat([system_dim, phys_params], dim=0)
        return torch.diag(diagonal)

    def set_prefix(
        self,
        phys_params: torch.Tensor,
        system_dim: torch.Tensor,
    ) -> None:
        """
        Selects a point in the Hamiltonian's physical parameter space (e.g., a magnetic field strength).
        """
        self.hamiltonian.set_phys_params(phys_params)
        self.hamiltonian.set_system_dim(system_dim)
        self.prefix = self._arrange_prefix(self.hamiltonian.phys_params, self.hamiltonian.system_dim)

        # TODO: create view into transformer mask on device

    def forward(
        self,
        spin_buffer: torch.Tensor,
        compute_phases: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Maps a buffer of parameter and spin tokens to next-token log-probabilities and, optionally, spin-phase
        phase contributions.
        """
        pass
