import torch


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
