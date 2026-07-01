from abc import ABC, abstractmethod

import torch


class Symmetry1D(ABC):
    """
    Generator of a 1D chain symmetry. Subclasses define the configuration-space
    action and the target sector character χ(g), encoded as `angle = arg χ(g)`.

    The desired sector condition is ψ(g·b) = χ(g) ψ(b), equivalently:
        Δ log|ψ|(b)  = 0
        Δ arg ψ(b)   = angle

    where both residuals should vanish for a perfectly symmetric wavefunction.
    """

    def __init__(self, angle: float = 0.0, weight: float = 1.0):
        self.angle = angle
        self.weight = weight

    @abstractmethod
    def apply(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Apply the symmetry operation to a batch of spin chains.

        bits : (L, batch)  integer tensor of 0s and 1s.
        Returns a tensor of the same shape with the transformed configurations.
        """
        ...


class SpinFlip(Symmetry1D):
    """
    Global spin-flip F: b_i -> 1 - b_i for all i.

    Targets the even (p = +1) sector of the TFIM ground state, so χ(F) = +1
    and angle = arg(+1) = 0.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(angle=0.0, weight=weight)

    def apply(self, bits: torch.Tensor) -> torch.Tensor:
        return 1 - bits


class Reflection(Symmetry1D):
    """
    Chain reflection R: (b_0, ..., b_{L-1}) -> (b_{L-1}, ..., b_0).

    Targets the even (r = +1) reflection-parity sector, so χ(R) = +1
    and angle = 0.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(angle=0.0, weight=weight)

    def apply(self, bits: torch.Tensor) -> torch.Tensor:
        return bits.flip(0)


class Translation(Symmetry1D):
    """
    One-site cyclic translation T: (b_0, b_1, ..., b_{L-1}) -> (b_{L-1}, b_0, ..., b_{L-2}).

    Targets the k = 0 momentum sector, so χ(T) = e^{ik} = 1 and angle = 0.
    Only valid for periodic boundary conditions; applying it to an open chain
    enforces a soft periodicity bias.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(angle=0.0, weight=weight)

    def apply(self, bits: torch.Tensor) -> torch.Tensor:
        return torch.roll(bits, 1, dims=0)
