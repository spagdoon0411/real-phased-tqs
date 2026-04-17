import math

from hamiltonian.hamiltonian import Hamiltonian
import torch
from torch import nn

from torch.nn.functional import log_softmax


class SinusoidalPositionalEncoding(nn.Module):
    """
    A standard Attention is All You Need sinusoidal positional encoding.
    """

    # TODO: tune encoding base frequency

    def __init__(self, d_model: int, max_len: int, device: torch.device):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(max_len, device=device).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)].unsqueeze(1)


class Embedding(nn.Module):
    """
    A layer mapping parameter and spin tokens to a d_model-dimensional embedding
    space via a learnable linear transformation.
    """

    def __init__(
        self,
        d_model: int,
        n_dimensions: int,
        n_params: int,
        device: torch.device,
        spin_dim: int = 2,
    ):
        super().__init__()
        self.input_dimensions = n_dimensions + n_params + spin_dim
        self.linear = nn.Linear(self.input_dimensions, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Deembedding(nn.Module):
    """
    Performs a linear map to obtain logits corresponding to next-token
    probability distributions, then provides probability and possibly phase values.
    """

    def __init__(self, d_model: int, possible_values: int, device: torch.device):
        super().__init__()
        self.linear = nn.Linear(d_model, possible_values, device=device)
        self.possible_values = possible_values

    def forward(self, x: torch.Tensor, compute_phases: bool = False) -> torch.Tensor:
        logits = self.linear(x)  # (seq_len, batch_size, possible_values)

        log_probs = log_softmax(logits, dim=-1)

        if compute_phases:
            phases = self._softsign(logits)
            return log_probs, phases

        return log_probs


class TransformerQuantumState(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
        batch_size: int,
        hamiltonian: Hamiltonian,
        device: torch.device,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hamiltonian = hamiltonian
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.set_prefix(hamiltonian.phys_params, hamiltonian.system_dim)
        self.spin_dim = 2  # Spin 1/2 fermions
        self.prefix_dim = hamiltonian.phys_params.shape[0] + hamiltonian.system_dim.shape[0]
        self.max_seq_len = max_len + self.prefix_dim

        # Layers
        self.embedding = Embedding(
            d_model,
            n_dimensions=hamiltonian.system_dim.shape[0],
            n_params=hamiltonian.phys_params.shape[0],
            device=device,
        )
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, self.max_seq_len, device)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=False,
                device=device,
            ),
            num_layers=n_layers,
        )
        self.deembedding = Deembedding(d_model, possible_values=self.spin_dim, device=device)

        self.register_buffer(
            "mask",
            torch.nn.Transformer.generate_square_subsequent_mask(self.max_seq_len, device=device),
        )

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
        diagonal = torch.cat([system_dim.exp(), phys_params], dim=0)
        self.prefix = torch.diag(diagonal)  # (prefix_dim, prefix_dim)

        # TODO: create view into transformer mask on device

    def init_spin_buffer(
        self,
        spin_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Arranges dimension information, physical parameters, and initial spin tokens in a buffer large enough to
        accommodate the maximum expected spin chain plus a prefix.
        """
        spin_buffer = torch.zeros(
            self.max_len + self.prefix_dim,  # sequence
            self.batch_size,  # batch
            self.prefix_dim + self.spin_dim,  # embed
            device=self.device,
        )
        spin_buffer[: self.prefix_dim, :, : self.prefix_dim] = self.prefix.unsqueeze(1)
        if spin_tokens is not None:
            # Populate any initial spin data
            spin_buffer[:, :, self.prefix_dim : self.prefix_dim + self.spin_dim] = spin_tokens
        return spin_buffer

    @staticmethod
    def _softsign(x: torch.Tensor) -> torch.Tensor:
        """
        Definition borrowed from Zhang and Ventra (see yuanhangzhang98/transformer_quantum_state).
        """
        return 2 * torch.pi * (1 + x / (1 + x.abs()))

    def forward(
        self,
        buffer: torch.Tensor,
        compute_phases: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Maps a buffer of parameter and spin tokens to next-token log-probabilities. If `compute_phases` is True,
        also computes phase contributions to the probability amplitude of the final basis state.
        """
        buffer = buffer.to(self.device)  # (seq_len, batch_size, prefix_dim + spin_dim)
        buffer = self.embedding(buffer)  # (seq_len, batch_size, d_model)
        buffer = self.pos_encoding(buffer)
        buffer = self.transformer(buffer, mask=self.mask, is_causal=True)
        res = self.deembedding(buffer, compute_phases=compute_phases)
        return res  # Either log_probs or (log_probs, phases)

    def sample_states(self, buffer: torch.Tensor, start_at: int = 0) -> torch.Tensor:
        """
        Constructs states autoregressively, filling in the buffer provided.

        NOTE: assumes the spins present in the buffer are unique---otherwise, sampling
        and downstream computations are outright incorrect.
        """

        target_len = buffer.shape[0] - self.prefix_dim

        for i in range(start_at, target_len):
            # NOTE: values downstream of log_probs but upstream of random sampling are not relevant to the energy loss.
            # See (Gradient Estimation Using Stochastic Computation Graphs, Schulman et al. 2016). Detaching frees
            # memory that would have been used for adjoints.

            target_idx = i + self.prefix_dim  # Index of next token to be sampled
            log_probs = self.forward(buffer[:target_idx], compute_phases=False)
            probs = log_probs.detach()[-1, :, :].exp()  # log_softmax handles normalization
            sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            buffer[target_idx, torch.arange(self.batch_size), self.prefix_dim + sampled_idx] = 1

            # TODO: fit more chains and reduce redundant computations by forcing params and system dimensions
            # to be constant across the batch dimension. Add a mode to do this.

        return buffer

    def construct_wavefunction(self, log_probs: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Produces site-wide contributions to the wavefunction from log-probs. Sum over the sequence dimension to obtain
        one probability-amplitude value per batch entry (corresponding to a single basis state).

        log_probs: natural log of the probability of sampling each value (spin) that was sampled.

        phases: phase contributions to the probability amplitude of the final basis state, selected from the
        "phase distribution" using the value actually sampled.
        """

        return torch.sqrt(log_probs.exp()) * torch.exp(1j * phases)
