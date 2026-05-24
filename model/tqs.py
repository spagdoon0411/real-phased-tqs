import math
import warnings

from hamiltonian.hamiltonian import Hamiltonian
import torch
from torch import nn
from torch.distributions import Binomial

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
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * math.sqrt(self.d_model)


class Deembedding(nn.Module):
    """
    Performs a linear map to obtain logits corresponding to next-token
    probability distributions, then provides probability and possibly phase values.
    """

    def __init__(self, d_model: int, possible_values: int, device: torch.device):
        super().__init__()
        self.linear = nn.Linear(d_model, possible_values, device=device)
        self.possible_values = possible_values

    @staticmethod
    def _softsign(x: torch.Tensor) -> torch.Tensor:
        """
        Definition borrowed from Zhang and Ventra (see yuanhangzhang98/transformer_quantum_state).
        """
        return 2 * torch.pi * (1 + x / (1 + x.abs()))

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
        hamiltonian: Hamiltonian,
        device: torch.device,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hamiltonian = hamiltonian
        self.device = device
        self.max_len = max_len
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
                dim_feedforward=dim_feedforward if dim_feedforward is not None else 4 * d_model,
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

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Re-initializes only the embedding and head linears. The TransformerEncoder
        # layers are left at PyTorch defaults (Kaiming-uniform) to avoid overscaling
        # residual contributions at init.
        nn.init.normal_(self.embedding.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.embedding.linear.bias)

        nn.init.xavier_uniform_(self.deembedding.linear.weight)
        nn.init.zeros_(self.deembedding.linear.bias)

        # def _init(module: nn.Module) -> None:
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)
        #     elif isinstance(module, nn.LayerNorm):
        #         nn.init.ones_(module.weight)
        #         nn.init.zeros_(module.bias)
        #     elif isinstance(module, nn.MultiheadAttention):
        #         if module.in_proj_weight is not None:
        #             nn.init.xavier_uniform_(module.in_proj_weight)
        #         if module.in_proj_bias is not None:
        #             nn.init.zeros_(module.in_proj_bias)
        #     elif isinstance(module, nn.Embedding):
        #         nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #
        # self.apply(_init)

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
        diagonal = torch.cat([system_dim.log(), phys_params], dim=0)
        self.prefix = torch.diag(diagonal).to(self.device)  # (prefix_dim, prefix_dim)

        # TODO: create view into transformer mask on device

    def init_spin_buffer(
        self,
        batch_size: int,
        spin_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Arranges dimension information, physical parameters, and initial spin tokens in a buffer large enough to
        accommodate the maximum expected spin chain plus a prefix.

        batch_size is the number of samples this buffer will hold. (E.g., can be used to initialize a microbatch
        or a buffer holding a total collection of samples).
        """
        spin_buffer = torch.zeros(
            self.max_len + self.prefix_dim,  # sequence
            batch_size,  # batch
            self.prefix_dim + self.spin_dim,  # embed
            device=self.device,
        )
        spin_buffer[: self.prefix_dim, :, : self.prefix_dim] = self.prefix.unsqueeze(1)
        if spin_tokens is not None:
            # Populate any initial spin data
            spin_buffer[:, :, self.prefix_dim : self.prefix_dim + self.spin_dim] = spin_tokens
        return spin_buffer

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

    def sample_states(self, num_walkers: int, sample_buffer_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Allocates a buffer of width `sample_buffer_size` and populates unique states as a binary tree
        into it, doubling the number of populated entries at each step. The root receives `num_walkers`
        multinomial draws; at every node the [zeros, ones] counts are propagated to the children via a
        Binomial draw against the next-token distribution. Returns `(samples, weights)` of shapes
        `(max_len, sample_buffer_size)` and `(sample_buffer_size,)`, with weights already normalized.
        """

        buffer = self.init_spin_buffer(batch_size=sample_buffer_size)
        target_len = self.max_len
        freq = torch.zeros(sample_buffer_size, device=self.device, dtype=torch.float32)
        freq[0] = float(num_walkers)
        active_n = 1

        for i in range(target_len):
            # NOTE: values downstream of log_probs but upstream of random sampling are not relevant to the energy loss.
            # See (Gradient Estimation Using Stochastic Computation Graphs, Schulman et al. 2016). Detaching frees
            # memory that would have been used for adjoints.

            # TODO: how do we know that allocator churn is not terrible when the forward pass is called repeatedly?
            # The active view into the buffer grows along both the sequence and batch dimensions across iterations.
            #
            # At the very least, force allocation of enough memory the first time around by passing the entire tensor
            # through the forward pass and ignoring sequence positions that have not been reached yet.

            target_idx = i + self.prefix_dim  # Index of next token to be sampled
            log_probs = self.forward(buffer, compute_phases=False)
            probs = log_probs.detach()[target_idx - 1, :active_n, :].exp()  # log_softmax handles normalization
            count_ones = Binomial(total_count=freq[:active_n], probs=probs[:, 1]).sample().long()
            count_zeros = freq[:active_n] - count_ones

            if 2 * active_n <= sample_buffer_size:
                # Place the 1-branch continuations in fresh slots at the bottom so existing entries don't churn.
                buffer[:target_idx, active_n : 2 * active_n, :] = buffer[:target_idx, :active_n, :]
                buffer[target_idx, :active_n, self.prefix_dim] = 1
                buffer[target_idx, active_n : 2 * active_n, self.prefix_dim + 1] = 1
                freq[:active_n] = count_zeros
                freq[active_n : 2 * active_n] = count_ones
                active_n *= 2
            else:
                # Buffer is full; stop branching and sample a single token per active entry.
                # Scale freq by the probability of the chosen spin so that the final weights
                # reflect P(full configuration), not just P(first floor(log2(batch)) spins).
                picks = torch.multinomial(probs, num_samples=1).squeeze(-1)
                idx = torch.arange(active_n, device=self.device)
                buffer[target_idx, idx, self.prefix_dim + picks] = 1
                # buffer[target_idx, torch.arange(active_n), self.prefix_dim + picks] = 1
                # chosen_probs = probs[torch.arange(active_n), picks]
                # freq[:active_n] = (freq[:active_n].float() * chosen_probs).long().clamp(min=1)

            # TODO: fit more chains and reduce redundant computations by forcing params and system dimensions
            # to be constant across the batch dimension. Add a mode to do this.

        samples = buffer[self.prefix_dim : self.prefix_dim + self.max_len, :, self.prefix_dim :].argmax(dim=-1)
        return samples, freq / freq.sum()

    def sample_iid_microbatches(
        self,
        num_walkers: int,
        microbatch_size: int,
        sample_buffer_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draws `num_walkers` IID autoregressive samples from the model in chunks of `microbatch_size`
        per forward pass, writes them into a pre-allocated buffer of width `sample_buffer_size`,
        deduplicates the populated prefix, and returns the unique configurations together with
        normalized multiplicity weights. Emits a warning if `num_walkers` is not divisible by
        `microbatch_size`.
        """
        if num_walkers <= 0:
            raise ValueError(f"num_walkers must be positive, got {num_walkers}")
        if microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
        if sample_buffer_size < num_walkers:
            raise ValueError(f"sample_buffer_size={sample_buffer_size} must be >= num_walkers={num_walkers}")
        if sample_buffer_size % microbatch_size != 0:
            raise ValueError(
                f"sample_buffer_size={sample_buffer_size} must be a multiple of microbatch_size={microbatch_size}"
            )

        if num_walkers % microbatch_size != 0:
            warnings.warn(
                f"num_walkers={num_walkers} is not divisible by microbatch_size={microbatch_size}; "
                f"the last microbatch will be partially used",
                stacklevel=2,
            )

        num_microbatches_needed = (num_walkers + microbatch_size - 1) // microbatch_size
        microbatch_buffer = self.init_spin_buffer(batch_size=microbatch_size)
        row_idx = torch.arange(microbatch_size, device=self.device)
        sample_buffer = torch.empty(
            self.max_len,
            sample_buffer_size,
            dtype=torch.int64,
            device=self.device,
        )
        for i in range(num_microbatches_needed):
            self._sample_microbatch_into(
                microbatch_buffer, row_idx, sample_buffer[:, i * microbatch_size : (i + 1) * microbatch_size]
            )
        return self._weighted_unique(sample_buffer[:, :num_walkers])

    def _weighted_unique(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deduplicates chain samples and returns `(samples_unique, weights)` where weights are
        the normalized multiplicities of each unique configuration.
        """
        samples_unique, counts = self._dedup_chains(samples)
        weights = counts.to(torch.get_default_dtype())
        return samples_unique, weights / weights.sum()

    def _sample_microbatch_into(
        self,
        microbatch_buffer: torch.Tensor,
        row_idx: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        """
        Fills the spin region of `microbatch_buffer` with one microbatch of IID autoregressive samples
        and writes the resulting `(max_len, microbatch_size)` chain tensor into `destination` in place.
        """
        spin_region = slice(self.prefix_dim, self.prefix_dim + self.max_len)
        microbatch_buffer[spin_region, :, self.prefix_dim :] = 0
        for i in range(self.max_len):
            self._sample_next_token(microbatch_buffer, i + self.prefix_dim, row_idx)
        torch.argmax(microbatch_buffer[spin_region, :, self.prefix_dim :], dim=-1, out=destination)

    def _sample_next_token(self, buffer: torch.Tensor, target_idx: int, row_idx: torch.Tensor) -> None:
        """
        Runs one forward pass, samples a single spin per row from the next-token distribution at
        `target_idx`, and writes the one-hot result into `buffer` in place.
        """
        log_probs = self.forward(buffer, compute_phases=False)
        probs = log_probs.detach()[target_idx - 1, :, :].exp()
        picks = torch.multinomial(probs, num_samples=1).squeeze(-1)
        buffer[target_idx, row_idx, self.prefix_dim + picks] = 1

    @staticmethod
    def _dedup_chains(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deduplicates `(n, N)` binary chain samples by bit-packing each chain into an int64 code
        and grouping with `torch.unique`. Returns `(samples_unique, counts)` of shapes `(n, n_unique)`
        and `(n_unique,)`.
        """
        n, N = samples.shape
        if n > 64:
            raise ValueError(f"_dedup_chains requires n <= 64 for int64 bit-packing, got n={n}")
        pow2 = 1 << torch.arange(n, device=samples.device, dtype=torch.int64)
        codes = (samples.to(torch.int64) * pow2.unsqueeze(1)).sum(dim=0)
        unique_codes, inverse, counts = torch.unique(codes, return_inverse=True, return_counts=True)
        first_pos = torch.empty_like(unique_codes, dtype=torch.long)
        first_pos.scatter_(0, inverse, torch.arange(N, device=samples.device))
        return samples[:, first_pos], counts

    def construct_wavefunction(self, log_probs: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Produces site-wide contributions to the wavefunction from log-probs. Sum over the sequence dimension to obtain
        one probability-amplitude value per batch entry (corresponding to a single basis state).

        log_probs: natural log of the probability of sampling each value (spin) that was sampled.

        phases: phase contributions to the probability amplitude of the final basis state, selected from the
        "phase distribution" using the value actually sampled.
        """

        return torch.exp(0.5 * log_probs + 1j * phases)
