"""
Benchmark iid sampler throughput across a grid of (num_walkers, microbatch_size)
configurations. Run from the repo root:

    uv run python scripts/benchmark_sampler.py

Results are printed as a table sorted by samples/second descending.
"""

import itertools
import sys
import time

import torch
from tabulate import tabulate

sys.path.insert(0, ".")

from hamiltonian.transverse_field_ising import TransverseFieldIsing
from model.tqs import TransformerQuantumState

# ---------------------------------------------------------------------------
# Model config — mirror main.py defaults
# ---------------------------------------------------------------------------
L = 8
h = 0.25
J = 1.0
periodic = False
d_model = 32
n_layers = 8
n_heads = 8
dim_feedforward = 4 * d_model

WARMUP_REPS = 5
BENCH_REPS = 20

WALKERS_GRID = [256, 512, 1024, 2048]
MICROBATCH_GRID = [64, 128, 256, 512]


def _sync_and_time(fn, reps: int) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def _peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def build_model(device: torch.device) -> TransformerQuantumState:
    hamiltonian = TransverseFieldIsing(
        system_dim=torch.tensor([float(L)]),
        phys_params=torch.tensor([h]),
        coupling=J,
        periodic=periodic,
        device=device,
    )
    return TransformerQuantumState(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        max_len=L,
        hamiltonian=hamiltonian,
        device=device,
    )


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(device)}\n")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS\n")
    else:
        device = torch.device("cpu")
        print("Device: CPU\n")

    model = build_model(device)
    model.eval()

    rows = []
    for nw, mb in itertools.product(WALKERS_GRID, MICROBATCH_GRID):
        if mb > nw:
            continue

        for _ in range(WARMUP_REPS):
            model.sample_iid_microbatches(nw, mb, nw)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        elapsed = _sync_and_time(lambda: model.sample_iid_microbatches(nw, mb, nw), BENCH_REPS)
        rows.append((nw, mb, elapsed * 1e3, nw / elapsed, _peak_mb()))

    rows.sort(key=lambda r: -r[3])

    print(
        tabulate(
            [[nw, mb, f"{ms:.1f}", f"{sps:.0f}", f"{mem:.0f}" if mem == mem else "n/a"]
             for nw, mb, ms, sps, mem in rows],
            headers=["walkers", "microbatch", "ms/call", "samples/s", "peak_MB"],
            tablefmt="rounded_outline",
        )
    )


if __name__ == "__main__":
    main()
