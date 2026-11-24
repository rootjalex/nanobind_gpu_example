import argparse
import contextlib
import csv
import functools
import pathlib
import statistics
import subprocess
import tempfile
import time

import nanobind_gpu_example
import numpy as np
import scipy.io
import scipy.sparse
import torch

ITERS = 10
# Banded matrices have BFS traversals with a huge number of small levels
MAX_BFS_LEVELS = 2000

IMPLS = {
    "dense": nanobind_gpu_example.spmspv_dense,
    "esc": nanobind_gpu_example.spmspv_esc,
    "esc_min_bits": lambda *args: nanobind_gpu_example.spmspv_esc(*args, min_bits=True),
    "hash": nanobind_gpu_example.spmspv_hash,
    "cusparse": nanobind_gpu_example.spmspv_cusparse,
}

# Used when --matrices is not given
SUITESPARSE_MATRICES = [
    "audikw_1",
    "cage15",
    "circuit5M",
    "FullChip",
    "HV15R",
    "nlpkkt160",
]


def random_csc(m: int, n: int, degree: int, skewed: bool, seed: int):
    """Random m x n CSC matrix on the GPU with n * degree nonzeros on average."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if skewed:
        # Power law column lengths
        lengths = torch.rand(n, generator=g, device="cuda").pow(-1 / 1.5)
        lengths = (lengths * degree / lengths.mean()).long().clamp(max=m)
    else:
        lengths = torch.full((n,), degree, device="cuda")
    pos = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
    pos[1:] = lengths.cumsum(0)
    nnz = int(pos[-1])

    # Sort the rows of each column
    cols = torch.repeat_interleave(torch.arange(n, device="cuda"), lengths)
    rows = torch.randint(0, m, (nnz,), generator=g, device="cuda")
    crd = (cols * m + rows).sort().values % m
    val = torch.rand(nnz, generator=g, device="cuda") + 0.5
    return pos.int(), crd.int(), val


def load_suitesparse(path: pathlib.Path):
    """Load a Matrix Market file as a CSC matrix on the GPU."""
    A = scipy.sparse.csc_array(scipy.io.mmread(path, spmatrix=False))
    A.sort_indices()
    pos = torch.from_numpy(A.indptr.astype(np.int32)).cuda()
    crd = torch.from_numpy(A.indices.astype(np.int32)).cuda()
    val = torch.from_numpy(A.data.astype(np.float32)).cuda()
    return (pos, crd, val), A.shape


def random_values(crd: torch.Tensor, seed: int):
    """Sparse vector with the nonzeros crd and random positive values."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    val = torch.rand(len(crd), generator=g, device="cuda") + 0.5
    return crd, val


def uniform_vector(n: int, nnz: int, seed: int):
    """Sparse vector with nnz nonzeros chosen uniformly at random."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    crd = torch.randperm(n, generator=g, device="cuda")[:nnz].sort().values
    return random_values(crd.int(), seed)


def bfs_frontiers(A, m: int, source: int) -> list[torch.Tensor]:
    """Frontier of every level of a BFS from source, following A x."""
    pos, crd, _ = A
    # All ones, so that no sum cancels to zero
    ones = torch.ones(len(crd), device="cuda")
    visited = torch.zeros(m, dtype=torch.bool, device="cuda")
    visited[source] = True
    frontier = torch.tensor([source], dtype=torch.int32, device="cuda")

    frontiers = []
    while len(frontier) > 0 and len(frontiers) < MAX_BFS_LEVELS:
        frontiers.append(frontier)
        x_val = torch.ones(len(frontier), device="cuda")
        reached, _ = nanobind_gpu_example.spmspv_dense(
            pos, crd, ones, frontier, x_val, m
        )
        frontier = reached[~visited[reached.long()]]
        visited[frontier.long()] = True
    return frontiers


def largest_bfs(A, m: int, n: int, seed: int) -> list[torch.Tensor]:
    """BFS frontiers from the random source that reaches the most vertices."""
    g = torch.Generator().manual_seed(seed)
    best = []
    for source in torch.randint(0, n, (4,), generator=g).tolist():
        frontiers = bfs_frontiers(A, m, source)
        if sum(map(len, frontiers)) > sum(map(len, best)):
            best = frontiers
        if sum(map(len, best)) > n // 2:
            break
    return best


def select_levels(frontiers: list[torch.Tensor]) -> list[int]:
    """The first level with at least 10^k nonzeros for each k, and the largest."""
    sizes = [len(frontier) for frontier in frontiers]
    levels = {sizes.index(max(sizes))}
    threshold = 100
    while threshold <= max(sizes):
        levels.add(next(i for i, size in enumerate(sizes) if size >= threshold))
        threshold *= 10
    return sorted(levels)


def bench_ms(fn) -> float:
    """Median time of fn() in ms, after a warmup."""
    t0 = time.perf_counter()
    fn()
    nanobind_gpu_example.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000.0
    # Slow calls get fewer iterations
    iters = ITERS if first_ms < 50 else 3

    for _ in range(3 if first_ms < 50 else 0):
        fn()
        nanobind_gpu_example.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        nanobind_gpu_example.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return statistics.median(times)


def bench_row(label: str, A, x, m: int, impls: dict) -> dict:
    """Time the implementations on z = A x, print and return one row."""
    pos = A[0]
    x_cols = x[0].long()
    products = int((pos[x_cols + 1] - pos[x_cols]).sum())

    z_nnz = {}
    ms = {}
    for name, impl in impls.items():
        try:
            z_crd, _ = impl(*A, *x, m)
            z_nnz[name] = len(z_crd)
            del z_crd
            ms[name] = bench_ms(functools.partial(impl, *A, *x, m))
        except RuntimeError:
            # Usually out of memory
            nanobind_gpu_example.free_unused_memory()

    # dense drops rows that sum to exactly zero, the others must agree
    others = {name: nnz for name, nnz in z_nnz.items() if name != "dense"}
    nnz = next(iter(others.values()), "")
    row = [label, len(x[0]), products, nnz]
    times = [
        f"{ms[name]:>14.3f}" if name in ms else f"{'failed':>14}" for name in impls
    ]
    print(" ".join(f"{value:>14}" for value in row), " ".join(times), flush=True)
    if len(set(others.values())) > 1:
        print(f"{'':>14} nnz(z) differs: {z_nnz}")

    return {
        "vector": label,
        "x_nnz": len(x[0]),
        "products": products,
        "z_nnz": nnz,
        **{f"{name}_ms": ms.get(name, "") for name in impls},
    }


def bench_matrix(A, m: int, n: int, impls: dict):
    """Returns the rows of the matrix, and the vector x of each row."""
    header = ["vector", "nnz(x)", "products", "nnz(z)", *impls]
    print(" ".join(f"{name:>14}" for name in header))

    vectors = {}
    if m == n:
        frontiers = largest_bfs(A, m, n, seed=0)
        reached = sum(map(len, frontiers))
        for level in select_levels(frontiers):
            vectors[f"bfs level {level}"] = random_values(frontiers[level], seed=level)
        print(f"{'':>14} bfs: {len(frontiers)} levels, reached {reached} of {n}")
        del frontiers
    for density in (1e-4, 1e-3, 1e-2, 1e-1):
        vectors[f"uniform {density:g}"] = uniform_vector(
            n, max(1, int(n * density)), seed=1
        )

    rows = [bench_row(label, A, x, m, impls) for label, x in vectors.items()]
    return rows, list(vectors.values())


def write_graphblast_inputs(directory: str, A, m: int, n: int, vectors) -> list:
    """Write A and the vectors for grb_spmspv. Returns our result per vector."""
    expected = []
    with open(f"{directory}/A.bin", "wb") as f:
        np.array([m, n, len(A[1])], dtype=np.int32).tofile(f)
        for array in A:
            array.cpu().numpy().tofile(f)
    with open(f"{directory}/x.bin", "wb") as f:
        np.array([len(vectors)], dtype=np.int32).tofile(f)
        for crd, val in vectors:
            np.array([len(crd)], dtype=np.int32).tofile(f)
            crd.cpu().numpy().tofile(f)
            val.cpu().numpy().tofile(f)
            z = nanobind_gpu_example.spmspv_esc(*A, crd, val, m, min_bits=True)
            expected.append((z[0].cpu().numpy(), z[1].cpu().numpy()))
    return expected


def run_graphblast(exe: pathlib.Path, directory: str, m: int, n: int, expected):
    """Median time in ms of GraphBLAST on each vector, or "" if it fails.

    A result that differs from our ESC result counts as a failure.
    """
    # GraphBLAST sorts on the bits needed for n rows, which is wrong for m > n
    options = ["--endbit", "1" if m <= n else "0"]
    try:
        # A crash is handled below, by the missing results
        result = subprocess.run(
            [exe, directory, *options],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [""] * len(expected)

    times = []
    with open(f"{directory}/z.bin", "rb") as f:
        # GraphBLAST prints other lines too
        lines = [line for line in result.stdout.splitlines() if "RESULT" in line]
        for (z_crd, z_val), line in zip(expected, lines):
            _, status, ms = line.split()[-3:]
            nnz = int(np.fromfile(f, dtype=np.int32, count=2)[1])
            crd = np.fromfile(f, dtype=np.int32, count=nnz)
            val = np.fromfile(f, dtype=np.float32, count=nnz)
            same = (
                int(status) == 0
                and np.array_equal(crd, z_crd)
                and np.abs(val - z_val).max(initial=0)
                <= 1e-3 * np.abs(z_val).max(initial=0)
            )
            times.append(float(ms) if same else "")
    # GraphBLAST crashed before the remaining vectors
    return times + [""] * (len(expected) - len(times))


def suitesparse_matrices(directory: pathlib.Path, min_nnz: int) -> list[str]:
    """Names of the matrices with at least min_nnz nonzeros, smallest first."""
    sizes = []
    for path in directory.glob("*/*.mtx"):
        with open(path, "rb") as f:
            symmetry = f.readline().split()[4]
            line = f.readline()
            while line.startswith(b"%"):
                line = f.readline()
        nnz = int(line.split()[2]) * (1 if symmetry == b"general" else 2)
        if nnz >= min_nnz:
            sizes.append((nnz, path.parent.name))
    return [name for _, name in sorted(sizes)]


def main():
    parser = argparse.ArgumentParser(description="SpMSpV benchmark, times in ms")
    parser.add_argument(
        "--suitesparse",
        type=pathlib.Path,
        help="directory with <name>/<name>.mtx files (default: random matrices)",
    )
    parser.add_argument(
        "--matrices",
        default=",".join(SUITESPARSE_MATRICES),
        help="comma separated SuiteSparse matrix names",
    )
    parser.add_argument(
        "--min-nnz",
        type=int,
        help="use every SuiteSparse matrix with at least this many nonzeros",
    )
    parser.add_argument("--csv", type=pathlib.Path, help="also write the rows here")
    parser.add_argument(
        "--impls",
        default=",".join(IMPLS),
        help="comma separated implementations to time",
    )
    parser.add_argument(
        "--graphblast",
        type=pathlib.Path,
        help="also time GraphBLAST with this grb_spmspv executable "
        "(see benchmarks/baselines/graphblast/build.sh)",
    )
    args = parser.parse_args()
    impls = {name: IMPLS[name] for name in args.impls.split(",")}

    with contextlib.ExitStack() as stack:
        writer = None
        if args.csv is not None:
            fields = ["matrix", "m", "n", "nnz", "vector", "x_nnz", "products", "z_nnz"]
            fields += [f"{name}_ms" for name in impls]
            if args.graphblast is not None:
                fields.append("graphblast_ms")
            # Line buffered, so a crash keeps the finished rows
            csv_file = stack.enter_context(open(args.csv, "w", newline="", buffering=1))
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

        for name, load in matrices(args):
            A, m, n = load()
            nnz = len(A[1])
            lengths = (A[0][1:] - A[0][:-1]).float()
            print(
                f"\n{name}: {m} x {n}, nnz {nnz}, "
                f"column length mean {lengths.mean():.1f} max {int(lengths.max())}"
            )
            rows, vectors = bench_matrix(A, m, n, impls)

            if args.graphblast is not None:
                with tempfile.TemporaryDirectory() as directory:
                    expected = write_graphblast_inputs(directory, A, m, n, vectors)
                    # GraphBLAST needs the GPU memory
                    del A, vectors
                    torch.cuda.empty_cache()
                    nanobind_gpu_example.free_unused_memory()
                    times = run_graphblast(args.graphblast, directory, m, n, expected)
                for row, ms in zip(rows, times):
                    row["graphblast_ms"] = ms
                shown = [f"{ms:.3f}" if ms != "" else "failed" for ms in times]
                print(f"{'':>14} graphblast: {' '.join(shown)}", flush=True)
            else:
                del A
                torch.cuda.empty_cache()
                nanobind_gpu_example.free_unused_memory()

            if writer is not None:
                for row in rows:
                    writer.writerow({"matrix": name, "m": m, "n": n, "nnz": nnz, **row})


def matrices(args):
    """Yields (name, load) per matrix, where load() returns (A, m, n)."""
    if args.suitesparse is None:
        m = n = 2**22
        for skewed in (False, True):
            kind = "skewed" if skewed else "uniform"
            yield (
                f"random ({kind} column lengths)",
                lambda skewed=skewed: (random_csc(m, n, 16, skewed, seed=0), m, n),
            )
        return

    names = args.matrices.split(",")
    if args.min_nnz is not None:
        names = suitesparse_matrices(args.suitesparse, args.min_nnz)
    for name in names:

        def load(name=name):
            A, (m, n) = load_suitesparse(args.suitesparse / name / f"{name}.mtx")
            return A, m, n

        yield name, load


if __name__ == "__main__":
    main()
