import argparse
import contextlib
import csv
import pathlib
import shlex
import subprocess
import tempfile
import time

import nanobind_gpu_example
import numpy as np
import scipy.io
import scipy.sparse
import torch

# Banded matrices have BFS traversals with a huge number of small levels
MAX_BFS_LEVELS = 2000
# A system is killed if it prints nothing for this long while it loads A,
BASELINE_LOAD_S = 120
# and for this long after that
BASELINE_VECTOR_S = 10
# and is run again at most this many times per matrix
BASELINE_RESTARTS = 5

# Our variants, timed by benchmarks/harness/ours
VARIANTS = ["dense", "esc", "esc_min_bits", "hash", "cusparse"]

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


def matrix_kind(path: pathlib.Path) -> str:
    """The "kind" in the header of a SuiteSparse Matrix Market file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("%"):
                break
            if line[1:].strip().startswith("kind:"):
                return line.split("kind:", 1)[1].strip()
    return ""


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
    for source in torch.randint(0, n, (16,), generator=g).tolist():
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


def input_vectors(A, m: int, n: int, all_levels: bool) -> dict:
    """The vectors x of a matrix by label: BFS frontiers and uniform vectors."""
    vectors = {}
    if m == n:
        frontiers = largest_bfs(A, m, n, seed=0)
        reached = sum(map(len, frontiers))
        levels = range(len(frontiers)) if all_levels else select_levels(frontiers)
        for level in levels:
            vectors[f"bfs level {level}"] = random_values(frontiers[level], seed=level)
        print(f"{'':>14} bfs: {len(frontiers)} levels, reached {reached} of {n}")
    for density in (1e-4, 1e-3, 1e-2, 1e-1):
        vectors[f"uniform {density:g}"] = uniform_vector(
            n, max(1, int(n * density)), seed=1
        )
    return vectors


def reference_cases(A, m: int, vectors) -> list:
    """The vectors and their results on the host: (x_crd, x_val, z_crd, z_val).

    Every system is checked against these results, which are from our ESC.
    """
    cases = []
    for crd, val in vectors:
        z = nanobind_gpu_example.spmspv_esc(*A, crd, val, m, min_bits=True)
        arrays = (crd, val, *z)
        cases.append(tuple(array.cpu().numpy() for array in arrays))
    return cases


def write_matrix(directory: str, A, m: int, n: int):
    """<directory>/A.bin: m, n, nnz, then the pos, crd and val arrays of A in CSC."""
    with open(f"{directory}/A.bin", "wb") as f:
        np.array([m, n, len(A[1])], dtype=np.int32).tofile(f)
        for array in A:
            array.cpu().numpy().tofile(f)


def write_vectors(directory: str, cases: list):
    """<directory>/x.bin: number of vectors, then nnz, crd and val of each vector."""
    with open(f"{directory}/x.bin", "wb") as f:
        np.array([len(cases)], dtype=np.int32).tofile(f)
        for crd, val, _, _ in cases:
            np.array([len(crd)], dtype=np.int32).tofile(f)
            crd.tofile(f)
            val.tofile(f)


def same_result(crd, val, z_crd, z_val) -> bool:
    """Whether crd and val are z. Rows of z that sum to zero may be left out."""
    tolerance = 1e-3 * np.abs(z_val).max(initial=0)
    kept = np.isin(z_crd, crd)
    return bool(
        np.array_equal(z_crd[kept], crd)
        and np.abs(val - z_val[kept]).max(initial=0) <= tolerance
        and np.abs(z_val[~kept]).max(initial=0) <= tolerance
    )


def run_system_once(command: str, directory: str, cases: list, load_s: float):
    """Runs a system on the vectors in <directory>/x.bin.

    Returns the median time in ms of each vector that the system finished,
    or "" if its result differs from the reference. The list is shorter than
    cases if the system crashed, or printed nothing for BASELINE_VECTOR_S
    (load_s before its first line). Also returns the seconds until its first
    line.
    """
    exe, *options = shlex.split(command)
    pathlib.Path(f"{directory}/z.bin").unlink(missing_ok=True)
    with open(f"{directory}/stdout.txt", "w") as stdout:
        process = subprocess.Popen(
            [exe, directory, *options], stdout=stdout, stderr=subprocess.DEVNULL
        )
        # The system prints a line after every vector
        printed = 0
        start = progress = time.monotonic()
        first_line_s = load_s
        while process.poll() is None:
            time.sleep(0.05)
            size = pathlib.Path(stdout.name).stat().st_size
            if size != printed:
                if printed == 0:
                    first_line_s = time.monotonic() - start
                printed = size
                progress = time.monotonic()
            # Loading A takes much longer than a vector
            limit = load_s if printed == 0 else BASELINE_VECTOR_S
            if time.monotonic() - progress > limit:
                process.kill()
                process.wait()

    # Some systems print other lines too
    lines = pathlib.Path(f"{directory}/stdout.txt").read_text().splitlines()
    lines = [line for line in lines if "RESULT" in line]
    times = []
    if not pathlib.Path(f"{directory}/z.bin").exists():
        return times, first_line_s
    with open(f"{directory}/z.bin", "rb") as f:
        for (_, _, z_crd, z_val), line in zip(cases, lines):
            _, status, ms = line.split()[-3:]
            head = np.fromfile(f, dtype=np.int32, count=2)
            nnz = int(head[1]) if len(head) == 2 else 0
            crd = np.fromfile(f, dtype=np.int32, count=nnz)
            val = np.fromfile(f, dtype=np.float32, count=nnz)
            if len(head) < 2 or len(val) < nnz:
                # The system stopped while writing this result
                break
            same = int(status) == 0 and same_result(crd, val, z_crd, z_val)
            times.append(float(ms) if same else "")
    return times, first_line_s


def run_system(command: str, directory: str, cases: list) -> list:
    """Median time in ms of a system on each vector, or "" if it fails.

    The system is run as `<exe> <directory> [options]`, see
    benchmarks/harness/harness.h. A result that differs from the reference
    counts as a failure. After a crash or a hang, the system is run again on
    the vectors after the one it failed on.
    """
    times = []
    restarts = 0
    load_s = BASELINE_LOAD_S
    while len(times) < len(cases) and restarts <= BASELINE_RESTARTS:
        remaining = cases[len(times) :]
        write_vectors(directory, remaining)
        finished, first_line_s = run_system_once(command, directory, remaining, load_s)
        times += finished
        # The next run loads A again, which takes as long as it did now
        load_s = min(load_s, first_line_s + BASELINE_VECTOR_S)
        if len(times) < len(cases):
            times.append("")
            restarts += 1
    return times + [""] * (len(cases) - len(times))


def bfs_total(rows: list[dict]) -> dict:
    """One row with the sums over every level of the BFS."""
    levels = [row for row in rows if row["vector"].startswith("bfs level")]
    total = {"vector": "bfs total"}
    for field in levels[0]:
        values = [row[field] for row in levels]
        # No total if an implementation failed on a level
        if field != "vector":
            total[field] = "" if "" in values else sum(values)
    return total


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
        "--ours",
        metavar="EXE",
        help="time our variants with this executable (see benchmarks/harness/ours)",
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma separated variants of ours to time",
    )
    parser.add_argument(
        "--all-bfs-levels",
        action="store_true",
        help="time every level of the BFS, and add a row with the totals",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="also time a baseline executable, e.g. graphblast=<dir>/grb_spmspv "
        'or lay_sort_merge="<dir>/lay_spmspv --kernel sort-merge" '
        "(see benchmarks/harness)",
    )
    args = parser.parse_args()
    # Every system is an executable, timed by benchmarks/harness/harness.h
    systems = {}
    if args.ours is not None:
        for name in args.variants.split(","):
            systems[name] = f"{shlex.quote(args.ours)} --variant {name}"
    systems.update(baseline.split("=", 1) for baseline in args.baseline)

    with contextlib.ExitStack() as stack:
        writer = None
        if args.csv is not None:
            fields = ["matrix", "kind", "m", "n", "nnz"]
            fields += ["vector", "x_nnz", "products", "z_nnz"]
            fields += [f"{name}_ms" for name in systems]
            # Line buffered, so a crash keeps the finished rows
            csv_file = stack.enter_context(open(args.csv, "w", newline="", buffering=1))
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

        for name, kind, load in matrices(args):
            A, m, n = load()
            nnz = len(A[1])
            lengths = (A[0][1:] - A[0][:-1]).float()
            print(
                f"\n{name}: {m} x {n}, nnz {nnz}, "
                f"column length mean {lengths.mean():.1f} max {int(lengths.max())}"
            )
            vectors = input_vectors(A, m, n, args.all_bfs_levels)
            cases = reference_cases(A, m, vectors.values())
            rows = []
            for (label, (x_crd, _)), (_, _, z_crd, _) in zip(vectors.items(), cases):
                columns = x_crd.long()
                products = int((A[0][columns + 1] - A[0][columns]).sum())
                rows.append(
                    {
                        "vector": label,
                        "x_nnz": len(x_crd),
                        "products": products,
                        "z_nnz": len(z_crd),
                    }
                )

            with tempfile.TemporaryDirectory() as directory:
                write_matrix(directory, A, m, n)
                # The systems need the GPU memory
                del A, vectors
                torch.cuda.empty_cache()
                nanobind_gpu_example.free_unused_memory()
                for system, command in systems.items():
                    times = run_system(command, directory, cases)
                    for row, ms in zip(rows, times):
                        row[f"{system}_ms"] = ms
                    failed = times.count("")
                    if failed > 0:
                        print(f"{'':>14} {system}: failed on {failed} of {len(times)}")

            if args.all_bfs_levels and m == n:
                rows.append(bfs_total(rows))

            header = ["vector", "nnz(x)", "products", "nnz(z)", *systems]
            print(" ".join(f"{column:>16}" for column in header))
            for row in rows:
                cells = [f"{value:>16}" for value in list(row.values())[:4]]
                for system in systems:
                    ms = row[f"{system}_ms"]
                    cells.append(f"{ms:>16.3f}" if ms != "" else f"{'failed':>16}")
                print(" ".join(cells), flush=True)

            if writer is not None:
                for row in rows:
                    matrix = {"matrix": name, "kind": kind, "m": m, "n": n, "nnz": nnz}
                    writer.writerow({**matrix, **row})


def matrices(args):
    """Yields (name, kind, load) per matrix, where load() returns (A, m, n)."""
    if args.suitesparse is None:
        m = n = 2**22
        for skewed in (False, True):
            kind = "skewed" if skewed else "uniform"
            yield (
                f"random ({kind} column lengths)",
                "random",
                lambda skewed=skewed: (random_csc(m, n, 16, skewed, seed=0), m, n),
            )
        return

    names = args.matrices.split(",")
    if args.min_nnz is not None:
        names = suitesparse_matrices(args.suitesparse, args.min_nnz)
    for name in names:
        path = args.suitesparse / name / f"{name}.mtx"

        def load(path=path):
            A, (m, n) = load_suitesparse(path)
            return A, m, n

        yield name, matrix_kind(path), load


if __name__ == "__main__":
    main()
