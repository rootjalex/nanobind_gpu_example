"""Download SNAP graphs and convert them to Matrix Market files.

    python benchmarks/datasets/snap.py <directory>

Writes <directory>/<name>/<name>.mtx, the layout that benchmarks/spmspv.py
expects for --suitesparse. An edge u -> v becomes the entry A[v, u], so that
A x follows the edges out of the vertices in x.
"""

import argparse
import gzip
import pathlib
import urllib.request

import numpy as np
import scipy.io
import scipy.sparse

URL = "https://snap.stanford.edu/data/"

# name: (file, undirected)
GRAPHS = {
    "soc-LiveJournal1": ("soc-LiveJournal1.txt.gz", False),
    "soc-Pokec": ("soc-pokec-relationships.txt.gz", False),
    "com-Orkut": ("bigdata/communities/com-orkut.ungraph.txt.gz", True),
    "com-LiveJournal": ("bigdata/communities/com-lj.ungraph.txt.gz", True),
    "com-Youtube": ("bigdata/communities/com-youtube.ungraph.txt.gz", True),
    "com-Amazon": ("bigdata/communities/com-amazon.ungraph.txt.gz", True),
    "com-DBLP": ("bigdata/communities/com-dblp.ungraph.txt.gz", True),
    "wiki-Talk": ("wiki-Talk.txt.gz", False),
    "wiki-topcats": ("wiki-topcats.txt.gz", False),
    "higgs-twitter": ("higgs-social_network.edgelist.gz", False),
    "web-Google": ("web-Google.txt.gz", False),
    "web-BerkStan": ("web-BerkStan.txt.gz", False),
    "web-Stanford": ("web-Stanford.txt.gz", False),
    "web-NotreDame": ("web-NotreDame.txt.gz", False),
    "cit-Patents": ("cit-Patents.txt.gz", False),
    "roadNet-CA": ("roadNet-CA.txt.gz", True),
    "roadNet-TX": ("roadNet-TX.txt.gz", True),
    "roadNet-PA": ("roadNet-PA.txt.gz", True),
    "as-Skitter": ("as-skitter.txt.gz", True),
    "amazon0601": ("amazon0601.txt.gz", False),
}


def download(file: str, directory: pathlib.Path) -> pathlib.Path:
    """Download one SNAP file into directory, unless it is already there."""
    path = directory / pathlib.Path(file).name
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        request = urllib.request.Request(URL + file, headers={"User-Agent": "python"})
        with urllib.request.urlopen(request) as response, open(partial, "wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        partial.rename(path)
    return path


def read_graph(path: pathlib.Path, undirected: bool) -> scipy.sparse.csc_array:
    """Read an edge list as the pattern of A, with A[v, u] = 1 for u -> v."""
    with gzip.open(path, "rt") as f:
        edges = np.loadtxt(f, dtype=np.int64, comments="#", usecols=(0, 1))
    # SNAP vertex ids have gaps
    ids, edges = np.unique(edges, return_inverse=True)
    src, dst = edges.reshape(-1, 2).T
    ones = np.ones(len(src), dtype=np.int8)
    A = scipy.sparse.coo_array((ones, (dst, src)), shape=(len(ids), len(ids)))
    A = A.tocsc()
    if undirected:
        A = A + A.T
    # Duplicate edges were summed
    A.data[:] = 1
    return A


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    raw = args.directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    for name, (file, undirected) in GRAPHS.items():
        target = args.directory / name / f"{name}.mtx"
        if target.exists():
            continue
        A = read_graph(download(file, raw), undirected)
        kind = "undirected graph" if undirected else "directed graph"
        target.parent.mkdir(exist_ok=True)
        scipy.io.mmwrite(
            target,
            A,
            comment=f" name: SNAP/{name}\n kind: {kind}",
            field="pattern",
            symmetry="symmetric" if undirected else "general",
        )
        print(f"{name}: {A.shape[0]} vertices, {A.nnz} nonzeros", flush=True)


if __name__ == "__main__":
    main()
