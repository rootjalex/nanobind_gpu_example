import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

# One fixed color per series
COLORS = {
    "dense": "#88CCEE",
    "esc": "#CC6677",
    "hash": "#999933",
    "esc_full": "#882255",
    "best": "#332288",
    "cusparse": "#117733",
    "graphblast": "#AA4499",
}
LABELS = {
    "dense": "Dense",
    "esc": "ESC",
    "hash": "Hash",
    "best": "Best of dense, ESC, hash",
    "cusparse": "cuSPARSE",
    "graphblast": "GraphBLAST",
}


def load(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read the benchmark CSV as one array per column."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = {name: table[name] for name in table.dtype.names}
    # ESC sorts on the bits needed for m rows unless stated otherwise
    data["esc_full"] = data.pop("esc_ms")
    data["esc"] = data.pop("esc_min_bits_ms")
    for name in ("dense", "hash", "cusparse"):
        data[name] = data.pop(f"{name}_ms")
    # Only present if the benchmark ran with --graphblast
    if "graphblast_ms" in data:
        data["graphblast"] = data.pop("graphblast_ms").astype(float)
    ours = np.stack([data["dense"], data["esc"], data["hash"]])
    data["best"] = np.nanmin(ours, axis=0)
    data["fastest"] = np.array(["dense", "esc", "hash"])[np.nanargmin(ours, axis=0)]
    return data


def geomean(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(values))))


def vector_groups(data) -> dict[str, np.ndarray]:
    """Rows of each plot: the BFS frontiers, and each density of uniform vectors."""
    groups = {"bfs": np.char.startswith(data["vector"], "bfs")}
    for label in np.unique(data["vector"]):
        if label.startswith("uniform"):
            groups["uniform" + label.split()[1]] = data["vector"] == label
    return groups


def scatter(ax, data, rows, series: list[str], labels: dict[str, str]):
    """Runtime of each series against the number of products, on log axes."""
    x = np.concatenate([data["products"][rows] for _ in series])
    y = np.concatenate([data[name][rows] for name in series])
    colors = np.concatenate([[COLORS[name]] * rows.sum() for name in series])
    # Shuffle so that no series is drawn on top of the others
    order = np.random.default_rng(0).permutation(len(x))
    ax.scatter(x[order], y[order], c=colors[order], s=5, linewidths=0, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Only label the powers of ten
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Products (nonzeros in the selected columns of A)")
    ax.set_ylabel("Runtime (ms)")
    ax.grid(True, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS[name], label=labels[name])
        for name in series
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=1)


def stats(values: np.ndarray) -> str:
    return (
        f"geo-mean {geomean(values):.3f}, min {values.min():.3f}, "
        f"median {np.median(values):.3f}, max {values.max():.3f}"
    )


def summary(data, rows, excluded: int, series: list[str], labels) -> str:
    """Summary statistics of one plot."""
    products = data["products"][rows]
    lines = [
        f"inputs: {rows.sum()}",
        f"excluded (a series failed): {excluded}",
        (
            f"products: min {products.min()}, median {int(np.median(products))}, "
            f"max {products.max()}"
        ),
        "",
        "runtime in ms",
    ]
    lines += [f"  {labels[name]}: {stats(data[name][rows])}" for name in series]

    times = np.stack([data[name][rows] for name in series])
    fastest = np.argmin(times, axis=0)
    lines += ["", "fastest"]
    for i, name in enumerate(series):
        count = int((fastest == i).sum())
        lines.append(f"  {labels[name]}: {count} ({100 * count / rows.sum():.1f}%)")

    # Speedup of the first series over each of the others
    for other in series[1:]:
        speedup = data[other][rows] / data[series[0]][rows]
        lines += [
            "",
            f"speedup of {labels[series[0]]} over {labels[other]}",
            f"  {stats(speedup)}",
        ]
    return "\n".join(lines) + "\n"


def figures(data, series: list[str], stem: pathlib.Path, labels=LABELS):
    """One plot and one summary file per kind of vector, named <stem>_<kind>."""
    for kind, rows in vector_groups(data).items():
        # Skip inputs where one of the series failed
        ok = rows.copy()
        for name in series:
            ok &= ~np.isnan(data[name])
        if not ok.any():
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        scatter(ax, data, ok, series, labels)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(f"{stem}_{kind}{suffix}", dpi=200)
        plt.close(fig)

        text = summary(data, ok, int(rows.sum() - ok.sum()), series, labels)
        pathlib.Path(f"{stem}_{kind}.txt").write_text(text)
        print(f"wrote {stem}_{kind}.pdf, .png and .txt")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmarks/spmspv.py results")
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("."))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.csv)

    figures(data, ["dense", "esc", "hash"], args.out / "spmspv_variants")
    figures(data, ["best", "cusparse"], args.out / "spmspv_vs_cusparse")
    figures(data, ["dense", "cusparse"], args.out / "spmspv_dense_vs_cusparse")
    figures(data, ["esc", "cusparse"], args.out / "spmspv_esc_vs_cusparse")
    figures(
        data,
        ["esc", "esc_full"],
        args.out / "spmspv_esc_bits",
        labels={"esc": "ESC (min bits)", "esc_full": "ESC (32 bits)"},
    )
    if "graphblast" in data:
        figures(data, ["esc", "graphblast"], args.out / "spmspv_vs_graphblast")


if __name__ == "__main__":
    main()
