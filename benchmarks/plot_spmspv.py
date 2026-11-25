import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# One fixed color per series
COLORS = {
    "dense": "#88CCEE",
    "esc": "#CC6677",
    "hash": "#999933",
    "esc_full": "#882255",
    "best": "#332288",
    "cusparse": "#117733",
    "graphblast": "#AA4499",
    "lay_atomic_merge": "#44AA99",
    "lay_atomic_naive": "#DDCC77",
    "lay_sort_merge": "#661100",
}
LABELS = {
    "dense": "Dense (Ours)",
    "esc": "ESC (Ours)",
    "hash": "Hash (Ours)",
    "best": "Best of ours",
    "cusparse": "cuSPARSE",
    "graphblast": "GraphBLAST",
    "lay_atomic_merge": "Dense-LB (Li et al.)",
    "lay_atomic_naive": "Dense-noLB (Li et al.)",
    "lay_sort_merge": "ESC-LB (Li et al.)",
}
# Marker of a failed run in the bar plots
FAILED = {"linestyle": "", "marker": "x", "markersize": 4, "markeredgewidth": 1.2}

# The kernels of Li et al. that are plotted
LAY = ["lay_atomic_merge", "lay_atomic_naive", "lay_sort_merge"]


# (word in the SuiteSparse kind, class)
CLASSES = [
    ("circuit", "circuit simulation"),
    ("structural", "structural"),
    ("fluid", "fluid dynamics"),
    ("2d/3d", "2D/3D"),
    ("optimization", "optimization"),
    ("linear programming", "optimization"),
    ("least squares", "optimization"),
    ("graph", "graph"),
    ("power network", "power network"),
    ("semiconductor", "semiconductor"),
    ("chemi", "chemistry"),
    ("electromagnetic", "electromagnetics"),
    ("model reduction", "model reduction"),
]


def matrix_class(kind: str) -> str:
    """A coarse class for a SuiteSparse kind such as "structural problem"."""
    for word, name in CLASSES:
        if word in kind.lower():
            return name
    return "other"


def load(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read the benchmark CSV as one array per column."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = {name: table[name] for name in table.dtype.names}
    # ESC sorts on the bits needed for m rows unless stated otherwise
    data["esc_full"] = data.pop("esc_ms")
    data["esc"] = data.pop("esc_min_bits_ms")
    for name in ("dense", "hash", "cusparse"):
        data[name] = data.pop(f"{name}_ms")
    # Only present if the benchmark ran with these baselines
    for name in ["graphblast", *LAY]:
        if f"{name}_ms" in data:
            data[name] = data.pop(f"{name}_ms").astype(float)
    if "kind" in data:
        data["class"] = np.array([matrix_class(kind) for kind in data["kind"]])
    ours = np.stack([data["dense"], data["esc"], data["hash"]])
    data["best"] = np.nanmin(ours, axis=0)
    data["fastest"] = np.array(["dense", "esc", "hash"])[np.nanargmin(ours, axis=0)]
    return data


def geomean(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(values))))


def vector_groups(data) -> dict[str, np.ndarray]:
    """Rows of each plot: the BFS frontiers, and each density of uniform vectors."""
    groups = {"bfs": np.char.startswith(data["vector"], "bfs level")}
    for label in np.unique(data["vector"]):
        if label.startswith("uniform"):
            groups["uniform" + label.split()[1]] = data["vector"] == label
    return groups


def label_log_axis(axis, low: float, high: float):
    """Labels a log axis from low to high with plain numbers.

    The default only labels the powers of ten, which is one label or none on
    a short axis. A shorter axis is also labelled between the powers.
    """
    decades = np.log10(high / low)
    subs = [1, 2, 3, 5, 7] if decades < 1 else [1, 2, 5] if decades < 3 else [1]
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


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
    label_log_axis(ax.yaxis, *ax.get_ylim())
    ax.set_xlabel("Partial products")
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


def failures(times: dict, labels) -> list[str]:
    """One line per series that failed on some input, with the number of them."""
    counts = {name: int(np.isnan(values).sum()) for name, values in times.items()}
    return [f"failed {labels[name]}: {n}" for name, n in counts.items() if n > 0]


def summary(data, rows, failed: list[str], series: list[str], labels) -> str:
    """Summary statistics of one plot, overall and per class of matrix."""
    products = data["products"][rows]
    lines = [
        f"inputs: {rows.sum()}",
        *failed,
        (
            f"products: min {products.min()}, median {int(np.median(products))}, "
            f"max {products.max()}"
        ),
        "",
        "runtime in ms",
    ]
    lines += [f"  {labels[name]}: {stats(data[name][rows])}" for name in series]

    def comparison(rows) -> list[str]:
        """How often each series is fastest, and the speedups of the first."""
        fastest = np.argmin(np.stack([data[name][rows] for name in series]), axis=0)
        text = []
        for i, name in enumerate(series):
            count = int((fastest == i).sum())
            share = 100 * count / rows.sum()
            text.append(f"  fastest {labels[name]}: {count} ({share:.1f}%)")
        for other in series[1:]:
            speedup = data[other][rows] / data[series[0]][rows]
            pair = f"{labels[series[0]]} over {labels[other]}"
            text.append(f"  speedup of {pair}: {stats(speedup)}")
        return text

    lines += ["", "all inputs", *comparison(rows)]
    if "class" in data:
        classes, counts = np.unique(data["class"][rows], return_counts=True)
        for name in classes[np.argsort(-counts)]:
            of_class = rows & (data["class"] == name)
            lines += ["", f"{name}: {of_class.sum()} inputs", *comparison(of_class)]
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

        failed = failures({name: data[name][rows] for name in series}, labels)
        text = summary(data, ok, failed, series, labels)
        pathlib.Path(f"{stem}_{kind}.txt").write_text(text)
        print(f"wrote {stem}_{kind}.pdf, .png and .txt")


def bar_figure(values: dict, names, ylabel: str, log: bool, line, labels, path):
    """Grouped bars: one group per matrix, one bar per series.

    With a line, the bars of a log axis start at the line. A series that
    failed on a matrix has an X in place of its bar.
    """
    fig, ax = plt.subplots(figsize=(0.45 * len(names) + 2.5, 3.2))
    width = 0.8 / len(values)
    bottom = line if log and line is not None else 0
    for i, (name, heights) in enumerate(values.items()):
        x = np.arange(len(names)) + (i + 0.5) * width - 0.4
        ax.bar(
            x,
            heights - bottom,
            0.85 * width,
            bottom=bottom,
            color=COLORS[name],
            label=labels[name],
        )
        # Just above the bottom of the axes
        failed = np.isnan(heights)
        ax.plot(
            x[failed],
            np.full(failed.sum(), 0.04),
            **FAILED,
            color=COLORS[name],
            transform=ax.get_xaxis_transform(),
        )
    if log:
        ax.set_yscale("log")
        label_log_axis(ax.yaxis, *ax.get_ylim())
    if line is not None:
        ax.axhline(line, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(values),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(f"{path}{suffix}", dpi=200)
    plt.close(fig)


def bar_table(values: dict, names, title: str, labels) -> list[str]:
    """The numbers of a bar plot, one line per matrix."""
    lines = [f"{title:<22}" + "".join(f"{labels[n]:>24}" for n in values)]
    for i, matrix in enumerate(names):
        cells = "".join(f"{heights[i]:>24.3f}" for heights in values.values())
        lines.append(f"{matrix:<22}{cells}")
    return lines


def bars(data, series: list[str], reference: str, stem: pathlib.Path, labels=LABELS):
    """Bar plots and summary files per kind of vector.

    <stem>_<kind> has the runtime of each series on each matrix, and
    <stem>_speedup_<kind> has the speedup of each series over the reference.
    The BFS plots show the total time of the levels of one BFS, which needs a
    benchmark run with --all-bfs-levels.
    """
    # Smallest matrix first
    names, first = np.unique(data["matrix"], return_index=True)
    names = names[np.argsort(data["nnz"][first])]

    groups = vector_groups(data)
    groups["bfs"] = data["vector"] == "bfs total"
    for kind, rows in groups.items():
        if not rows.any():
            continue
        times = {}
        for name in series:
            times[name] = np.full(len(names), np.nan)
            for i, matrix in enumerate(names):
                values = data[name][rows & (data["matrix"] == matrix)]
                if len(values) > 0:
                    times[name][i] = values[0]
        failed = failures(times, labels)

        bar_figure(times, names, "Runtime (ms)", True, None, labels, f"{stem}_{kind}")
        lines = bar_table(times, names, "runtime in ms", labels)
        lines += ["", *failed, "over the matrices where both ran"]
        for other in series[1:]:
            both = ~np.isnan(times[series[0]]) & ~np.isnan(times[other])
            speedup = times[other][both] / times[series[0]][both]
            pair = f"{labels[series[0]]} over {labels[other]}"
            lines.append(f"  speedup of {pair}: {stats(speedup)}")
        pathlib.Path(f"{stem}_{kind}.txt").write_text("\n".join(lines) + "\n")
        print(f"wrote {stem}_{kind}.pdf, .png and .txt")

        # Above 1 is faster than the reference
        shown = list(series)
        if reference != "cusparse":
            # cuSPARSE is far below the others
            shown.remove("cusparse")
        speedups = {name: times[reference] / times[name] for name in shown}
        path = f"{stem}_speedup_{kind}"
        # A linear axis, unless the speedups are too far apart to read on one
        log = np.nanmax(list(speedups.values())) > 10
        bar_figure(speedups, names, "Speedup", log, 1, labels, path)
        title = f"speedup over {labels[reference]}"
        lines = bar_table(speedups, names, title, labels)
        cells = "".join(f"{geomean(s[~np.isnan(s)]):>24.3f}" for s in speedups.values())
        lines += [f"{'geo-mean':<22}{cells}", "", *failed]
        pathlib.Path(f"{path}.txt").write_text("\n".join(lines) + "\n")
        print(f"wrote {path}.pdf, .png and .txt")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmarks/spmspv.py results")
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument(
        "--bars",
        action="store_true",
        help="one bar plot of every implementation per matrix, for a few matrices",
    )
    parser.add_argument(
        "--reference",
        default="dense",
        help="the series that the speedups of --bars are over",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.csv)

    if args.bars:
        series = ["dense", "esc", "hash", "cusparse", "graphblast"]
        series += LAY
        series = [name for name in series if name in data]
        bars(data, series, args.reference, args.out / "spmspv")
        return

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
    if "lay_atomic_merge" in data:
        # Against the kernel of Li et al. with the same schedule
        for ours, theirs in (("dense", "lay_atomic_merge"), ("esc", "lay_sort_merge")):
            figures(
                data,
                [ours, theirs],
                args.out / f"spmspv_{ours}_vs_lay",
                labels={ours: "Ours", theirs: "Li et al."},
            )


if __name__ == "__main__":
    main()
