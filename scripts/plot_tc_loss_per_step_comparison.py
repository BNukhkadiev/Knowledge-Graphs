#!/usr/bin/env python3
"""Plot interval loss vs training step for no-pretrain (V), P1, and P2 runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def latest_timestamped_run(run_root: Path) -> Path | None:
    """Pick the newest subdirectory under ``run_root`` (by mtime)."""
    if not run_root.is_dir():
        return None
    subs = [p for p in run_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)


def load_steps_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("step", "interval_loss"):
        if col not in df.columns:
            raise ValueError(f"{path}: expected column {col!r}")
    return df[["step", "interval_loss"]].copy()


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.9,
        }
    )


def plot_tc(
    tc_id: str,
    paths: dict[str, Path],
    out_path: Path,
) -> None:
    series = {
        "V (no pretrain)": ("#2563eb", paths["v"]),
        "P1": ("#c026d3", paths["p1"]),
        "P2": ("#ea580c", paths["p2"]),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.25))

    for label, (color, csv_path) in series.items():
        df = load_steps_csv(csv_path)
        df = df.sort_values("step")
        ax.plot(
            df["step"],
            df["interval_loss"],
            label=label,
            color=color,
            linewidth=2.15,
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            alpha=0.95,
        )

    ax.set_xlabel("Training step (batch index)")
    ax.set_ylabel("Interval loss")
    ax.set_title(f"{tc_id.upper()} — loss per step during Word2Vec training")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Repository output directory (default: ./output)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("latex/assets"),
        help="Directory for PNG files (default: latex/assets)",
    )
    args = parser.parse_args()
    output_root: Path = args.output_root
    out_dir: Path = args.out_dir

    apply_plot_style()

    for tc in ("07", "08", "09", "10"):
        v_root = output_root / f"tc{tc}_rdf2vec"
        p1_root = output_root / f"tc{tc}_rdf2vec_p1"
        p2_root = output_root / f"tc{tc}_rdf2vec_p2"

        v_run = latest_timestamped_run(v_root)
        p1_run = latest_timestamped_run(p1_root)
        p2_run = latest_timestamped_run(p2_root)
        if v_run is None or p1_run is None or p2_run is None:
            raise SystemExit(
                f"Missing run directory for tc{tc}: "
                f"v={v_run}, p1={p1_run}, p2={p2_run}"
            )

        v_csv = v_run / "rdf2vec_word2vec_loss_steps.csv"
        p1_csv = p1_run / "finetune_loss_steps.csv"
        p2_csv = p2_run / "finetune_loss_steps.csv"
        for p in (v_csv, p1_csv, p2_csv):
            if not p.is_file():
                raise SystemExit(f"Missing CSV: {p}")

        out_png = out_dir / f"tc{tc}_loss_per_step_v_p1_p2.png"
        plot_tc(
            f"tc{tc}",
            {"v": v_csv, "p1": p1_csv, "p2": p2_csv},
            out_png,
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
