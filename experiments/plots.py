"""Utilities for loading experiment results and generating publication-grade plots.

All experimental metrics are grounded in `experiments/results/*.csv`.
Uses standard Python, NumPy, and Matplotlib (zero external dependencies).
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_table(name: str, numeric: bool = True):
    """Load results/<name>.csv as a list of dictionary records."""
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if numeric:
        for row in rows:
            for key, val in row.items():
                if val == "":
                    row[key] = np.nan
                else:
                    try:
                        row[key] = float(val)
                    except ValueError:
                        pass
    return rows


def show_table(name: str, columns: list[str]):
    """Print selected columns of results/<name>.csv as an aligned text table."""
    rows = load_table(name, numeric=False)
    widths = [max(len(c), max(len(r.get(c, "")) for r in rows)) for c in columns]
    header = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(r.get(c, "").ljust(w) for c, w in zip(columns, widths)))


def plot_model_comparison(rows: list[dict]):
    """Bar chart of 5-fold CV F1 (mean +/- std) across model milestones."""
    colors = {
        "baseline": "#94a3b8",
        "neural net": "#3b82f6",
        "trees": "#f97316",
        "ensemble": "#10b981",
    }
    mean = np.array([r["cv_f1_mean"] for r in rows])
    std = np.nan_to_num(np.array([r["cv_f1_std"] for r in rows]))
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=150)
    bar_colors = [colors.get(r.get("family", ""), "#64748b") for r in rows]
    ax.bar(x, mean, color=bar_colors, edgecolor="black", width=0.55)

    has_std = std > 0
    ax.errorbar(
        x[has_std],
        mean[has_std],
        yerr=std[has_std],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    for i, m in enumerate(mean):
        ax.text(i, m + std[i] + 0.001, f"{m:.4f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in rows], rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0.37, 0.445)
    ax.set_ylabel("5-Fold CV F1 (Dev Set)", fontsize=9)
    ax.set_title(
        "Report Table I: Model Comparison on 80% Dev Folds (Error Bar = Std)",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_f1_vs_rate(rows: list[dict], live: tuple = None, mark: tuple = (0.13, 0.14)):
    """Plot positive-class F1 vs. predicted positive fraction (reproducing Report Fig 1)."""
    rate = np.array([r["rate"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)

    for key, label, color in (
        ("clean", "Official Clean Ensemble (Dev OOF)", "#10b981"),
        ("leaky", "Leaky Audit Benchmark (Dev OOF)", "#ef4444"),
    ):
        m = np.array([r[f"f1_{key}_mean"] for r in rows])
        s = np.array([r[f"f1_{key}_std"] for r in rows])
        ax.plot(rate, m, color=color, lw=2.0, label=label)
        ax.fill_between(rate, m - s, m + s, color=color, alpha=0.18)

    if live is not None:
        ax.plot(
            live[0],
            live[1],
            "--",
            color="#64748b",
            lw=1.5,
            label="Live Sample: Reg. Logistic",
        )

    for r in mark:
        ax.axvline(r, color="black", lw=1.0, ls=":")
        side = "right" if r == min(mark) else "left"
        ax.text(
            r,
            ax.get_ylim()[0] + 0.005,
            f" {int(round(100 * r))}% ",
            ha=side,
            fontsize=8,
            fontweight="bold",
        )

    ax.axvline(0.088, color="#94a3b8", lw=0.9, ls="--")
    ax.text(
        0.088,
        ax.get_ylim()[1] - 0.015,
        " Prevalence 8.83%",
        ha="left",
        va="top",
        fontsize=8,
        color="#475569",
    )
    ax.set_xlabel("Share of Rows Predicted Positive", fontsize=9)
    ax.set_ylabel("Positive-Class F1", fontsize=9)
    ax.set_title(
        "Report Fig. 1: F1 vs. Calibrated Decision Threshold (Plateau at 12–16%)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_leakage(rows: list[dict]):
    """Grouped bar chart comparing clean vs leaky inputs across model families."""
    models = list(dict.fromkeys(r["model"] for r in rows))
    lookup = {(r["model"], r["variant"]): r for r in rows}
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    palette = (
        ("clean", "#10b981", "Clean (Official)"),
        ("clean+override", "#cbd5e1", "Clean + HAREHAB1"),
        ("leaky", "#ef4444", "Leaky (All 7 Features)"),
    )
    width = 0.26
    for j, (variant, color, label) in enumerate(palette):
        m = [lookup[(name, variant)]["nested_f1_mean"] for name in models]
        s = [lookup[(name, variant)]["nested_f1_std"] for name in models]
        ax.bar(
            x + (j - 1) * width,
            m,
            width,
            yerr=s,
            capsize=2,
            color=color,
            edgecolor="black",
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.41, 0.452)
    ax.set_ylabel("Nested CV F1", fontsize=9)
    ax.set_title(
        "Report Section V: Target Leakage Impact Across Models (+0.010 F1 Inflation)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_importance(rows: list[dict], top: int = 10):
    """Horizontal bar chart of permutation AUC drop for top predictive features."""
    top_rows = rows[:top][::-1]
    y = np.arange(len(top_rows))
    drops = [r["auc_drop_mean"] for r in top_rows]
    errs = [r["auc_drop_sd"] for r in top_rows]

    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(top_rows) + 1.0), dpi=150)
    ax.barh(
        y,
        drops,
        xerr=errs,
        color="#f97316",
        edgecolor="black",
        capsize=2,
        height=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r["column"] for r in top_rows], fontsize=9)
    ax.set_xlabel("Permutation AUC Drop (Mean +/- SD across Folds)", fontsize=9)
    ax.set_title(
        f"Report Section IV: Top {top} Permutation Feature Importance (GBM)",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_rejected(rows: list[dict], max_abs: float = 0.005):
    """Horizontal bar chart of F1 impact for rejected modeling hypotheses."""
    filtered = [r for r in rows if abs(r["delta_f1"]) <= max_abs]
    filtered = sorted(filtered, key=lambda r: r["delta_f1"])
    deltas = np.array([r["delta_f1"] for r in filtered])
    y = np.arange(len(filtered))

    fig, ax = plt.subplots(figsize=(8.0, 0.28 * len(filtered) + 1.2), dpi=150)
    colors = np.where(deltas > 0, "#3b82f6", "#ef4444")
    ax.barh(y, deltas, color=colors, edgecolor="black", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['experiment'][:40]} [{r.get('variant', 'clean')}]" for r in filtered],
        fontsize=8,
    )
    ax.axvline(0, color="black", lw=0.9)
    ax.axvline(0.001, color="black", lw=0.8, ls=":")
    ax.text(
        0.001,
        len(filtered) - 0.5,
        " Acceptance (+0.001)",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("Validation F1 Delta vs. Clean Reference Baseline", fontsize=9)
    ax.set_title(
        "Report Section VI: Rigorous Audit of Rejected Hypotheses",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig
