"""Load the result tables of experiments/results and draw the notebook plots.

The csv files are transcribed from the experiment logs (column "source");
nothing here retrains a model. Used by notebooks/03_experiments.ipynb.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_table(name, numeric=True):
    """Read results/<name>.csv as a list of dicts.

    numeric: convert number cells to float (empty cell = nan); False keeps
    the text exactly as written in the csv.
    """
    with open(os.path.join(RESULTS, name + ".csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if numeric:
        for row in rows:
            for key, value in row.items():
                try:
                    row[key] = float(value) if value != "" else np.nan
                except ValueError:
                    pass
    return rows


def show_table(name, columns):
    """Print the given columns of results/<name>.csv as aligned text."""
    rows = load_table(name, numeric=False)
    width = [max(len(c), *(len(r[c]) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, width)))
    for r in rows:
        print("  ".join(r[c].ljust(w) for c, w in zip(columns, width)))


def plot_model_comparison(rows):
    """Bar chart of the CV F1 (mean +/- std over folds) of every model."""
    colors = {
        "baseline": "#9aa5b1",
        "neural net": "#4c78a8",
        "trees": "#f58518",
        "ensemble": "#54a24b",
    }
    mean = np.array([r["cv_f1_mean"] for r in rows])
    std = np.nan_to_num(np.array([r["cv_f1_std"] for r in rows]))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(rows))
    ax.bar(x, mean, color=[colors[r["family"]] for r in rows])
    has = std > 0  # no std logged for some baselines
    ax.errorbar(x[has], mean[has], yerr=std[has], fmt="none", ecolor="k", capsize=3)
    for i, m in enumerate(mean):
        ax.text(i, m + std[i] + 0.0005, "%.4f" % m, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in rows], rotation=25, ha="right")
    ax.set_ylim(0.405, 0.442)
    ax.set_ylabel("CV F1 (5-fold dev)")
    ax.set_title("Model comparison (error bar = std over folds)")
    fig.tight_layout()
    return fig


def plot_f1_vs_rate(rows, live=None, mark=(0.13, 0.14)):
    """F1 vs share of rows predicted positive (mean over folds, band = std).

    live: optional (rates, f1) of the live subsample illustration.
    """
    rate = np.array([r["rate"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, label, color in (
        ("clean", "official GBM+MLP (dev OOF)", "#54a24b"),
        ("leaky", "with leaky columns (dev OOF)", "#e45756"),
    ):
        m = np.array([r["f1_%s_mean" % key] for r in rows])
        s = np.array([r["f1_%s_std" % key] for r in rows])
        ax.plot(rate, m, color=color, label=label)
        ax.fill_between(rate, m - s, m + s, color=color, alpha=0.2)
    if live is not None:
        ax.plot(
            live[0], live[1], "--", color="#9aa5b1", label="live: reg. logistic, 30k"
        )
    for r in mark:
        ax.axvline(r, color="k", lw=0.8, ls=":")
        side = "right" if r == min(mark) else "left"
        label = " %d%% " % round(100 * r)
        ax.text(r, ax.get_ylim()[0], label, ha=side, fontsize=8, va="bottom")
    ax.axvline(0.088, color="#9aa5b1", lw=0.8)
    ax.text(
        0.088, ax.get_ylim()[1], " prevalence 8.8%", ha="left", va="top", fontsize=8
    )
    ax.set_xlabel("share of rows predicted +1")
    ax.set_ylabel("F1")
    ax.set_title("Decision rule: predict +1 for the top-scored share")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_rejected(rows, max_abs=0.005):
    """Horizontal bars of the F1 change of every rejected alternative.

    Rows with |delta| > max_abs are left out of the plot (they stay in the table).
    """
    rows = [r for r in rows if abs(r["delta_f1"]) <= max_abs]
    rows = sorted(rows, key=lambda r: r["delta_f1"])
    delta = np.array([r["delta_f1"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(rows) + 1))
    y = np.arange(len(rows))
    colors = np.where(delta > 0, "#4c78a8", "#e45756")
    ax.barh(y, delta, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(
        ["%s [%s]" % (r["experiment"], r["variant"]) for r in rows], fontsize=8
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.001, color="k", lw=0.8, ls=":")
    ax.text(0.001, len(rows) - 0.5, " acceptance +0.001", fontsize=8)
    ax.set_xlabel("F1 change vs reference")
    ax.set_title("Rejected alternatives")
    fig.tight_layout()
    return fig


def plot_leakage(rows):
    """Grouped bars: nested CV F1 of every model, clean vs leaky inputs."""
    models = list(dict.fromkeys(r["model"] for r in rows))
    get = dict(((r["model"], r["variant"]), r) for r in rows)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, (variant, color) in enumerate(
        (("clean", "#54a24b"), ("clean+override", "#b8b8b8"), ("leaky", "#e45756"))
    ):
        m = [get[(name, variant)]["nested_f1_mean"] for name in models]
        s = [get[(name, variant)]["nested_f1_std"] for name in models]
        ax.bar(
            x + (j - 1) * 0.27, m, 0.27, yerr=s, capsize=2, color=color, label=variant
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.41, 0.45)
    ax.set_ylabel("nested CV F1 (5-fold dev)")
    ax.set_title("Leakage ablation")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    return fig


def plot_importance(rows, top=15):
    """Horizontal bars of the permutation AUC drop (mean +/- sd over folds)."""
    rows = rows[:top][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(rows) + 1))
    ax.barh(
        y,
        [r["auc_drop_mean"] for r in rows],
        xerr=[r["auc_drop_sd"] for r in rows],
        color="#f58518",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r["column"] for r in rows], fontsize=8)
    ax.set_xlabel("AUC drop when the column is shuffled")
    ax.set_title("GBM feature importance (top %d)" % top)
    fig.tight_layout()
    return fig
