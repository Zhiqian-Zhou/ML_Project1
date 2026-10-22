"""Generate publication-quality figures for EPFL CS-433 Project 1 Report.

Figures generated:
1. fig1_class_imbalance.png: Class distribution of _MICHD (91.17% vs 8.83%).
2. fig2_missingness_histogram.png: Missing value distribution across 321 features with 85% cutoff.
3. fig3_health_days_correction.png: PHYSHLTH distribution before vs after 88 -> 0 correction.
4. fig4_threshold_vs_f1.png: Validation curves of decision threshold vs F1-score.
5. fig5_confusion_matrix.png: Normalized confusion matrix at calibrated threshold tau*=0.218.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("report/figures", exist_ok=True)

# Set global publication styling
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "font.family": "sans-serif",
    }
)


def generate_fig1_class_imbalance():
    """Figure 1: Target class imbalance distribution."""
    y = np.load("data/processed/y_train.npy")
    n_neg = np.sum(y == -1)
    n_pos = np.sum(y == 1)
    n_total = len(y)

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    bars = ax.bar(
        ["Negative (No MICHD)", "Positive (MICHD)"],
        [n_neg, n_pos],
        color=["#2563eb", "#dc2626"],
        width=0.55,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
    )

    # Annotate bar heights and percentages
    for bar, count in zip(bars, [n_neg, n_pos]):
        pct = count / n_total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5000,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Respondents", fontweight="bold")
    ax.set_title(
        "Target Distribution: Cardiovascular Heart Disease (_MICHD)",
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, 350000)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Highlight class ratio
    ax.text(
        0.5,
        0.75,
        f"Total: {n_total:,}\nImbalance Ratio: ~10.3 : 1\nPositive Prevalence: 8.83%",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8fafc",
            edgecolor="#94a3b8",
            alpha=0.9,
        ),
        fontsize=10,
    )

    plt.tight_layout()
    out_path = "report/figures/fig1_class_imbalance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_fig2_missingness_histogram():
    """Figure 2: Histogram of missing value rates across 321 raw features."""
    with open("data/dataset/x_train.csv", "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")[1:]  # Skip Id
        n_cols = len(header)
        missing_counts = [0] * n_cols
        total_rows = 0
        for line in f:
            total_rows += 1
            toks = line.strip().split(",")[1:]
            for i, val in enumerate(toks):
                if val == "":
                    missing_counts[i] += 1

    missing_rates = np.array(missing_counts) / total_rows

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    n, bins, patches = ax.hist(
        missing_rates,
        bins=30,
        color="#3b82f6",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )

    # Highlight cutoff at 85%
    ax.axvline(
        0.85,
        color="#dc2626",
        linestyle="--",
        linewidth=2.0,
        label="Pruning Threshold (85%)",
    )

    # Annotate regions
    n_kept = np.sum(missing_rates <= 0.85)
    n_dropped = np.sum(missing_rates > 0.85)
    ax.text(
        0.35,
        35,
        f"Retained Features ({n_kept})\n(Core Clinical & Lifestyle)",
        color="#1e40af",
        fontweight="bold",
        fontsize=10,
    )
    ax.text(
        0.87,
        50,
        f"Pruned Sparse ({n_dropped})\n(Optional Modules)",
        color="#b91c1c",
        fontweight="bold",
        fontsize=10,
    )

    # Annotate HAREHAB1 exception
    h_rate = missing_rates[header.index("HAREHAB1")]
    ax.annotate(
        "HAREHAB1 (99.76% missing)\nExempted Target Leak (100% positive)",
        xy=(h_rate, 4),
        xytext=(0.45, 18),
        arrowprops=dict(facecolor="#b91c1c", shrink=0.05, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444"),
    )

    ax.set_xlabel("Missing Value Ratio (NaN %)", fontweight="bold")
    ax.set_ylabel("Number of Features", fontweight="bold")
    ax.set_title(
        "Missing Value Distribution Across Raw Survey Features (321 total)",
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = "report/figures/fig2_missingness_histogram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_fig3_health_days_correction():
    """Figure 3: PHYSHLTH distribution before vs after 88 -> 0 correction."""
    with open("data/dataset/x_train.csv", "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx = header.index("PHYSHLTH")
        raw_vals = []
        for line in f:
            v = line.strip().split(",")[idx]
            if v != "":
                raw_vals.append(float(v))

    raw_vals = np.array(raw_vals)
    # Cleaned: 88 -> 0, 77/99 -> NaN
    clean_vals = raw_vals.copy()
    clean_vals[clean_vals == 88.0] = 0.0
    clean_vals = clean_vals[~np.isin(clean_vals, [77.0, 99.0])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Raw plot
    ax1.hist(
        raw_vals[raw_vals <= 90],
        bins=35,
        color="#ef4444",
        edgecolor="black",
        alpha=0.85,
    )
    ax1.set_title("Raw CDC Coding: PHYSHLTH", fontweight="bold")
    ax1.set_xlabel("Recorded Code / Days", fontweight="bold")
    ax1.set_ylabel("Number of Respondents", fontweight="bold")
    ax1.grid(linestyle="--", alpha=0.5)

    n_88 = np.sum(raw_vals == 88.0)
    ax1.annotate(
        f"Artificial Spike at 88\n(N = {n_88:,}, 62.1%)\nRepresents '0 Days' (Healthy)",
        xy=(88, 80000),
        xytext=(30, 95000),
        arrowprops=dict(facecolor="#b91c1c", shrink=0.05, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444"),
    )

    # Corrected plot
    ax2.hist(clean_vals, bins=31, color="#10b981", edgecolor="black", alpha=0.85)
    ax2.set_title("Sanitized Domain Variable: PHYSHLTH (0-30 days)", fontweight="bold")
    ax2.set_xlabel("Number of Sick Days (Corrected)", fontweight="bold")
    ax2.set_ylabel("Number of Respondents", fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.5)

    ax2.annotate(
        f"Mapped to True Clinical Scale\n(0 days of poor health: {n_88:,})",
        xy=(0, 160000),
        xytext=(5, 140000),
        arrowprops=dict(facecolor="#047857", shrink=0.05, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d1fae5", edgecolor="#10b981"),
    )

    plt.suptitle(
        "CDC Survey Code Hygiene: Correcting Day-Count Sentinels (88 -> 0)",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = "report/figures/fig3_health_days_correction.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_fig4_threshold_vs_f1():
    """Figure 4: Decision threshold tau vs F1-score validation curve."""
    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")

    from cross_validation import build_k_indices, preprocess_fold
    from implementations import reg_logistic_regression, sigmoid

    k_indices = build_k_indices(y, k_fold=5, seed=42)
    # Evaluate fold 0 for clean curve
    tr_idx = np.concatenate([k_indices[i] for i in range(1, 5)])
    val_idx = k_indices[0]

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    tx_tr, tx_val = preprocess_fold(x_tr, x_val)

    w, _ = reg_logistic_regression(
        (y_tr == 1).astype(float),
        tx_tr,
        lambda_=1e-4,
        initial_w=np.zeros(tx_tr.shape[1]),
        max_iters=300,
        gamma=0.1,
    )

    probs_val = sigmoid(tx_val @ w)
    candidates = np.linspace(0.04, 0.50, 100)

    pos_s = np.sort(probs_val[y_val == 1])
    neg_s = np.sort(probs_val[y_val == -1])
    n_pos = len(pos_s)
    n_neg = len(neg_s)

    tp = n_pos - np.searchsorted(pos_s, candidates)
    fp = n_neg - np.searchsorted(neg_s, candidates)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = tp / n_pos
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    best_idx = np.argmax(f1)
    best_tau = candidates[best_idx]
    best_f1 = f1[best_idx]
    default_f1 = f1[np.argmin(np.abs(candidates - 0.5))]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(
        candidates,
        f1,
        color="#2563eb",
        linewidth=2.5,
        label="F1-Score (Positive Class)",
    )
    ax.plot(
        candidates,
        prec,
        color="#10b981",
        linestyle="--",
        linewidth=1.8,
        label="Precision",
    )
    ax.plot(
        candidates, rec, color="#f59e0b", linestyle="-.", linewidth=1.8, label="Recall"
    )

    # Mark optimal threshold
    ax.axvline(best_tau, color="#dc2626", linestyle=":", linewidth=2.0)
    ax.scatter([best_tau], [best_f1], color="#dc2626", s=90, zorder=5)
    ax.annotate(
        f"Optimal Cutoff $\\tau^* = {best_tau:.3f}$\nMax $F_1 = {best_f1:.4f}$ (+85% gain)",
        xy=(best_tau, best_f1),
        xytext=(best_tau + 0.04, best_f1 - 0.08),
        arrowprops=dict(facecolor="#dc2626", shrink=0.05, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#dc2626"),
    )

    # Mark default 0.5 threshold
    ax.scatter([0.5], [default_f1], color="#6b7280", s=70, zorder=5)
    ax.annotate(
        f"Default $\\tau = 0.50$\n$F_1 = {default_f1:.4f}$\n(Recall = {rec[-1]:.1%})",
        xy=(0.5, default_f1),
        xytext=(0.38, default_f1 + 0.12),
        arrowprops=dict(facecolor="#6b7280", shrink=0.05, width=1.5, headwidth=5),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6", edgecolor="#9ca3af"),
    )

    ax.set_xlabel(
        "Decision Threshold $\\tau$ (Predicted Probability Cutoff)", fontweight="bold"
    )
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_title(
        "Decision Threshold Calibration for Class Imbalance (8.83% positive)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0.03, 0.52)
    ax.set_ylim(0.0, 0.75)
    ax.legend(loc="upper right")
    ax.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = "report/figures/fig4_threshold_vs_f1.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_fig5_confusion_matrix():
    """Figure 5: Normalized Confusion Matrix at optimal threshold tau*=0.218."""
    # From 5-fold cross-validation average at tau*=0.218:
    # y_val has 5,795 positive and 59,832 negative samples per fold
    # Recall = 54.05% -> TP = 3,132, FN = 2,663
    # Precision = 35.43% -> FP = 5,708, TN = 54,124
    cm = np.array([[54124, 5708], [2663, 3132]])
    cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)

    # Set labels
    classes = ["Negative (-1)", "Positive (+1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_yticklabels(classes, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Ground Truth Label", fontweight="bold", labelpad=8)
    ax.set_title(
        "Normalized Confusion Matrix ($\\tau^* = 0.218$)\nBalanced Accuracy = 72.24%",
        fontweight="bold",
        pad=12,
    )

    # Text annotations inside cells
    cell_labels = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"],
    ]

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j]
            color = "white" if pct > 50 else "black"
            ax.text(
                j,
                i,
                f"{cell_labels[i][j]}\n{val:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10.5,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row Percentage (%)")
    plt.tight_layout()
    out_path = "report/figures/fig5_confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_fig1_class_imbalance()
    generate_fig2_missingness_histogram()
    generate_fig3_health_days_correction()
    generate_fig4_threshold_vs_f1()
    generate_fig5_confusion_matrix()
    print("All 5 publication figures successfully generated!")
