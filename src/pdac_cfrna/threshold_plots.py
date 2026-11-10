"""Plots for threshold selection: precision-recall curves and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from pdac_cfrna.thresholds import confusion_counts  # noqa: E402


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold: float,
    path: Path,
) -> None:
    """Precision-recall curve for each named (y, score) pair, with the chosen threshold marked.

    The dotted horizontal lines are the precision of a model with no skill (the share of PDAC).
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    colours = ["#1f77b4", "#d95f02", "#2ca02c"]
    for (name, (y, s)), colour in zip(curves.items(), colours, strict=False):
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, color=colour, lw=2, label=name)
        ax.axhline(y.mean(), color=colour, lw=1, ls=":")
        c = confusion_counts(y, s, threshold)
        called = c["tp"] + c["fp"]
        ax.scatter(
            [c["tp"] / (c["tp"] + c["fn"])],
            [c["tp"] / called if called else 0.0],
            color=colour,
            s=70,
            zorder=3,
            edgecolor="black",
        )
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-recall curve (dots: threshold {threshold:.2f})")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrices(
    sets: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold: float,
    path: Path,
) -> None:
    """One confusion matrix per named (y, score) pair at the same threshold."""
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 4), dpi=150, squeeze=False)
    for ax, (name, (y, s)) in zip(axes[0], sets.items(), strict=True):
        c = confusion_counts(y, s, threshold)
        grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        share = grid / grid.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                colour = "white" if share[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]}\n({share[i, j]:.0%} of row)",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=11,
                )
        ax.set_xticks([0, 1], ["called control", "called PDAC"])
        ax.set_yticks([0, 1], ["true control", "true PDAC"])
        ax.set_title(f"{name}\nthreshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
