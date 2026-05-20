"""
evaluate.py — Model evaluation utilities for ViHSD toxic-comment detection.

This module is *training-agnostic*: it accepts predictions and returns
metrics / plots / archived prediction files. The training loop lives in
src/train.py (Week 3).

Public API:
    evaluate_model        — compute metric dict
    plot_confusion_matrix — seaborn heatmap (raw + %)
    save_predictions      — per-sample CSV for error analysis
    compare_models        — merge multiple metric dicts into a DataFrame
    plot_model_comparison — annotated bar chart, with per-class F1 panel

Run as script for a quick smoke-test:
    python -m src.evaluate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Project root on sys.path ─────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import LABEL_COLORS, LABEL_MAP, PATHS

RANDOM_STATE: int = 42
_LABELS      = sorted(LABEL_MAP.keys())            # [0, 1, 2]
_LABEL_NAMES = [LABEL_MAP[i] for i in _LABELS]     # ['CLEAN', 'OFFENSIVE', 'HATE']


# ─────────────────────────────────────────────────────────────
# 1.  Metric computation
# ─────────────────────────────────────────────────────────────
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    train_time: float = 0.0,
    inference_time: float = 0.0,
) -> Dict[str, Any]:
    """Compute a comprehensive set of metrics for one model.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Ground-truth and predicted labels (0=CLEAN, 1=OFFENSIVE, 2=HATE).
    model_name : str
        Human-readable identifier — used for downstream plots/CSVs.
    train_time, inference_time : float
        Optional wall-clock seconds; pass 0.0 when unknown.

    Returns
    -------
    dict with the following keys:
        model, accuracy,
        precision_macro / recall_macro / f1_macro,
        precision_weighted / recall_weighted / f1_weighted,
        precision_clean / recall_clean / f1_clean,
        precision_offensive / recall_offensive / f1_offensive,
        precision_hate / recall_hate / f1_hate,
        confusion_matrix (np.ndarray 3×3),
        classification_report (str),
        train_time, inference_time.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_p = precision_score(y_true, y_pred, average=None, labels=_LABELS, zero_division=0)
    per_r = recall_score   (y_true, y_pred, average=None, labels=_LABELS, zero_division=0)
    per_f = f1_score       (y_true, y_pred, average=None, labels=_LABELS, zero_division=0)

    return {
        "model": model_name,

        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro":  precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_macro":     recall_score   (y_true, y_pred, average="macro",    zero_division=0),
        "f1_macro":         f1_score       (y_true, y_pred, average="macro",    zero_division=0),

        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score   (y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted":        f1_score       (y_true, y_pred, average="weighted", zero_division=0),

        "precision_clean":     float(per_p[0]),
        "recall_clean":        float(per_r[0]),
        "f1_clean":            float(per_f[0]),

        "precision_offensive": float(per_p[1]),
        "recall_offensive":    float(per_r[1]),
        "f1_offensive":        float(per_f[1]),

        "precision_hate":      float(per_p[2]),
        "recall_hate":         float(per_r[2]),
        "f1_hate":             float(per_f[2]),

        "confusion_matrix":      confusion_matrix(y_true, y_pred, labels=_LABELS),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=_LABELS,
            target_names=_LABEL_NAMES,
            zero_division=0,
        ),

        "train_time":     float(train_time),
        "inference_time": float(inference_time),
    }


# ─────────────────────────────────────────────────────────────
# 2.  Confusion matrix plot
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a 3×3 confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Ground-truth and predicted labels.
    model_name : str
        Used as the figure title.
    save_path : str | Path | None
        If given, save the figure at ``dpi=150``.
    normalize : bool, default True
        When True, cells are coloured by **row-normalised** value
        (diagonal = recall per class). Each cell is annotated with both
        the percentage AND the raw count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm      = confusion_matrix(y_true, y_pred, labels=_LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if normalize:
        sns.heatmap(
            cm_norm,
            annot=False,
            cmap="Blues",
            vmin=0.0, vmax=1.0,
            linewidths=0.5,
            xticklabels=_LABEL_NAMES,
            yticklabels=_LABEL_NAMES,
            cbar_kws={"label": "row-normalised proportion"},
            ax=ax,
        )
        for i in range(len(_LABELS)):
            for j in range(len(_LABELS)):
                pct = cm_norm[i, j]
                raw = cm[i, j]
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{pct:.1%}\n({raw:,})",
                    ha="center", va="center",
                    fontsize=10,
                    color="white" if pct > 0.5 else "black",
                )
        subtitle = "row-normalised (recall view)"
    else:
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            cmap="Blues",
            linewidths=0.5,
            xticklabels=_LABEL_NAMES,
            yticklabels=_LABEL_NAMES,
            ax=ax,
        )
        subtitle = "raw counts"

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name}\nConfusion matrix — {subtitle}")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ─────────────────────────────────────────────────────────────
# 3.  Prediction archival
# ─────────────────────────────────────────────────────────────
def save_predictions(
    texts: Union[pd.Series, List[str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    save_path: Union[str, Path],
) -> pd.DataFrame:
    """Persist per-sample predictions to CSV for Week-5 error analysis.

    Parameters
    ----------
    texts : Series | list[str]
        Raw or cleaned comment strings (length n).
    y_true, y_pred : array-like of int, shape (n,)
    y_proba : array-like, shape (n, 3)
        Class probabilities (column order = CLEAN/OFFENSIVE/HATE).
    save_path : str | Path
        Output CSV path.

    Returns
    -------
    pd.DataFrame
        Columns: text, true_label, true_label_name,
                 predicted_label, predicted_label_name,
                 correct (bool), confidence (max proba),
                 proba_clean, proba_offensive, proba_hate.
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    df = pd.DataFrame({
        "text":                 list(texts),
        "true_label":           y_true,
        "true_label_name":      [LABEL_MAP[int(i)] for i in y_true],
        "predicted_label":      y_pred,
        "predicted_label_name": [LABEL_MAP[int(i)] for i in y_pred],
        "correct":              (y_true == y_pred),
        "confidence":           y_proba.max(axis=1).round(4),
        "proba_clean":          y_proba[:, 0].round(4),
        "proba_offensive":      y_proba[:, 1].round(4),
        "proba_hate":           y_proba[:, 2].round(4),
    })

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df):,} predictions → {save_path}")
    return df


# ─────────────────────────────────────────────────────────────
# 4.  Cross-model comparison
# ─────────────────────────────────────────────────────────────
_COMPARE_COLS: List[str] = [
    "model", "features",
    "accuracy", "f1_macro",
    "f1_clean", "f1_offensive", "f1_hate",
    "precision_macro", "recall_macro",
    "f1_weighted",
    "train_time", "inference_time",
]


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Merge several metric dicts into one comparison table.

    The dict key is parsed as ``"{model}_{features}"`` (e.g.
    ``"LR_TFIDF"``); whatever sits after the last underscore is treated
    as the *features* identifier. If parsing fails the whole key is
    placed in the ``model`` column and ``features`` is left empty.

    Parameters
    ----------
    results_dict
        ``{"LR_BoW": evaluate_model_output, "LR_TFIDF": ..., ...}``.

    Returns
    -------
    pd.DataFrame
        Columns: model, features, accuracy, f1_macro,
        f1_clean, f1_offensive, f1_hate,
        precision_macro, recall_macro, f1_weighted,
        train_time, inference_time.
        Sorted by ``f1_macro`` descending. The verbose keys
        (``confusion_matrix``, ``classification_report``) are dropped.
    """
    rows = []
    for key, metrics in results_dict.items():
        if "_" in key:
            model_id, feat_id = key.rsplit("_", 1)
        else:
            model_id, feat_id = key, ""

        rows.append({
            "model":              model_id,
            "features":           feat_id,
            "accuracy":           metrics.get("accuracy"),
            "f1_macro":           metrics.get("f1_macro"),
            "f1_clean":           metrics.get("f1_clean"),
            "f1_offensive":       metrics.get("f1_offensive"),
            "f1_hate":            metrics.get("f1_hate"),
            "precision_macro":    metrics.get("precision_macro"),
            "recall_macro":       metrics.get("recall_macro"),
            "f1_weighted":        metrics.get("f1_weighted"),
            "train_time":         metrics.get("train_time", 0.0),
            "inference_time":     metrics.get("inference_time", 0.0),
        })

    df = pd.DataFrame(rows, columns=_COMPARE_COLS)
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    float_cols = [c for c in _COMPARE_COLS if c not in ("model", "features", "train_time", "inference_time")]
    df[float_cols] = df[float_cols].apply(lambda s: s.round(4))
    return df


# ─────────────────────────────────────────────────────────────
# 5.  Comparison bar chart
# ─────────────────────────────────────────────────────────────
def plot_model_comparison(
    df: pd.DataFrame,
    metric: str = "f1_macro",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Annotated bar chart comparing models on ``metric``.

    Always renders two panels:
        • LEFT  — overall ``metric`` (default = ``f1_macro``).
        • RIGHT — grouped per-class F1 (CLEAN / OFFENSIVE / HATE)
                  using the project's label colours.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compare_models`.
    metric : str, default 'f1_macro'
        Column of ``df`` to visualise in the left panel.
    save_path : str | Path | None
        If given, save the figure at ``dpi=150``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metric not in df.columns:
        raise KeyError(f"metric '{metric}' not in DataFrame columns")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel 1: single metric (vertical bars) ────────────
    ax = axes[0]
    palette = sns.color_palette("Blues_d", n_colors=len(df))
    labels  = (df["model"] + "\n" + df["features"]).tolist()
    bars    = ax.bar(labels, df[metric], color=palette, edgecolor="white")
    for b, v in zip(bars, df[metric]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, min(1.0, df[metric].max() + 0.1))
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    ax.grid(axis="y", alpha=0.4)
    ax.tick_params(axis="x", rotation=15, labelsize=9)

    # ── Panel 2: per-class F1 (grouped bars) ──────────────
    ax2     = axes[1]
    perclass    = ["f1_clean", "f1_offensive", "f1_hate"]
    class_names = ["CLEAN", "OFFENSIVE", "HATE"]
    colors      = [LABEL_COLORS[i] for i in _LABELS]
    x       = np.arange(len(df))
    width   = 0.25

    for i, (col, lbl, color) in enumerate(zip(perclass, class_names, colors)):
        vals = df[col].tolist()
        bars = ax2.bar(x + (i - 1) * width, vals, width,
                       label=lbl, color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="center", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("F1 score")
    ax2.set_title("Per-class F1")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────
# 6.  Demo on dummy data
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_STATE)
    N = 500
    y_true_demo = rng.choice(_LABELS, size=N, p=[0.83, 0.07, 0.10])

    def _noisy(y: np.ndarray, noise: float) -> np.ndarray:
        mask = rng.random(len(y)) < noise
        out  = y.copy()
        out[mask] = rng.choice(_LABELS, size=mask.sum())
        return out

    y_pred_good = _noisy(y_true_demo, noise=0.15)
    y_pred_weak = _noisy(y_true_demo, noise=0.40)

    # 1. evaluate_model -----------------------------------------------------
    print("=" * 60)
    print("evaluate_model — Demo_Good")
    print("=" * 60)
    metrics_good = evaluate_model(y_true_demo, y_pred_good, "Demo_Good",
                                  train_time=1.20, inference_time=0.05)
    print(metrics_good["classification_report"])

    metrics_weak = evaluate_model(y_true_demo, y_pred_weak, "Demo_Weak",
                                  train_time=0.80, inference_time=0.04)

    # 2. plot_confusion_matrix ---------------------------------------------
    FIG_DIR = Path(PATHS["figures_dir"])
    plot_confusion_matrix(
        y_true_demo, y_pred_good,
        model_name="Demo_Good",
        save_path=FIG_DIR / "demo_confusion_matrix.png",
        normalize=True,
    )
    print("Confusion matrix saved.")

    # 3. save_predictions --------------------------------------------------
    dummy_texts = [f"comment_{i}" for i in range(N)]
    dummy_proba = rng.dirichlet(alpha=[5, 1, 1], size=N)
    pred_path   = Path(PATHS["results_dir"]) / "demo_predictions.csv"
    pred_df     = save_predictions(dummy_texts, y_true_demo, y_pred_good,
                                   dummy_proba, pred_path)
    print("\nPredictions preview:")
    print(pred_df.head(3).to_string(index=False))

    # 4. compare_models ----------------------------------------------------
    print("\n" + "=" * 60)
    print("compare_models")
    print("=" * 60)
    comp_df = compare_models({
        "LR_BoW":   metrics_good,
        "SVM_TFIDF": metrics_weak,
    })
    print(comp_df[["model", "features", "f1_macro",
                   "f1_clean", "f1_offensive", "f1_hate"]].to_string(index=False))

    # 5. plot_model_comparison --------------------------------------------
    plot_model_comparison(
        comp_df,
        metric="f1_macro",
        save_path=FIG_DIR / "demo_model_comparison.png",
    )
    print(f"\nArtefacts saved under: {FIG_DIR} and {pred_path.parent}")