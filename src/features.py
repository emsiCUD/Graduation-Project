"""
features.py — Fit and persist bag-of-words / TF-IDF feature matrices.

IMPORTANT (no data leakage):
    All vectorizers are fitted **exclusively on the training split** and
    then applied (transform-only) to dev/test.

Usage – fit from scratch:
    python -m src.features

Usage – import in a notebook:
    from src.features import fit_vectorizers, transform_and_save, load_vectorizers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm

# ── Ensure project root is on sys.path when run as a script ──
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import COLUMNS, PATHS
from src.preprocess import VietnameseTextCleaner, batch_clean

# ── Constants ────────────────────────────────────────────────
VECTORIZER_DIR = Path(_ROOT) / "models" / "vectorizers"
PROCESSED_DIR  = Path(PATHS["processed_dir"])

_COMMON_KWARGS: dict = dict(
    max_features=10_000,
    min_df=2,
    max_df=0.95,
    analyzer="word",
)

VECTORIZER_SPECS: Dict[str, dict] = {
    "bow":         {"cls": CountVectorizer,  "kwargs": {}},
    "tfidf_uni":   {"cls": TfidfVectorizer,  "kwargs": {"sublinear_tf": True}},
    "tfidf_unibi": {"cls": TfidfVectorizer,  "kwargs": {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
    }},
}


# ── Public API ───────────────────────────────────────────────
def fit_vectorizers(
    train_texts: pd.Series,
    save_dir: Optional[Path | str] = None,
) -> Dict[str, CountVectorizer | TfidfVectorizer]:
    """Fit one CountVectorizer and two TF-IDF vectorizers on *train* texts only.

    Parameters
    ----------
    train_texts
        Cleaned training texts (output of :func:`batch_clean`).
    save_dir
        Directory where fitted ``.pkl`` files are written.
        Defaults to ``models/vectorizers/``.

    Returns
    -------
    dict
        Keys: ``"bow"``, ``"tfidf_uni"``, ``"tfidf_unibi"``.
        Values: fitted vectorizer objects.

    Notes
    -----
    Saving is done with :func:`joblib.dump` which handles sparse objects
    efficiently. The three files written are::

        bow.pkl
        tfidf_uni.pkl
        tfidf_unibi.pkl
    """
    save_dir = Path(save_dir) if save_dir else VECTORIZER_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    filled: pd.Series = train_texts.fillna("").astype(str)
    vectorizers: Dict[str, CountVectorizer | TfidfVectorizer] = {}

    for name, spec in tqdm(VECTORIZER_SPECS.items(), desc="Fitting vectorizers"):
        kwargs = {**_COMMON_KWARGS, **spec["kwargs"]}
        vec = spec["cls"](**kwargs)
        vec.fit(filled)
        vectorizers[name] = vec

        out_path = save_dir / f"{name}.pkl"
        joblib.dump(vec, out_path)
        print(f"  [{name}] vocab size={len(vec.vocabulary_):,}  →  {out_path}")

    return vectorizers


def transform_and_save(
    vectorizers: Dict[str, CountVectorizer | TfidfVectorizer],
    texts_dict: Dict[str, pd.Series],
    labels_dict: Dict[str, pd.Series],
    save_dir: Optional[Path | str] = None,
) -> None:
    """Transform all splits and persist sparse matrices + label arrays.

    Parameters
    ----------
    vectorizers
        Dict returned by :func:`fit_vectorizers` (keys: bow, tfidf_uni,
        tfidf_unibi).
    texts_dict
        ``{"train": pd.Series, "dev": pd.Series, "test": pd.Series}``
        of *cleaned* texts.
    labels_dict
        ``{"train": pd.Series, "dev": pd.Series, "test": pd.Series}``
        of integer label IDs.
    save_dir
        Destination directory. Defaults to ``data/processed/``.

    File naming convention::

        X_{split}_{feature}.npz   (scipy sparse matrix)
        y_{split}.npy              (numpy int32 array, written once per split)
    """
    save_dir = Path(save_dir) if save_dir else PROCESSED_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    for split, texts in texts_dict.items():
        filled = texts.fillna("").astype(str)

        # Labels – saved once per split
        y = labels_dict[split].to_numpy(dtype=np.int32)
        y_path = save_dir / f"y_{split}.npy"
        np.save(y_path, y)
        print(f"  [y_{split}]  shape={y.shape}  →  {y_path}")

        # Feature matrices
        for feat_name, vec in vectorizers.items():
            X: sp.csr_matrix = vec.transform(filled)
            x_path = save_dir / f"X_{split}_{feat_name}.npz"
            sp.save_npz(str(x_path), X)
            print(f"  [X_{split}_{feat_name}]  shape={X.shape}  nnz={X.nnz:,}  →  {x_path}")


def load_vectorizers(
    save_dir: Optional[Path | str] = None,
) -> Dict[str, CountVectorizer | TfidfVectorizer]:
    """Load previously serialised vectorizers from ``save_dir``.

    Parameters
    ----------
    save_dir
        Directory containing the ``.pkl`` files. Defaults to
        ``models/vectorizers/``.

    Returns
    -------
    dict
        Same structure as returned by :func:`fit_vectorizers`.
    """
    save_dir = Path(save_dir) if save_dir else VECTORIZER_DIR
    vectorizers: dict = {}
    for name in VECTORIZER_SPECS:
        path = save_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Vectorizer '{name}' not found at {path}. "
                "Run fit_vectorizers() first."
            )
        vectorizers[name] = joblib.load(path)
    return vectorizers


def load_feature_matrix(
    split: str,
    feature: str,
    data_dir: Optional[Path | str] = None,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Convenience loader for a single (split, feature) pair.

    Parameters
    ----------
    split
        One of ``"train"``, ``"dev"``, ``"test"``.
    feature
        One of ``"bow"``, ``"tfidf_uni"``, ``"tfidf_unibi"``.
    data_dir
        Root of processed data. Defaults to ``data/processed/``.

    Returns
    -------
    X : scipy.sparse.csr_matrix
    y : numpy.ndarray (int32)
    """
    data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
    X = sp.load_npz(str(data_dir / f"X_{split}_{feature}.npz"))
    y = np.load(data_dir / f"y_{split}.npy")
    return X, y


# ── Main pipeline ─────────────────────────────────────────────
def _run_pipeline() -> None:
    """End-to-end: load raw CSVs → clean → fit → transform → save."""
    text_col  = COLUMNS["text"]
    label_col = COLUMNS["label"]

    # 1. Load raw splits
    print("=" * 55)
    print("Step 1 / 4 — Loading raw CSVs")
    print("=" * 55)
    raw: Dict[str, pd.DataFrame] = {
        split: pd.read_csv(PATHS[f"raw_{split}"])
        for split in ("train", "dev", "test")
    }
    for split, df in raw.items():
        print(f"  {split}: {len(df):,} rows")

    # 2. Clean text
    print("\n" + "=" * 55)
    print("Step 2 / 4 — Cleaning text")
    print("=" * 55)
    cleaner = VietnameseTextCleaner()
    texts: Dict[str, pd.Series] = {}
    labels: Dict[str, pd.Series] = {}
    for split, df in raw.items():
        texts[split]  = batch_clean(df[text_col], cleaner=cleaner, desc=split)
        labels[split] = df[label_col].astype(int)

    # 3. Fit vectorizers on train only
    print("\n" + "=" * 55)
    print("Step 3 / 4 — Fitting vectorizers (train only)")
    print("=" * 55)
    vecs = fit_vectorizers(texts["train"])

    # 4. Transform all splits and save
    print("\n" + "=" * 55)
    print("Step 4 / 4 — Transforming and saving")
    print("=" * 55)
    transform_and_save(vecs, texts, labels)

    # Summary table
    print("\n" + "=" * 55)
    print("Summary — feature matrix shapes")
    print("=" * 55)
    header = f"{'split':<6}  {'feature':<14}  {'rows':>8}  {'cols':>8}  {'nnz':>12}"
    print(header)
    print("-" * len(header))
    for split in ("train", "dev", "test"):
        for feat in ("bow", "tfidf_uni", "tfidf_unibi"):
            X, y = load_feature_matrix(split, feat)
            print(
                f"{split:<6}  {feat:<14}  {X.shape[0]:>8,}  "
                f"{X.shape[1]:>8,}  {X.nnz:>12,}"
            )
    print("\nDone. Vectorizers in:", VECTORIZER_DIR)
    print("Feature matrices in:", PROCESSED_DIR)


if __name__ == "__main__":
    _run_pipeline()
