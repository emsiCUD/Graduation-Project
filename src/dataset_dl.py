"""
dataset_dl.py — PyTorch ``Dataset`` plumbing for Week-4 deep-learning models.

Provides everything the BiLSTM and PhoBERT training scripts will need:

* :class:`Vocab`             – word-level vocabulary built from training texts.
* :class:`ViHSDDataset`      – ``torch.utils.data.Dataset`` with two modes
  (``'bilstm'`` or ``'phobert'``).
* :class:`FilteredViHSDDataset` – training-only wrapper that drops samples
  shorter than ``min_length`` (avoids NaN-gradient corner cases).
* :func:`collate_fn_bilstm`  – pads variable-length token-id sequences.
* :func:`collate_fn_phobert` – stacks pre-padded HF tokenizer output.
* :func:`build_embedding_matrix` – read a PhoW2V / fastText ``.vec`` file
  and align it with the project vocabulary.
* :func:`get_class_weights`  – inverse-frequency class weights for an
  imbalanced 3-class loss.

The BiLSTM side intentionally uses a *word-level* tokenizer (``str.split``
applied to already-cleaned text) so the surface form matches what
``02_preprocessing.ipynb`` produced and what PhoW2V was trained on.

Run as a script for a tiny smoke test (no GPU required):

    python -m src.dataset_dl
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Ensure project root importable when run as a script ──────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ──────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────

PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<bos>", "<eos>"
PAD_ID,    UNK_ID,    BOS_ID,    EOS_ID    = 0, 1, 2, 3


class Vocab:
    """Word-level vocabulary with frequency-based pruning.

    Indices ``0..3`` are reserved for ``<pad>``, ``<unk>``, ``<bos>``,
    ``<eos>``. Tokens with fewer than ``min_freq`` occurrences are mapped
    to ``<unk>``. When ``max_size`` is set, only the most frequent
    ``max_size - len(specials)`` tokens are kept.
    """

    SPECIALS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)

    def __init__(self) -> None:
        self.itos: List[str] = list(self.SPECIALS)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.freqs: Counter = Counter()

    # ── Construction ──────────────────────────────────────────

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        min_freq: int = 2,
        max_size: int = 20_000,
    ) -> "Vocab":
        """Build a vocabulary from cleaned, whitespace-tokenisable texts."""
        vocab = cls()
        for txt in texts:
            if not isinstance(txt, str):
                continue
            vocab.freqs.update(txt.split())

        keep_n = max_size - len(cls.SPECIALS)
        # Counter.most_common is stable; deterministic tie-breaking via token order below
        candidates = [tok for tok, c in vocab.freqs.most_common() if c >= min_freq]
        # If we hit the freq cutoff before max_size, candidates is shorter — fine.
        for tok in candidates[:keep_n]:
            vocab.stoi[tok] = len(vocab.itos)
            vocab.itos.append(tok)
        return vocab

    # ── Encoding ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.itos)

    def text_to_ids(
        self,
        text: str,
        max_len: Optional[int] = None,
        add_special: bool = False,
    ) -> List[int]:
        """Convert a cleaned text to a list of token ids.

        Unknown tokens map to ``<unk>``. When ``add_special=True`` the
        sequence is wrapped with ``<bos>``/``<eos>``. ``max_len`` truncates
        (post-special-tokens) but does not pad — padding is the
        collate-fn's job.
        """
        ids = [self.stoi.get(tok, UNK_ID) for tok in text.split()]
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    # ── Serialisation ─────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"itos": self.itos, "freqs": dict(self.freqs)}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocab":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        vocab = cls()
        vocab.itos = list(blob["itos"])
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        vocab.freqs = Counter(blob.get("freqs", {}))
        return vocab


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class ViHSDDataset(Dataset):
    """Unified ``Dataset`` for both BiLSTM and PhoBERT pipelines.

    Parameters
    ----------
    texts
        Iterable of *cleaned* Vietnamese strings (same surface form as
        ``data/processed/{split}_cleaned.csv``'s ``cleaned`` column).
    labels
        Iterable of integer labels in ``{0, 1, 2}``.
    tokenizer
        * ``mode='bilstm'``  → must be a :class:`Vocab` instance.
        * ``mode='phobert'`` → must be a HuggingFace ``PreTrainedTokenizer``
          (or fast variant).
    max_len
        Maximum sequence length. PhoBERT pads/truncates here; BiLSTM
        truncates here and the collate-fn pads to the batch's longest.
    mode
        ``'bilstm'`` or ``'phobert'``.
    """

    VALID_MODES = ("bilstm", "phobert")

    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer,
        max_len: int = 256,
        mode: str = "bilstm",
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self.texts = [str(t) if t is not None else "" for t in texts]
        self.labels = [int(y) for y in labels]
        if len(self.texts) != len(self.labels):
            raise ValueError(
                f"texts and labels must be the same length: "
                f"{len(self.texts)} vs {len(self.labels)}"
            )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.texts)

    # ── Per-mode item builders ────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        if self.mode == "bilstm":
            return self._item_bilstm(text, label)
        return self._item_phobert(text, label)

    def _item_bilstm(self, text: str, label: int) -> Dict[str, torch.Tensor]:
        assert isinstance(self.tokenizer, Vocab), (
            "mode='bilstm' requires a Vocab tokenizer"
        )
        ids = self.tokenizer.text_to_ids(text, max_len=self.max_len)
        if not ids:                       # avoid zero-length sequences
            ids = [UNK_ID]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length":    torch.tensor(len(ids), dtype=torch.long),
            "label":     torch.tensor(label, dtype=torch.long),
        }

    def _item_phobert(self, text: str, label: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Length-based filtering wrapper (training only)
# ──────────────────────────────────────────────────────────────

class FilteredViHSDDataset(Dataset):
    """Drop training samples whose tokenised length is below ``min_length``.

    Why this exists: Week-4 E1 analysis showed ~11% of training samples
    have length≤2 after underthesea word-segmentation. A 1-token sample
    through a 2-layer BiLSTM produces a near-degenerate hidden state and
    can drive ``CrossEntropyLoss`` to NaN under fp16 / aggressive lr.

    **Use on training only.** Dev and test must stay at the full
    distribution, otherwise reported metrics overstate generalisation.

    Parameters
    ----------
    base
        A ``ViHSDDataset`` instance in ``mode='bilstm'``. The PhoBERT
        mode produces fixed-max_len padded tensors, so length filtering
        is meaningless there.
    min_length
        Minimum token count (post-truncation, includes ``<bos>``/``<eos>``)
        required to keep a sample. Default 3, matching the E1 verdict.
    verbose
        Print the filter summary when True.
    save_path
        Optional path for a small JSON dump of the filter stats. Useful
        for the Week-4 decision log.

    Attributes
    ----------
    base
        The wrapped ``ViHSDDataset``.
    indices
        Sorted list of base-dataset indices that survived the filter.
    stats
        Dict with keys ``original_size``, ``filtered_size``, ``removed``,
        ``pct_removed``, ``min_length``.
    """

    def __init__(
        self,
        base: "ViHSDDataset",
        min_length: int = 3,
        verbose: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        length_tokenizer: Optional["Vocab"] = None,
    ) -> None:
        # Pick the tokenizer used to compute *length* for the filter. For
        # ``mode='bilstm'`` we already have a Vocab on the base dataset; for
        # ``mode='phobert'`` we need an explicit Vocab so the filter uses the
        # same word-level length metric as the BiLSTM pipeline — keeping the
        # train-set composition identical across the two models.
        if base.mode == "bilstm":
            if not isinstance(base.tokenizer, Vocab):
                raise TypeError(
                    "BiLSTM mode requires a Vocab tokenizer on the base dataset."
                )
            length_tok = base.tokenizer
        elif base.mode == "phobert":
            if length_tokenizer is None:
                raise ValueError(
                    "FilteredViHSDDataset on mode='phobert' needs an explicit "
                    "`length_tokenizer=vocab` so the length metric matches the "
                    "BiLSTM filter (keeps train composition identical)."
                )
            if not isinstance(length_tokenizer, Vocab):
                raise TypeError("length_tokenizer must be a Vocab instance.")
            length_tok = length_tokenizer
        else:
            raise ValueError(f"Unsupported base.mode: {base.mode!r}")

        self.base = base
        self.min_length = int(min_length)

        kept: List[int] = []
        for i in range(len(base)):
            ids = length_tok.text_to_ids(base.texts[i], max_len=base.max_len)
            if not ids:
                ids = [UNK_ID]                                  # mirrors _item_bilstm
            if len(ids) >= self.min_length:
                kept.append(i)
        self.indices = kept

        original = len(base)
        kept_n = len(kept)
        self.stats: Dict[str, float] = {
            "original_size": original,
            "filtered_size": kept_n,
            "removed":       original - kept_n,
            "pct_removed":   100.0 * (original - kept_n) / max(original, 1),
            "min_length":    self.min_length,
            "base_mode":     base.mode,
        }

        if verbose:
            print(f"FilteredViHSDDataset(min_length={self.min_length}, mode={base.mode}):")
            print(f"  original size: {original:,}")
            print(f"  filtered size: {kept_n:,}")
            print(f"  removed      : {original - kept_n:,}  ({self.stats['pct_removed']:.2f}%)")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
            if verbose:
                print(f"  stats saved  : {save_path}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base[self.indices[idx]]


# ──────────────────────────────────────────────────────────────
# Collate functions
# ──────────────────────────────────────────────────────────────

def collate_fn_bilstm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Right-pad ``input_ids`` to the longest sequence in the batch.

    Drops length-0 samples silently to defend against an upstream bug —
    ``pack_padded_sequence`` corrupts state on a 0-length input, and the
    downstream error is unrecognisable. A real Dataset should never yield
    length=0 (``_item_bilstm`` substitutes ``[UNK]``), but the guard is
    cheap and the failure mode would be too painful otherwise.
    """
    batch = [b for b in batch if int(b["length"].item()) >= 1]
    if not batch:
        raise RuntimeError("collate_fn_bilstm: every sample in the batch had length<1")

    lengths = torch.stack([b["length"] for b in batch])
    max_len = int(lengths.max().item())
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        n = int(b["length"].item())
        input_ids[i, :n] = b["input_ids"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}


def collate_fn_phobert(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack pre-padded PhoBERT tensors into a batch."""
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["label"] for b in batch]),
    }


# ──────────────────────────────────────────────────────────────
# Embedding matrix
# ──────────────────────────────────────────────────────────────

def _open_embedding_file(path: Path):
    """Open ``.vec`` or ``.txt`` (optionally gzipped) for line-wise reading."""
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def build_embedding_matrix(
    vocab: Vocab,
    embedding_path: Union[str, Path],
    dim: int = 300,
    unk_init_range: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
    compound_fallback: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Align a pretrained embedding file with the project ``Vocab``.

    The embedding file is expected in fastText / PhoW2V text format::

        <vocab_size> <dim>
        word v1 v2 … vd
        …

    The header line is *optional* — if the first line has the wrong shape
    it is parsed as a word vector instead. ``dim`` is sanity-checked
    against the file; on mismatch the file's dim wins and a warning is
    printed.

    Compound fallback
    -----------------
    The cleaned text from :mod:`src.preprocess` joins multi-syllable words
    with ``_`` (underthesea ``format='text'`` convention), e.g. ``hôm_nay``.
    Pre-trained embeddings trained on **raw** Vietnamese text (fastText
    cc.vi, anything off Common Crawl) won't have such tokens — coverage
    would collapse from ~70% (PhoW2V, same tokenisation) to ~30%.

    When ``compound_fallback=True`` we make a single extra pass: for any
    compound vocab token not found directly, we average the embeddings of
    its underscore-separated parts. This typically recovers another
    25-40% of coverage on fastText.

    Returns
    -------
    embeddings
        ``torch.FloatTensor`` of shape ``(len(vocab), file_dim)``. The
        ``<pad>`` row is forced to zero. Other rows missing from the file
        are initialised uniformly in ``[-unk_init_range, +unk_init_range]``.
    stats
        ``{"vocab_size", "found_direct", "found_compound", "found",
        "missing", "coverage", "dim"}``.
    """
    embedding_path = Path(embedding_path)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    rng = np.random.RandomState(seed)
    # Provisional matrix — we may resize once we know the file dim.
    matrix = rng.uniform(-unk_init_range, unk_init_range, size=(len(vocab), dim)).astype(np.float32)
    found_mask = np.zeros(len(vocab), dtype=bool)

    # Pre-compute compound-word parts we'll want to capture during the scan.
    compound_parts: Dict[str, List[str]] = {}
    parts_needed: set = set()
    if compound_fallback:
        for tok, idx in vocab.stoi.items():
            if idx < len(Vocab.SPECIALS) or "_" not in tok:
                continue
            parts = tok.split("_")
            if len(parts) >= 2 and all(parts):
                compound_parts[tok] = parts
                parts_needed.update(parts)

    parts_vectors: Dict[str, np.ndarray] = {}

    t0 = time.perf_counter()
    file_dim: Optional[int] = None

    with _open_embedding_file(embedding_path) as f:
        first_line = f.readline().rstrip("\n")
        parts = first_line.split(" ")
        # Heuristic: a header looks like "<int> <int>", a vector line is "<word> <float> …"
        is_header = len(parts) == 2 and all(p.isdigit() for p in parts)
        if is_header:
            file_dim = int(parts[1])
            line_iter = f
        else:
            file_dim = len(parts) - 1
            line_iter = _prepend_line(first_line, f)

        if file_dim != dim:
            if verbose:
                print(
                    f"⚠ Requested dim={dim} but file is {file_dim}-dim; "
                    f"using file dim."
                )
            matrix = rng.uniform(-unk_init_range, unk_init_range,
                                 size=(len(vocab), file_dim)).astype(np.float32)

        for raw in line_iter:
            line = raw.rstrip("\n")
            if not line:
                continue
            sp = line.find(" ")
            if sp == -1:
                continue
            word = line[:sp]
            in_vocab = word in vocab.stoi
            is_part = word in parts_needed
            if not in_vocab and not is_part:
                continue
            vec = np.fromstring(line[sp + 1:], sep=" ", dtype=np.float32)
            if vec.shape[0] != file_dim:
                continue
            if in_vocab:
                idx = vocab.stoi[word]
                matrix[idx] = vec
                found_mask[idx] = True
            if is_part:
                parts_vectors[word] = vec

    found_direct = int(found_mask.sum())

    # Compound fallback: average parts for any unfound compound token.
    recovered = 0
    if compound_fallback:
        for tok, parts in compound_parts.items():
            idx = vocab.stoi[tok]
            if found_mask[idx]:
                continue
            available = [parts_vectors[p] for p in parts if p in parts_vectors]
            if not available:
                continue
            matrix[idx] = np.mean(available, axis=0)
            found_mask[idx] = True
            recovered += 1

    # <pad> stays zero so it can never contribute to the mean / sum pooling
    matrix[PAD_ID] = 0.0
    found_mask[PAD_ID] = True   # treat as "covered" for stats

    found = int(found_mask.sum())
    coverage = found / len(vocab)
    stats = {
        "vocab_size":     len(vocab),
        "found_direct":   found_direct,
        "found_compound": recovered,
        "found":          found,
        "missing":        len(vocab) - found,
        "coverage":       coverage,
        "dim":            file_dim,
    }
    if verbose:
        dt = time.perf_counter() - t0
        compound_msg = f" (+{recovered:,} via compound fallback)" if recovered else ""
        print(
            f"  loaded {embedding_path.name} in {dt:.1f}s  |  "
            f"coverage {found:,}/{len(vocab):,} ({coverage:.1%}){compound_msg}  |  dim={file_dim}"
        )

    return torch.from_numpy(matrix), stats


def _prepend_line(line: str, f):
    """Yield ``line`` first, then everything else from ``f``."""
    yield line
    for nxt in f:
        yield nxt


# ──────────────────────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────────────────────

def get_class_weights(y_train: Iterable[int]) -> torch.Tensor:
    """Inverse-frequency class weights (``sklearn`` 'balanced' scheme)."""
    from sklearn.utils.class_weight import compute_class_weight
    y = np.asarray(list(y_train))
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":   # pragma: no cover
    sample_texts = [
        "hôm nay trời đẹp quá",
        "thằng đó nói chuyện ngu vl",
        "bài viết hay quá cảm ơn tác giả",
        "đm cái lũ kia",
    ]
    sample_labels = [0, 1, 0, 2]

    vocab = Vocab.build_from_texts(sample_texts, min_freq=1, max_size=100)
    print(f"vocab size: {len(vocab)}  (first 10 itos: {vocab.itos[:10]})")

    ds = ViHSDDataset(sample_texts, sample_labels, tokenizer=vocab, max_len=16, mode="bilstm")
    print(f"len(ds) = {len(ds)}; item 0 = {ds[0]}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn_bilstm)
    batch = next(iter(loader))
    print({k: tuple(v.shape) for k, v in batch.items()})
    print("class weights:", get_class_weights(sample_labels))
