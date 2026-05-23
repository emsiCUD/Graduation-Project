"""
dataset_dl.py — PyTorch ``Dataset`` plumbing for Week-4 deep-learning models.

Provides everything the BiLSTM and PhoBERT training scripts will need:

* :class:`Vocab`             – word-level vocabulary built from training texts.
* :class:`ViHSDDataset`      – ``torch.utils.data.Dataset`` with two modes
  (``'bilstm'`` or ``'phobert'``).
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
# Collate functions
# ──────────────────────────────────────────────────────────────

def collate_fn_bilstm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Right-pad ``input_ids`` to the longest sequence in the batch."""
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

    Returns
    -------
    embeddings
        ``torch.FloatTensor`` of shape ``(len(vocab), file_dim)``. The
        ``<pad>`` row is forced to zero. Other rows missing from the file
        are initialised uniformly in ``[-unk_init_range, +unk_init_range]``.
    stats
        ``{"vocab_size": int, "found": int, "missing": int, "coverage": float, "dim": int}``.
    """
    embedding_path = Path(embedding_path)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    rng = np.random.RandomState(seed)
    # Provisional matrix — we may resize once we know the file dim.
    matrix = rng.uniform(-unk_init_range, unk_init_range, size=(len(vocab), dim)).astype(np.float32)
    found_mask = np.zeros(len(vocab), dtype=bool)

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
            idx = vocab.stoi.get(word)
            if idx is None:
                continue
            vec = np.fromstring(line[sp + 1:], sep=" ", dtype=np.float32)
            if vec.shape[0] != file_dim:
                continue
            matrix[idx] = vec
            found_mask[idx] = True

    # <pad> stays zero so it can never contribute to the mean / sum pooling
    matrix[PAD_ID] = 0.0
    found_mask[PAD_ID] = True   # treat as "covered" for stats

    found = int(found_mask.sum())
    coverage = found / len(vocab)
    stats = {
        "vocab_size": len(vocab),
        "found":      found,
        "missing":    len(vocab) - found,
        "coverage":   coverage,
        "dim":        file_dim,
    }
    if verbose:
        dt = time.perf_counter() - t0
        print(
            f"  loaded {embedding_path.name} in {dt:.1f}s  |  "
            f"coverage {found:,}/{len(vocab):,} ({coverage:.1%})  |  dim={file_dim}"
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
