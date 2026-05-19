"""
preprocess.py — Vietnamese text cleaning pipeline for the ViHSD project.

The main entry point is the :class:`VietnameseTextCleaner` class, which
exposes one method per cleaning step plus a :meth:`clean` orchestrator
that runs them in a fixed order. A helper :func:`batch_clean` is provided
for vectorised cleaning over a ``pandas.Series`` with a tqdm progress bar.

Stop-word removal is intentionally NOT performed here — that decision is
deferred to per-experiment ablations in the modelling notebooks.

Run this file directly to see a 5-sentence demo:

    python -m src.preprocess
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

# ── Reproducibility ───────────────────────────────────────────
RANDOM_STATE: int = 42
random.seed(RANDOM_STATE)

# ── Optional dependency: underthesea ──────────────────────────
try:
    from underthesea import word_tokenize as _uts_word_tokenize
    _HAS_UNDERTHESEA = True
except ImportError:                                    # pragma: no cover
    _uts_word_tokenize = None
    _HAS_UNDERTHESEA = False


# ── Module-level regex patterns (compiled once) ───────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# Cover the main emoji blocks plus dingbats / misc symbols.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FFFF"   # Misc Symbols & Pictographs … Symbols & Pictographs Ext-A
    "\U0001F600-\U0001F64F"   # Emoticons
    "\U0001F680-\U0001F6FF"   # Transport & Map
    "\U0001F1E0-\U0001F1FF"   # Regional indicators (flags)
    "☀-➿"           # Misc symbols + dingbats
    "⌀-⏿"           # Misc technical
    "⬀-⯿"           # Misc symbols & arrows
    "️"                  # Variation selector
    "‍"                  # Zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# Vietnamese vowels (with all tone-marked forms). Used to decide max repeats.
_VN_VOWELS = set(
    "aăâeêioôơuưy"
    "àáạảãằắặẳẵầấậẩẫ"
    "èéẹẻẽềếệểễ"
    "ìíịỉĩ"
    "òóọỏõồốộổỗờớợởỡ"
    "ùúụủũừứựửữ"
    "ỳýỵỷỹ"
)
_REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

# Keep: Vietnamese letters (incl. tone marks + đ), ASCII digits, whitespace.
_KEEP_CHARS_RE = re.compile(
    r"[^"
    r"a-zA-Z0-9\s"
    r"àáạảãằắặẳẵầấậẩẫ"
    r"èéẹẻẽềếệểễ"
    r"ìíịỉĩ"
    r"òóọỏõồốộổỗờớợởỡ"
    r"ùúụủũừứựửữ"
    r"ỳýỵỷỹ"
    r"ăâêôơưđ"
    r"ÀÁẠẢÃẰẮẶẲẴẦẤẬẨẪ"
    r"ÈÉẸẺẼỀẾỆỂỄ"
    r"ÌÍỊỈĨ"
    r"ÒÓỌỎÕỒỐỘỔỖỜỚỢỞỠ"
    r"ÙÚỤỦŨỪỨỰỬỮ"
    r"ỲÝỴỶỸ"
    r"ĂÂÊÔƠƯĐ"
    r"]",
    flags=re.UNICODE,
)

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


# ── Default location of the teencode dictionary ───────────────
_DEFAULT_TEENCODE_PATH = Path(__file__).resolve().parent / "teencode.json"


class VietnameseTextCleaner:
    """Modular Vietnamese text cleaner for hate-speech detection.

    Each step is exposed as its own method so it can be unit-tested or
    swapped in ablations. :meth:`clean` runs the canonical pipeline:
    lowercase → remove URLs → remove emojis → collapse repeated chars
    → normalize teencode → strip special chars → normalize whitespace
    → word-tokenize (optional).

    Parameters
    ----------
    teencode_path
        Path to a JSON file mapping ``teencode -> standard form``. Defaults
        to ``src/teencode.json`` shipped with this repo.
    use_tokenizer
        When ``True`` (default), :meth:`clean` finishes by calling
        :meth:`tokenize`. Set to ``False`` if you want plain cleaned text.
    lowercase
        Whether to lowercase before further processing. Defaults to ``True``.
    """

    def __init__(
        self,
        teencode_path: Optional[os.PathLike] = None,
        use_tokenizer: bool = True,
        lowercase: bool = True,
    ) -> None:
        self.use_tokenizer = use_tokenizer
        self.lowercase = lowercase
        self.teencode_map: Dict[str, str] = self._load_teencode(
            Path(teencode_path) if teencode_path else _DEFAULT_TEENCODE_PATH
        )
        # One alternation regex with word boundaries – built once.
        if self.teencode_map:
            keys_sorted = sorted(map(re.escape, self.teencode_map), key=len, reverse=True)
            self._teencode_re = re.compile(
                r"(?<!\w)(" + "|".join(keys_sorted) + r")(?!\w)",
                flags=re.IGNORECASE | re.UNICODE,
            )
        else:
            self._teencode_re = None

    # ── I/O helpers ───────────────────────────────────────────
    @staticmethod
    def _load_teencode(path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ── Cleaning steps ────────────────────────────────────────
    def remove_urls(self, text: str) -> str:
        """Strip ``http(s)://...`` and ``www....`` URLs."""
        return _URL_RE.sub(" ", text)

    def remove_emojis(self, text: str) -> str:
        """Strip emoji characters and common pictographs."""
        return _EMOJI_RE.sub(" ", text)

    def normalize_repeated_chars(self, text: str) -> str:
        """Collapse runs of 3+ identical characters.

        Vietnamese vowels keep up to 2 repetitions (e.g. ``"aaaa" → "aa"``)
        because doubled vowels carry stylistic intensity; consonants are
        reduced to a single occurrence (e.g. ``"ngonnnn" → "ngon"``).
        """
        def _repl(m: "re.Match[str]") -> str:
            ch = m.group(1)
            return ch * 2 if ch.lower() in _VN_VOWELS else ch

        return _REPEAT_RE.sub(_repl, text)

    def normalize_teencode(self, text: str) -> str:
        """Replace teencode tokens with their standard Vietnamese form.

        Matching is case-insensitive and respects word boundaries so that
        e.g. ``"ko"`` only matches the standalone word, not the ``"ko"``
        inside ``"kho"``.
        """
        if not self._teencode_re:
            return text
        return self._teencode_re.sub(
            lambda m: self.teencode_map[m.group(1).lower()],
            text,
        )

    def remove_special_chars(self, text: str) -> str:
        """Keep only Vietnamese letters, ASCII digits, and whitespace."""
        return _KEEP_CHARS_RE.sub(" ", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse any whitespace run into a single space and strip ends."""
        return _WHITESPACE_RE.sub(" ", text).strip()

    def tokenize(self, text: str) -> str:
        """Word-segment Vietnamese text using ``underthesea``.

        Returns a string with multi-syllable words joined by ``_`` (the
        ``format='text'`` convention). Raises ``ImportError`` if
        ``underthesea`` is not installed.
        """
        if not _HAS_UNDERTHESEA:
            raise ImportError(
                "underthesea is required for tokenize(). "
                "Install with: pip install underthesea"
            )
        return _uts_word_tokenize(text, format="text")

    # ── Orchestrator ─────────────────────────────────────────
    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on a single string.

        Order:
            1. handle NaN / non-string
            2. lowercase (optional)
            3. remove URLs
            4. remove emojis
            5. collapse repeated characters
            6. normalize teencode
            7. strip special characters
            8. normalize whitespace
            9. word-tokenize (if ``use_tokenizer`` and underthesea is available)
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text)
        if self.lowercase:
            text = text.lower()
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.normalize_repeated_chars(text)
        text = self.normalize_teencode(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        if self.use_tokenizer and _HAS_UNDERTHESEA and text:
            text = self.tokenize(text)
        return text


# ── Vectorised helper ─────────────────────────────────────────
def batch_clean(
    texts: pd.Series,
    cleaner: Optional[VietnameseTextCleaner] = None,
    desc: str = "Cleaning",
) -> pd.Series:
    """Clean a ``pandas.Series`` of strings with a tqdm progress bar.

    Parameters
    ----------
    texts
        Input series of raw comments. NaN values are mapped to ``""``.
    cleaner
        An optional pre-built :class:`VietnameseTextCleaner`. If ``None``,
        a default instance is constructed.
    desc
        Description shown next to the progress bar.

    Returns
    -------
    pandas.Series
        Cleaned strings, indexed identically to ``texts``.
    """
    if cleaner is None:
        cleaner = VietnameseTextCleaner()

    tqdm.pandas(desc=desc)
    return texts.progress_apply(cleaner.clean)


# ── Demo ──────────────────────────────────────────────────────
if __name__ == "__main__":
    samples: List[str] = [
        "Hayyy quáaaa ngonnnnn 😍😍 truy cập https://example.com đi mn!!!",
        "T ko bik j luôn :))) vl thật sự, dc thì cmt nhé",
        "Bọn mắt híp lò xo thụt 😡 cái lzzzz đó ntn nhỉ ??? @user1 #toxic",
        "Đm cái thằng này dz quá, nch là vcl ròi 🤬🤬🤬",
        "Hôm nay đi học ko nhỉ? Hum qa mình mệt quá huhu 😭",
    ]

    print(f"underthesea available: {_HAS_UNDERTHESEA}\n")
    cleaner = VietnameseTextCleaner()
    print(f"Teencode entries loaded: {len(cleaner.teencode_map)}\n")

    for i, s in enumerate(samples, 1):
        cleaned = cleaner.clean(s)
        print(f"[{i}] RAW    : {s}")
        print(f"    CLEAN  : {cleaned}")
        print()