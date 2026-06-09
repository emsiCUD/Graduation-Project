"""
predictor.py — Standalone PhoBERT predictor for the Streamlit demo.

Independent of `src/predict.py` (which is the Week-3 LR predictor used for
comparison). This module wraps the fine-tuned PhoBERT-base-v2 checkpoint
saved under ``models/dl/phobert_best/`` and the canonical Vietnamese text
cleaner from ``src/preprocess.py``.

Usage
-----
    from app.predictor import PhoBERTPredictor
    p = PhoBERTPredictor()                       # loads model once
    out = p.predict("Câu cần phân tích")
    # → {'label_id': 0, 'label_name': 'CLEAN', 'confidence': 0.97,
    #    'probabilities': {'CLEAN': 0.97, 'OFFENSIVE': 0.02, 'HATE': 0.01},
    #    'cleaned_text': 'câu cần phân_tích',
    #    'inference_ms': 18.3,
    #    'truncated': False}

Designed to be import-safe (no top-level torch ops, no GPU allocation at
module load) so the Streamlit cache controls when the model actually
materialises.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("phobert_demo")

# Add project root to sys.path so `from src.preprocess import ...` works
# whether the demo is launched from the repo root or from the app/ folder.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocess import VietnameseTextCleaner
from configs.config import PHOBERT_CONFIG


# 3-class label map matches Weeks 3-5 (sklearn-friendly integer labels).
LABEL_MAP: Dict[int, str] = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
DEFAULT_MODEL_DIR = _ROOT / "models" / "dl" / "phobert_best"
# Read from the shared config so the demo tokenises EXACTLY like training/eval
# did — PhoBERT-base-v2 was fine-tuned and tested at max_len=256 (the source of
# the reported test macro-F1 = 0.6618). Hard-coding a smaller value would
# silently truncate long comments and diverge from the thesis numbers.
DEFAULT_MAX_LEN   = int(PHOBERT_CONFIG.get("max_len", 256))


class PhoBERTPredictor:
    """Single-process PhoBERT inference wrapper.

    Parameters
    ----------
    model_dir
        Path to the saved HuggingFace model directory (contains
        ``config.json``, ``model.safetensors`` / ``pytorch_model.bin``,
        and the tokenizer files). Defaults to ``models/dl/phobert_best/``.
    device
        Explicit device ("cuda", "cpu"). When ``None`` (default), uses
        CUDA if available else CPU.
    max_len
        PhoBERT max sub-word tokens. Matches the training value (256).
    use_fp16
        Run forward in fp16 autocast on CUDA (≈2× faster, no quality
        loss for inference). Silently disabled on CPU.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
        max_len: int = DEFAULT_MAX_LEN,
        use_fp16: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"PhoBERT model directory not found: {self.model_dir}. "
                f"Train and save the model via notebooks/04c_phobert.ipynb first."
            )

        # Device selection.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        logger.info("Loading PhoBERT from %s on %s (fp16=%s)…",
                    self.model_dir, self.device, self.use_fp16)
        _t0 = time.perf_counter()

        # Tokenizer — PhoBERT-v2 has no fast tokenizer, must use slow.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), use_fast=False
        )
        _model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        try:
            self.model = _model.to(self.device).eval()
        except (RuntimeError, OSError) as e:
            # CUDA OOM / driver / context-contention errors would otherwise
            # crash the whole Streamlit server. Fall back to CPU so the demo
            # keeps working (135M params runs fine on CPU, ~1-2s/comment).
            if self.device.type == "cuda":
                logger.warning("CUDA load failed (%s) — falling back to CPU.", e)
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.use_fp16 = False
                self.model = _model.to(self.device).eval()
            else:
                raise

        # Vietnamese cleaner — underthesea word segmentation enabled.
        # Same canonical pipeline used during training.
        self.cleaner = VietnameseTextCleaner(use_tokenizer=True, lowercase=True)

        self.max_len = int(max_len)
        logger.info("PhoBERT ready in %.1fs (max_len=%d).",
                    time.perf_counter() - _t0, self.max_len)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """Predict a single comment.

        Returns a dict with everything the UI needs:
        ``label_id``, ``label_name``, ``confidence``, ``probabilities``
        (per-class dict), ``cleaned_text``, ``inference_ms``, ``truncated``.

        An empty / whitespace-only input is rejected with a ``ValueError``.
        """
        if text is None or not str(text).strip():
            raise ValueError("Input text is empty.")

        t0 = time.perf_counter()

        # 1. Preprocess (clean + underthesea segmentation).
        cleaned = self.cleaner.clean(str(text))
        if not cleaned.strip():
            # Cleaner stripped everything (URL-only, emoji-only, etc.).
            raise ValueError("Input collapses to empty after cleaning.")

        # 2. Tokenise for PhoBERT. Detect truncation BEFORE we cap.
        # `encode` returns a list; comparing with max_len tells us
        # whether the model will actually see a truncated input.
        raw_ids = self.tokenizer.encode(cleaned, add_special_tokens=True)
        truncated = len(raw_ids) > self.max_len

        enc = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)

        # 3. Forward (autocast on GPU, plain on CPU).
        if self.use_fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = self.model(**enc).logits.float()
        else:
            logits = self.model(**enc).logits

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_id = int(probs.argmax())
        pred_name = LABEL_MAP[pred_id]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "label_id":      pred_id,
            "label_name":    pred_name,
            "confidence":    float(probs[pred_id]),
            "probabilities": {LABEL_MAP[i]: float(probs[i]) for i in (0, 1, 2)},
            "cleaned_text":  cleaned,
            "inference_ms":  float(elapsed_ms),
            "truncated":     bool(truncated),
            "n_subword_tokens": int(min(len(raw_ids), self.max_len)),
        }

    # ------------------------------------------------------------------
    def info(self) -> Dict:
        """Lightweight metadata for the UI footer."""
        return {
            "model_dir":   str(self.model_dir),
            "device":      str(self.device),
            "fp16":        self.use_fp16,
            "max_len":     self.max_len,
            "num_labels":  int(self.model.config.num_labels),
            "param_count": sum(p.numel() for p in self.model.parameters()),
        }


if __name__ == "__main__":
    # Tiny CLI smoke test:  python -m app.predictor
    # Windows consoles default to cp1252 and choke on Vietnamese output;
    # force UTF-8 so the smoke test prints cleanly on every platform.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    p = PhoBERTPredictor()
    print(p.info())
    for s in [
        "Hôm nay trời rất đẹp và tôi cảm thấy vui.",
        "Đậu má thầy ok chưa ?",
        "Phạm Văn Lộc vãi lol",
    ]:
        out = p.predict(s)
        print(f"\n{s!r}")
        print(f"  → {out['label_name']}  conf={out['confidence']:.3f}  "
              f"({out['inference_ms']:.1f} ms)")
        print(f"  probs: {out['probabilities']}")
