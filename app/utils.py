"""
utils.py — Helper functions for the Streamlit demo app.

Responsibilities:
  - Load trained model artifacts (SVM, PhoBERT, etc.)
  - Preprocess raw input text before inference
  - Run prediction and return (label, confidence) tuple
"""

import os
import pickle
from typing import Tuple

# ── Paths ─────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

TRADITIONAL_MODEL_PATH = os.path.join(MODELS_DIR, "traditional", "svm_tfidf.pkl")
PHOBERT_MODEL_PATH = os.path.join(MODELS_DIR, "phobert")

# ── Model loader ──────────────────────────────────────────────
_model_cache: dict = {}


def load_model(model_choice: str):
    """
    Load and cache the selected model.

    Args:
        model_choice: One of "SVM + TF-IDF" or "PhoBERT"

    Returns:
        Loaded model object (sklearn pipeline or HuggingFace model+tokenizer tuple)
    """
    if model_choice in _model_cache:
        return _model_cache[model_choice]

    if model_choice == "SVM + TF-IDF":
        with open(TRADITIONAL_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    elif model_choice == "PhoBERT":
        # Lazy import to avoid loading transformers when not needed
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_PATH)
        model_obj = AutoModelForSequenceClassification.from_pretrained(PHOBERT_MODEL_PATH)
        model_obj.eval()
        model = (model_obj, tokenizer)

    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    _model_cache[model_choice] = model
    return model


# ── Predictor ─────────────────────────────────────────────────
def predict_comment(model, text: str, model_choice: str) -> Tuple[str, float]:
    """
    Run inference on a single Vietnamese comment.

    Args:
        model:        Loaded model object returned by load_model()
        text:         Raw Vietnamese comment string
        model_choice: "SVM + TF-IDF" or "PhoBERT"

    Returns:
        Tuple of (label, confidence) where label is "toxic" or "non-toxic"
        and confidence is a float in [0, 1].
    """
    if model_choice == "SVM + TF-IDF":
        label_idx = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        confidence = float(max(proba))
        label = "toxic" if label_idx == 1 else "non-toxic"

    elif model_choice == "PhoBERT":
        import torch

        model_obj, tokenizer = model
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            logits = model_obj(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        label_idx = int(torch.argmax(probs))
        confidence = float(probs[label_idx])
        label = "toxic" if label_idx == 1 else "non-toxic"

    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    return label, confidence
