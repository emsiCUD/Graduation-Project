"""
predict.py — End-to-end inference pipeline for the ViHSD toxic-comment champion.

Pipeline: raw text → VietnameseTextCleaner → TF-IDF word ⊕ TF-IDF char
       → LogisticRegression → {label, label_id, confidence, probabilities}

Usage
-----
    from src.predict import ToxicCommentPredictor

    clf = ToxicCommentPredictor()
    clf.predict("đm thằng này ngu vl")
    # → {'label': 'OFFENSIVE', 'label_id': 1, 'confidence': 0.82,
    #    'probabilities': {'CLEAN': 0.08, 'OFFENSIVE': 0.82, 'HATE': 0.10}}

    clf.predict_batch(["xin chào", "bài viết hay quá"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
from scipy.sparse import hstack

from configs.config import LABEL_MAP, ROOT_DIR
from src.preprocess import VietnameseTextCleaner


_DEFAULT_MODEL_PATH = Path(ROOT_DIR) / "models" / "baselines" / "LR_champion.pkl"
_DEFAULT_WORD_VEC   = Path(ROOT_DIR) / "models" / "vectorizers" / "tfidf_unibi.pkl"
_DEFAULT_CHAR_VEC   = Path(ROOT_DIR) / "models" / "vectorizers" / "tfidf_char.pkl"


class ToxicCommentPredictor:
    """End-to-end predictor: text → label, confidence, per-class probabilities."""

    def __init__(
        self,
        model_path: Union[str, Path] = _DEFAULT_MODEL_PATH,
        word_vectorizer_path: Union[str, Path] = _DEFAULT_WORD_VEC,
        char_vectorizer_path: Union[str, Path] = _DEFAULT_CHAR_VEC,
    ) -> None:
        self.model        = joblib.load(model_path)
        self.word_vec     = joblib.load(word_vectorizer_path)
        self.char_vec     = joblib.load(char_vectorizer_path)
        self.cleaner      = VietnameseTextCleaner()
        self._class_names = [LABEL_MAP[int(c)] for c in self.model.classes_]

    def _vectorize(self, cleaned_texts: List[str]):
        Xw = self.word_vec.transform(cleaned_texts)
        Xc = self.char_vec.transform(cleaned_texts)
        return hstack([Xw, Xc]).tocsr()

    def _format(self, proba_row: np.ndarray) -> Dict:
        pred_idx = int(np.argmax(proba_row))
        label_id = int(self.model.classes_[pred_idx])
        return {
            "label":         LABEL_MAP[label_id],
            "label_id":      label_id,
            "confidence":    float(proba_row[pred_idx]),
            "probabilities": {
                self._class_names[i]: float(proba_row[i])
                for i in range(len(self._class_names))
            },
        }

    def predict(self, text: str) -> Dict:
        cleaned = self.cleaner.clean(text)
        X = self._vectorize([cleaned])
        proba = self.model.predict_proba(X)[0]
        return self._format(proba)

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        cleaned = [self.cleaner.clean(t) for t in texts]
        X = self._vectorize(cleaned)
        probas = self.model.predict_proba(X)
        return [self._format(row) for row in probas]


if __name__ == "__main__":
    samples = [
        "xin chào mọi người",
        "đm thằng này ngu vl",
        "bài viết hay quá",
        "bọn nó toàn đồ rác rưởi",
        "ko hiểu sao thằng đó cứ nói linh tinh",
    ]
    clf = ToxicCommentPredictor()
    for s in samples:
        out = clf.predict(s)
        print(f"{s:50s} → {out['label']:10s} ({out['confidence']:.3f})")