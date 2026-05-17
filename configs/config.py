"""
config.py — Central configuration for the Vietnamese Toxic Comment Detection project.

All paths, hyperparameters, and constants are defined here so they can be
imported consistently across notebooks and src/ modules.

Usage:
    from configs.config import PATHS, TRAIN_CONFIG, PHOBERT_CONFIG
"""

import os

# ── Project root (one level above this file) ──────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── File & folder paths ───────────────────────────────────────
PATHS = {
    # Raw data
    "raw_train": os.path.join(ROOT_DIR, "data", "raw", "train.csv"),
    "raw_dev": os.path.join(ROOT_DIR, "data", "raw", "dev.csv"),
    "raw_test": os.path.join(ROOT_DIR, "data", "raw", "test.csv"),
    "vicsd_train": os.path.join(ROOT_DIR, "data", "raw", "ViCTSD_train.csv"),
    "vicsd_valid": os.path.join(ROOT_DIR, "data", "raw", "ViCTSD_valid.csv"),
    "vicsd_test": os.path.join(ROOT_DIR, "data", "raw", "ViCTSD_test.csv"),
    "custom_data": os.path.join(ROOT_DIR, "data", "raw", "custom_data.csv"),

    # Processed data
    "processed_dir": os.path.join(ROOT_DIR, "data", "processed"),

    # Models
    "model_traditional": os.path.join(ROOT_DIR, "models", "traditional"),
    "model_deep_learning": os.path.join(ROOT_DIR, "models", "deep_learning"),
    "model_phobert": os.path.join(ROOT_DIR, "models", "phobert"),

    # Results
    "results_dir": os.path.join(ROOT_DIR, "results"),
    "figures_dir": os.path.join(ROOT_DIR, "results", "figures"),
    "metrics_csv": os.path.join(ROOT_DIR, "results", "metrics_summary.csv"),
    "error_analysis": os.path.join(ROOT_DIR, "results", "error_analysis.xlsx"),
}


# ── Column names ──────────────────────────────────────────────
COLUMNS = {
    "text": "free_text",    # Raw text column in ViHSD
    "label": "label_id",    # Target column (0=CLEAN, 1=OFFENSIVE, 2=HATE)
}


# ── Label mapping (ViHSD - Luu et al., 2021) ──────────────────
LABEL_MAP = {
    0: "CLEAN",
    1: "OFFENSIVE",
    2: "HATE",
}
LABEL_COLORS = {
    0: "#4CAF50",   # green
    1: "#FF9800",   # orange
    2: "#F44336",   # red
}


# ── Traditional ML training config ───────────────────────────
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,

    # TF-IDF
    "tfidf_max_features": 50_000,
    "tfidf_ngram_range": (1, 2),

    # SVM
    "svm_C": 1.0,
    "svm_kernel": "linear",

    # Logistic Regression
    "lr_C": 1.0,
    "lr_max_iter": 1000,
}


# ── Deep learning config ──────────────────────────────────────
DEEP_LEARNING_CONFIG = {
    "max_len": 256,
    "embedding_dim": 300,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "dropout": 0.3,
    "random_state": 42,
}


# ── PhoBERT config ────────────────────────────────────────────
PHOBERT_CONFIG = {
    "pretrained_model": "vinai/phobert-base-v2",
    "max_len": 256,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "random_state": 42,
}