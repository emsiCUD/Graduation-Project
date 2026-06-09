# Vietnamese Toxic Comment Detection (ViHSD)

Graduation thesis project: a 3-class **Vietnamese hate-speech / toxic-comment
detector** trained and evaluated on the **ViHSD** dataset (Luu et al., 2021).
The project benchmarks a traditional-ML baseline, a deep-learning baseline, and
a fine-tuned transformer, then performs a quantitative + qualitative error
analysis and ships an interactive demo.

Labels: **CLEAN (0)**, **OFFENSIVE (1)**, **HATE (2)**.
Primary metric: **macro-F1** (chosen because of the ~12:1 class imbalance).

---

## Results (test set, n = 6,680)

| Model | macro-F1 | F1 CLEAN | F1 OFFENSIVE | F1 HATE | Accuracy | Params |
|---|---|---|---|---|---|---|
| LR + char n-grams (Week 3) | 0.6183 | 0.9144 | 0.3976 | 0.5428 | 0.8334 | 25K |
| BiLSTM v2 (PhoW2V) | 0.5902 | 0.8981 | 0.3694 | 0.5030 | 0.8048 | 3.5M |
| **PhoBERT-base-v2 (champion)** | **0.6618** | **0.9312** | **0.4374** | **0.6166** | **0.8624** | 135M |

PhoBERT is the deployment champion. The OFFENSIVE class is the hard ceiling for
every architecture — ~33% of OFFENSIVE samples cross the CLEAN boundary even for
PhoBERT, which the error analysis attributes mainly to **annotation ambiguity**
rather than model capacity. See `report/` for the full write-up.

---

## Dataset

ViHSD official splits are preserved for comparability:

| Split | Size | CLEAN | OFFENSIVE | HATE |
|---|---|---|---|---|
| Train | 24,048 | 82.7% | 6.7% | 10.6% |
| Dev | 2,672 | 82.0% | 7.9% | 10.1% |
| Test | 6,680 | 83.0% | 6.6% | 10.3% |

> Luu, S. T., Nguyen, K. V., & Nguyen, N. L.-T. (2021). *A Large-scale Dataset
> for Hate Speech Detection on Vietnamese Social Media Texts.* IEA/AIE 2021.

---

## Repository structure

```
.
├── notebooks/         # End-to-end pipeline (run in order, see below)
├── src/               # Reusable modules
│   ├── preprocess.py      # VietnameseTextCleaner (+ teencode.json dictionary)
│   ├── features.py        # TF-IDF word + char vectorizers
│   ├── dataset_dl.py      # Vocab, datasets, collate fns, embedding matrix
│   ├── models_dl.py       # BiLSTMClassifier, TextCNN
│   ├── evaluate.py        # metrics, confusion matrices, prediction CSVs
│   ├── predict.py         # ToxicCommentPredictor (LR end-to-end inference)
│   └── utils.py           # set_seed, Timer, get_device
├── configs/config.py  # Centralised paths, label maps, hyperparameters
├── app/               # Interactive demo (PhoBERT)
│   ├── predictor.py       # PhoBERTPredictor inference wrapper
│   ├── streamlit_demo.py  # Streamlit UI
│   └── requirements_demo.txt
├── report/            # Weekly summaries + final error-analysis write-up
├── results/           # Metrics (csv/json), predictions, figures
├── models/            # Saved artefacts (baselines, vectorizers, dl, embeddings)
├── data/              # raw + processed ViHSD splits
└── requirements.txt
```

---

## Setup

```powershell
# From the project root (Windows / PowerShell)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

For the demo only, a lighter dependency set is in `app/requirements_demo.txt`.
On Windows + NVIDIA GPU, install the CUDA build of PyTorch first (see comments
in that file).

---

## Running the pipeline

The notebooks are numbered and meant to be run in order. The test set stays
untouched until the Week-4 final evaluation.

| Notebook | Stage |
|---|---|
| `01_eda` | Exploratory data analysis |
| `02_preprocessing` | Cleaning pipeline + feature matrices |
| `03_baseline_ml` → `03c_final_evaluation` | Traditional-ML baselines, tuning, test eval |
| `04a_setup_verify` → `04d_test_evaluation` | BiLSTM/TextCNN, PhoBERT, 3-way test comparison |
| `05a` → `05c` | Quantitative + qualitative error analysis, synthesis |

---

## Running the demo

```powershell
.venv\Scripts\python -m streamlit run app/streamlit_demo.py
```

Then open the **Local URL** printed in the terminal (e.g. `http://localhost:8501`).
The PhoBERT champion (`models/dl/phobert_best/`) loads once (~10 s) and the app
predicts the label, per-class probabilities, and shows the preprocessed text.

Notes:
- `.streamlit/config.toml` disables the file watcher (avoids a known
  Streamlit + PyTorch issue on model load).
- The predictor auto-falls back to CPU if CUDA is unavailable or fails, so the
  demo runs on any machine (135M params ≈ 1–2 s/comment on CPU).
- Reproducibility: `random_state = 42` throughout; PhoBERT trained on a single
  RTX 3060 (6 GB) with fp16, `max_len = 256`.

---

## Key takeaways (error analysis)

- **Annotation ceiling** dominates: 71% of the failure sample is label-questionable.
- **Implicit / contextual toxicity** (no explicit slur) is the largest *model-improvable* gap.
- Long comments are hardest (signal dilution + over-escalation); short comments are easiest.
- PhoBERT is **over-confident** (ECE ≈ 0.106) — don't use raw softmax confidence
  for auto-moderation; route mid-confidence cases to human review.
