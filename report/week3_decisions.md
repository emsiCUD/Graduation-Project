# Week 3 — Decision Log

Generated automatically by `notebooks/03b_hyperparameter_tuning.ipynb`.

## Champion

**LR + char + threshold tuning** with macro-F1 = **0.6392** on dev
(++0.0377 vs LR baseline at 0.6015).

Per-class F1 on dev:

| Class | F1 |
|-------|-----:|
| CLEAN | 0.9208 |
| OFFENSIVE | 0.4365 |
| HATE | 0.5603 |

## Did char n-grams help?

* `LR_tuned_word`  f1_macro = 0.6024,
  f1_offensive = 0.4000, f1_hate = 0.5121.
* `LR_tuned+char`  f1_macro = 0.6392,
  f1_offensive = 0.4365, f1_hate = 0.5603.
* Δ f1_macro = **+0.0368**,
  Δ f1_offensive = **+0.0365**,
  Δ f1_hate = **+0.0481**.

## Did threshold tuning help?

* Best `(off_bias, hate_bias)` for **macro**: (1.0, 1.0)
  → f1_macro = 0.6392, f1_offensive = 0.4365.
* Best `(off_bias, hate_bias)` for **OFFENSIVE**: (1.0, 1.0)
  → f1_macro = 0.6392, f1_offensive = 0.4365.

If the best biases are (1.0, 1.0) the model is already well calibrated and
no probability re-weighting is needed.

## Weakest class & Week-4 plan

* Weakest class in champion: **OFFENSIVE** (F1 = 0.4365).
* Classical ML appears to plateau around macro-F1 ≈ 0.60–0.65; F1 OFFENSIVE
  remains stuck below 0.45.
* Week 4 plan: move to deep learning — BiLSTM with FastText embeddings,
  then PhoBERT fine-tuning. Both can learn context that distinguishes
  OFFENSIVE from HATE without relying on surface lexical cues.
