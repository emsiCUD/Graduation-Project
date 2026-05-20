# Week 3 — Decision Log

Generated automatically by `notebooks/03b_hyperparameter_tuning.ipynb`.

## Champion

**SVM tuned** with macro-F1 = **0.6341** on dev
(++0.0370 vs LR baseline at 0.5971).

Per-class F1 on dev:

| Class | F1 |
|-------|-----:|
| CLEAN | 0.9259 |
| OFFENSIVE | 0.4169 |
| HATE | 0.5596 |

## Did char n-grams help?

* `LR_tuned_word`  f1_macro = 0.5971,
  f1_offensive = 0.3955, f1_hate = 0.5086.
* `LR_tuned+char`  f1_macro = 0.6233,
  f1_offensive = 0.4083, f1_hate = 0.5536.
* Δ f1_macro = **+0.0262**,
  Δ f1_offensive = **+0.0129**,
  Δ f1_hate = **+0.0450**.

## Did threshold tuning help?

* Best `(off_bias, hate_bias)` for **macro**: (1.0, 1.0)
  → f1_macro = 0.6233, f1_offensive = 0.4083.
* Best `(off_bias, hate_bias)` for **OFFENSIVE**: (1.5, 1.0)
  → f1_macro = 0.6216, f1_offensive = 0.4172.

If the best biases are (1.0, 1.0) the model is already well calibrated and
no probability re-weighting is needed.

## Weakest class & Week-4 plan

* Weakest class in champion: **OFFENSIVE** (F1 = 0.4169).
* Classical ML appears to plateau around macro-F1 ≈ 0.60–0.65; F1 OFFENSIVE
  remains stuck below 0.45.
* Week 4 plan: move to deep learning — BiLSTM with FastText embeddings,
  then PhoBERT fine-tuning. Both can learn context that distinguishes
  OFFENSIVE from HATE without relying on surface lexical cues.
