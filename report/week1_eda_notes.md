# Week 1 — EDA Notes (ViHSD)

> Generated alongside `notebooks/01_eda.ipynb`. Figures live in `results/figures/`.

## 1. Dataset summary

- **Source:** ViHSD — *A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts* (Luu et al., 2021).
- **Splits provided (official):**

  | split | rows |
  |-------|-----:|
  | train | 24,048 |
  | dev   |  2,672 |
  | test  |  6,680 |
  | **total** | **33,400** |

- **Schema:** 2 columns — `free_text` (string), `label_id` (int 0/1/2).
- **Label definitions:**
  - `0 = CLEAN` — bình thường, không công kích.
  - `1 = OFFENSIVE` — chửi bới, lăng mạ cá nhân.
  - `2 = HATE` — thù ghét nhắm vào nhóm (giới tính, vùng miền, sắc tộc, tôn giáo...).

We reuse the official splits — **do not re-split** — to stay comparable with prior published numbers.

## 2. Class imbalance

Train distribution:

| label | name | % |
|------:|------|--:|
| 0 | CLEAN     | 82.69% |
| 1 | OFFENSIVE |  6.68% |
| 2 | HATE      | 10.63% |

- **Imbalance ratio (majority / minority) = 12.38 : 1.**
- Dev and test follow nearly the same distribution (good — no covariate shift on label prior).
- **Implication:** accuracy is misleading. Primary metric must be **macro-F1**, with per-class F1 reported. Mitigation strategies to try next: class weights, focal loss, oversampling minority classes (SMOTE on TF-IDF, or weighted sampler for PhoBERT).

See `results/figures/label_distribution.png`.

## 3. Comment length

Train (free_text):

| metric | words | chars |
|---|---:|---:|
| mean   |  11.5 |   49 |
| median |   8   |   ~33 |
| p95    |  32   |   ~155 |
| max    | 1,701 | 20,816 |

- 95% of comments are **≤ 32 words**.
- Tiny number of pathological outliers (single sample > 20k chars in train).
- **Decision for Week 2:** truncate/drop very long samples; cap input length at:
  - TF-IDF / DL baselines: **100 tokens**
  - PhoBERT: **128 sub-word tokens** (covers >95% of data)

See `results/figures/length_distribution.png`.

## 4. Data quality

| issue | count in train |
|---|---:|
| missing `free_text`        | 2 |
| empty `free_text` (whitespace only) | 2 |
| duplicated text rows       | 1,490 |

- **Missing/empty:** drop (4 rows total).
- **Duplicates:** decide based on label consistency — if same text + same label, drop; if same text + different label, keep both and flag (annotator disagreement).

## 5. Noise patterns

Observed in raw train comments:

- **Emoji** — common (heart, laugh, angry...). Need either remove or convert to text token.
- **Teencode / abbreviations** — `k, ko, đc, dc, vl, vcl, j, z, mn, ntn, ...`
- **Character repetition** — `quááá`, `hayyy`, `huhuhu`. Normalise `(.)\1{2,}` → `\1\1`.
- **URLs, @mentions, #hashtags** — present, low-signal for hate detection → replace with placeholder tokens.
- **Case + diacritics** — comments mix correct/incorrect tone marks.

See top words and wordclouds in `results/figures/top_words_per_label.png` and `results/figures/wordcloud_per_label.png`. Hate / offensive classes show heavily skewed vocabulary toward profanity and group-targeted slurs — confirms annotation is consistent with definitions.

## 6. Decisions feeding into Week 2

1. **Drop** rows with missing or whitespace-only text.
2. **Deduplicate** by `(free_text, label_id)` pair.
3. Build `src/preprocess.py` with the following pipeline:
   - lowercasing
   - Unicode NFC normalisation + tone-mark canonicalisation
   - replace URL / mention / hashtag with `<url>` / `<user>` / `<tag>`
   - collapse repeated characters (`(.)\1{2,}` → `\1\1`)
   - emoji → strip (baseline) or map to sentiment tag (ablation)
   - teencode dictionary normalisation (`k`→`không`, `đc`→`được`, ...)
   - word-level tokenisation with **underthesea** for classical / DL models
   - PhoBERT models use the model's own BPE tokenizer — only the cleaning steps above are applied beforehand.
4. **Metric:** macro-F1 primary, per-class F1 + confusion matrix secondary.
5. **Length caps:** 100 tokens (classical/DL), 128 sub-words (PhoBERT).
6. **Imbalance handling:** start with class-weighted loss; compare against focal loss and oversampling in ablations.

## 7. Open questions / risks

- ViHSD does not release annotator IDs → cannot compute per-annotator agreement ourselves; rely on paper-reported κ.
- Some "HATE" examples are sarcastic and require world knowledge — likely model upper-bound is well below 100% macro-F1.
- Group-targeted slurs overlap heavily between OFFENSIVE and HATE — expect highest confusion between classes 1 and 2.

## 8. Reproducing this analysis

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
```

Figures are regenerated into `results/figures/`.
