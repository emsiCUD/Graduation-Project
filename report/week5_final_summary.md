# Week 5 — Error Analysis Summary

_Synthesis of quantitative (`05a`) + qualitative (`05b`) analysis. Manual tagging: n=100._

## Top 5 findings (ranked by importance)

1. **Annotation ceiling** — `annotation_doubt` tagged in 71% of the failure sample, and **83%** of the universal-failure subset D. When LR, BiLSTM, and PhoBERT all fail on the same case, it's usually the label, not the model. This caps achievable OFFENSIVE F1 around ~67-70% with current annotations.
2. **Implicit/contextual toxicity dominates** — `implicit_contextual` in 79% of failures: toxic via tone/target with no explicit slur (`no_toxic_vocab` in 60%). This is the largest *model-improvable* gap.
3. **Length-pattern inversion (H2 refuted)** — short comments are the EASIEST (F1≈0.71), long comments the HARDEST (F1≈0.58). OFFENSIVE F1 collapses from ~0.60 (short) to ~0.03 (very long); HATE moves the opposite way. Long text dilutes the offensive signal and triggers over-escalation (`long_context` = 73% over_escalate).
4. **Confidence miscalibration (H4 confirmed)** — ECE = 0.106; PhoBERT emits conf≈0.90-1.00 on wrong predictions. Softmax confidence is NOT a reliable abstention signal here.
5. **Surface-pattern categories are rare** — `teen_code` 19%, `sarcasm` 5%, `code_switching` 1% (H3 refuted — code-switching is negligible). Cheap dictionary fixes have limited ceiling.

## Hypothesis outcomes

| Hypothesis | Outcome | Evidence |
|---|---|---|
| H1 — OFFENSIVE annotation ambiguity | **CONFIRMED** | annotation_doubt 71%, D-subset 83% |
| H2 — short comments hardest | **REFUTED (inverted)** | short F1≈0.71 best, long F1≈0.58 worst |
| H3 — code-switching a major failure mode | **REFUTED** | code_switching only 1% |
| H4 — PhoBERT over-confident | **CONFIRMED** | ECE=0.106, conf≈0.9 on wrongs |

## Three-tier failure framework

- **Tier 1 — fixable (engineering)**: `teen_code` (19%). Extend the teencode/obfuscation normalisation dictionary in preprocessing.
- **Tier 2 — model-improvable (research)**: `implicit_contextual` (79%). Needs implicit-toxicity training data, target-aware features, or a larger/instruction-tuned model.
- **Tier 3 — fundamentally hard (data)**: `annotation_doubt` (71%). Bounded by label quality — only re-annotation or soft labels move this.

## Implications for deployment

- Do NOT use raw softmax confidence for auto-moderation thresholds (miscalibrated). Calibrate (temperature scaling) or route mid-confidence cases to human review.
- Expect the model to UNDER-flag subtle/implicit toxicity and OVER-escalate long comments — tune review queues accordingly (e.g. always human-review long OFFENSIVE-likely posts).
- PhoBERT is the deployment champion (test F1_macro 0.6618) but its edge over a 25k-param LR is modest (+0.044); for high-QPS settings LR + a PhoBERT fallback on borderline cases is viable.

## Limitations of this analysis

- Manual tagging n=100 (stratified, not random) — category %s describe the *failure* distribution, not the whole test set.
- Single annotator → no inter-annotator agreement; `annotation_doubt` is one person's judgement.
- Categories are non-exclusive (a case can be implicit + annotation_doubt), so %s sum > 100.
- Length effect is correlational; not controlled for class balance within buckets.

## Recommendations

1. **Short-term**: extend the teencode/obfuscation dictionary (`src/preprocess.py`) — cheap, addresses Tier 1.
2. **Medium-term**: re-annotate the OFFENSIVE/HATE boundary with stricter guidelines (or adopt soft labels) — lifts the Tier-3 ceiling that currently caps everyone.
3. **Long-term**: build/augment an implicit-toxicity dataset and consider target-aware or instruction-tuned models for Tier 2.

## Ready for Week 6

- 6 publication-ready figures in `results/figures/F1`-`F6_*.png` (dpi≥150, consistent palette).
- `results/master_results_table.csv` complete (every model × split).
- Error-analysis narrative (this doc + `week5_quantitative_findings.md` + `week5_qualitative_findings.md`) ready to drop into the thesis chapter.
