# Week 5 Phase 2: Qualitative Error Findings

_Auto-generated from 98/100 tagged cases (`manual_tagging_progress.csv`)._

## Category frequency

| Category | Count | % of pool |
|---|---|---|
| implicit_contextual | 79 | 79.0% |
| annotation_doubt | 71 | 71.0% |
| no_toxic_vocab | 60 | 60.0% |
| teen_code | 19 | 19.0% |
| very_short | 19 | 19.0% |
| long_context | 15 | 15.0% |
| sarcasm | 5 | 5.0% |
| code_switching | 1 | 1.0% |

## Universal failures (subset D)

- 30 tagged cases. Most common category: **annotation_doubt** (83% of D).
- Top-3 in D: annotation_doubt (83%), no_toxic_vocab (63%), implicit_contextual (60%)

## Category × failure direction

| Category | under_flag | over_escalate | correct |
|---|---|---|---|
| sarcasm | 3 | 1 | 1 |
| code_switching | 0 | 1 | 0 |
| teen_code | 9 | 7 | 3 |
| implicit_contextual | 27 | 30 | 22 |
| annotation_doubt | 30 | 30 | 11 |
| very_short | 5 | 9 | 5 |
| long_context | 2 | 11 | 2 |
| no_toxic_vocab | 25 | 21 | 14 |

## Phase-1 findings cross-check

- **F5 (subtle toxicity)**: in OFFENSIVE→CLEAN failures, `no_toxic_vocab` = 73%, `code_switching` = 0%.
- **F2 (long-context OFFENSIVE crash)**: of 15 long-context cases, 2 are OFFENSIVE under-flagged to CLEAN.
- **Annotation doubt**: 71 cases (71% of pool) flagged as label-questionable — supports the OFFENSIVE annotation-ambiguity hypothesis from Week 4.

## Recommendations

- If `no_toxic_vocab` + `implicit_contextual` dominate OFFENSIVE→CLEAN: the model lacks targeted/implicit-toxicity signal — consider data augmentation with implicit examples or a context-aware threshold.
- If `annotation_doubt` is high: a re-annotation pass (or soft labels) is the highest-leverage fix — caps the achievable ceiling otherwise.
- If `teen_code` / `code_switching` are frequent: extend preprocessing normalisation (teencode dictionary) and/or consider a multilingual checkpoint.
