# Thingy measured results

Frozen model, evaluation seed 202, 420 fresh fact graphs. 1,813 learned parameters. Training chain lengths 0–4; evaluation 0–6.

These are shared **controller ablations**, not four language transformers.

| Arm | Exact accuracy |
|---|---:|
| one_read | 32.4% (136/420) |
| text_feedback | 100.0% (420/420) |
| latent_recurrence | 100.0% (420/420) |
| self_ping | 100.0% (420/420) |

## Paired interventions on multi-hop episodes

| Intervention | Exact accuracy |
|---|---:|
| intact | 100.0% (300/300) |
| delete | 7.0% (21/300) |
| wrong | 6.7% (20/300) |
| freeze | 7.0% (21/300) |
| replay | 100.0% (300/300) |
| display_only | 100.0% (300/300) |

Correct replay on intact-correct cases: 100.0% (300/300). Wrong-ping-affected subset: 100.0% (280/280).

Different codebooks under the identical query: 100.0% (333/333) paired success.

## Interruption and selective restoration

The saved prefix is followed by an unrelated task. Facts and working state are cleared before each restore. Original public query metadata is retained in every arm. C includes all episodic facts and the fast operator. **C+r recovery is designed into this state split**, not a learned compression result. Numeric state counts exclude fixed theta, public query metadata, and diagnostic trace text.

| Restored state | Accuracy | Retained numeric scalars |
|---|---:|---:|
| codebook | 6.7% (20/300) | 1280 |
| workspace | 0.0% (0/300) | 37 |
| residue | 0.0% (0/300) | 1 |
| codebook+residue | 100.0% (300/300) | 1281 |
| workspace+residue | 0.0% (0/300) | 38 |
| all | 100.0% (300/300) | 1334 |

Zero-hop silent halt: 100.0% (60/60). Mean pings per episode: 3.00.

**Predeclared mechanism gate: PASS.**

- held_out_at_least_90pct: PASS
- delete_drop_at_least_20pp: PASS
- wrong_drop_at_least_20pp: PASS
- correct_replay_at_least_95pct: PASS

## What this establishes

A genuinely trained tiny controller uses a readable address channel that causally changes a fast operator and the next private state. Deleting, replacing, freezing, and replaying that channel produce measurable effects. Text and latent recurrence can also solve this deliberately simple world; these results do not establish an accuracy advantage over either.

## Limits

- Small shared-controller ablations, not trained transformer/CoT baselines.
- The parser, address basis, remaining-query cursor, fact matrices, and delta write are engineered.
- Codebook includes all facts plus fast operator; C+r reconstructs h by design, not learned compression.
- Soft recurrent training differs from hard sparse packet evaluation.
- Payload compares full instrumented JSON with short addresses; no compression advantage is claimed.
- No evidence of consciousness, spontaneous private language, or a replicated Jacobian lens.

Reproduce with `python -m thingy benchmark --seed 202 --episodes 420 --out results/reproduced`.
