# The machine, precisely

## Learned theta and fixed addresses

For each time step, the controller sees a fixed eight-dimensional instruction
embedding, sixteen private object coordinates, `min(remaining_query_length, 4) / 8`,
and continuous uncertainty `1 - sum(h*h)`. A 48-unit tanh hidden layer
feeds five address logits: four relations and HALT. The public decoder has its
own 16 × 16 weights and 16 biases. Total learned parameters: **1,813**.

Object coordinates are a deliberately chosen named basis, not a discovered
semantic basis. Renaming a slot changes its displayed name without changing its
vector. Instruction aliases have independent fixed random embeddings; the
neural controller learns their shared operational consequences.

## Episode C, private h, workspace W, and unfinished r

An episode supplies four matrices `B[r, target, source]`. Each generated matrix
is a fresh random permutation. User worlds may omit edges. An absent edge has
zero mass and causes unknown; it is never silently filled in.

The codebook contains **B and a fast operator F**. Initialize F as the identity.
Let a be the original query source vector, and h the private object distribution.
The hard runtime packet contains a sparse source key `k = top1(h)` and a learned
relation address `p = argmax(controller(features))`. Its candidate is:

    v = B[p] k

If v is known and the write is enabled:

    F_next = F + (v - F a) a^T / (a^T a)
    h_next = F_next a

This is a real rank-one write followed by a read. The current source column of
F stores the output of the composition so far. The temporary operator is a
summary for this query, not a globally learned new relation on every object.

W stores the source, selected relation, controller probabilities, and candidate.
The next computation consumes the source/relation packet. The displayed trace
is copied from executed values after the fact and is not an input to inference.
W is only a few active named coordinates even though JSON diagnostics include
full zero-filled arrays.

r is a cursor into the original public query. It is an engineered work queue;
it does not store the target. Its progress is advanced by the host after each
non-HALT cycle. X is presently an internal ping cost in the training objective,
not a separately learned motivational subsystem.

HALT is learned. If it arrives before all requested work is consumed, the agent
reports early halt. If it arrives afterward, a separate learned public decoder
maps h to the answer object. The sentence is a template. The host bounds cycles
at 40 by default and abstains when the budget is exhausted.

## Differentiable curriculum

For soft training, candidate vectors are `[B[0]h, ..., B[3]h, h]`, the last being
HALT's identity action. Controller probabilities mix them. Teacher mixing fades
from 0.8 to zero during the first 45% of optimization. Auxiliary packet
cross-entropy also fades to zero. The final phase minimizes answer cross-entropy
and a small cost on non-HALT probability.

The continuous mixing read is equivalent to writing that candidate with the
delta rule and reading the anchor column; the recurrent training code evaluates
the resulting candidate directly. It differentiates private-state dependencies
through every controller call, including the uncertainty feature. Hard top1
selection is not differentiated. The soft/hard distinction is explicit.

The tests check numerical derivatives of **every learned parameter tensor**,
fresh-graph training improvement, serialization, and frozen inference. Training
uses `SeedSequence([seed, 1])`; soft validation uses `[seed, 2]`. Frozen benchmark
worlds use `[evaluation_seed, 100]` with separate alternate/distractor streams.

## Causal branches and restoration

Delete and freeze skip the write/read while advancing the same task timing.
Replacing a relation or source changes the candidate, fast operator, and next h.
The replay experiment resets to an identical pre-intervention snapshot before
sending the correct packet; it does not try to repair the damaged branch while
retaining its other changes. Display-only edits mutate a diagnostic copy.

`interrupt()` clears B, F, h, W and r. The original public source/query remains
available in every condition. Restoring C recovers h by evaluating F a; restoring
W can recover its candidate object, but not missing facts. Restoring r recovers
the cursor. Full restoration also restores h and counters. Partial restoration
starts a fresh cycle budget. No original fact transcript is replayed.

Because C contains B and F and r supplies the pending position, **C+r is a
designed sufficient state**. Its success is a correctness property of this
machine, not evidence of discovered compression or biological consolidation.
The report counts 1,280 numeric scalars for C, 37 for W, one for r and 16 for h.
These counts exclude shared theta, common public query metadata and diagnostic
text, and are not claims about actual Python/JSON memory use.

Snapshots identify both episode and controller, validate arrays before mutation,
and copy arrays to prevent branch aliasing. Local HTTP clients have separate
sessions, locks and saved snapshots. Restore requests build a validated branch
before replacing the live agent.

Numeric imports reject tensor magnitudes above 1e6, far beyond the trained
checkpoint, and normalize conversion overflow into a validation error. Runtime
arithmetic rejects nonfinite probabilities rather than converting them into an
ordinary address. Snapshot terminal statuses must agree with unfinished work.
The browser's watch loops carry generation tokens, so a paused sleeper cannot
resume inside a later run. Loading a new world resets the existing browser session.

## Implementation map

| File | Responsibility |
|---|---|
| `thingy/tasks.py` | Fresh graph worlds, strict grammar, scoring oracle |
| `thingy/neural.py` | Learned controller/public decoder, exact recurrent gradients |
| `thingy/training.py` | Adam and guided-to-autonomous curriculum |
| `thingy/engine.py` | Sparse packets, fast writes/reads, trace and snapshot branches |
| `thingy/benchmark.py` | Paired interventions, ablations, interruption reports |
| `thingy/server.py` | Loopback sessions and live engine API |
| `thingy/web/index.html` | Browser explorer; no inference duplicated in JavaScript |
| `thingy/__main__.py` | Demo, train, benchmark and serve commands |

The inference engine never calls the target oracle. Tests replace that oracle
with a failing stub to catch leakage. Targets are used only by the trainer and
scorer. This task family is intentionally small and easy. Larger, matched-budget
comparisons and autonomous promotion from unstructured private state are future
research, not hidden behind the measured percentages.
