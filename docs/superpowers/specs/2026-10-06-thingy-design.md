# Thingy: a learned, inspectable self-address loop

Antti's attached discussion supplies the design and the instruction to build it.
The desired machine has slow learned weights, a fast episode-specific codebook,
a private state, sparse readable/writable workspace packets, unfinished residue,
and a separate public decoder. Internal packets must cause subsequent computation.

## First implementation

A CPU-scale research instrument, Python 3.10+ and NumPy only. No API key, hosted
model, downloaded language model, or GPU is required. The existing MIT license
stays in place. A local browser explorer runs on loopback using the standard
library HTTP server. The project lives in the existing GitHub repository.

The training world has 16 named objects and four arbitrary relations. Each
episode supplies fresh random relation permutations. Queries ask for a chain of
relations starting at an object. For example, Zapp means tired; tired helps rest;
ask Zapp means helps. Object names can be replaced without retraining. Natural
language input is deliberately limited to a documented binding/query grammar.

The engineered address interface is a small named vocabulary, inspired by the
discussion; it is not an implementation of the Anthropic Jacobian lens or a
claim to have discovered spontaneous language or consciousness.

## Architecture and actual learning

* **Permanent theta:** a two-layer tanh neural controller and separate learned
  public decoder. Controller input combines the current instruction embedding,
  the private state, the amount of unfinished work, and its uncertainty.
  After review, the length feature is capped at the training range (0–4) while
  the complete pending query remains in r. This prevents out-of-range length
  values from changing the learned relation address.
* **Temporary codebook C:** fresh episodic relation matrices plus a fast linear
  operator. Facts enter this state without gradient updates to theta.
* **Private h:** a continuous distribution in the named object basis.
* **Workspace W:** sparse source and relation distributions, together with the
  latest retrieved candidate. This is a real input to the state update, not a
  prose interpretation printed after the answer.
* **Residue r:** the unfinished query and cursor; no target answer is stored.
* **X:** a configurable cost per internal ping. A learned HALT address supports
  silence at query completion. A host cycle cap guards runaway recurrence.

The emitted relation packet selects an episodic operator. Its read produces a
candidate distribution v. A delta-rule rank-one write changes the fast operator:

    F' = F + (v - F a) a^T / (a^T a)

Here a is the query's original source vector, so F' a stores the current result
of the composed trajectory. Next private state is computed by reading F' a.
This deliberately makes C plus r sufficient for recovery of h; recovery is a
designed state property, not an emergent memory-compression discovery. Facts and
the fast operator are both part of C. This must be clear in recovery reports.

Train with exact NumPy gradients through the continuous recurrent relaxation.
Use supervised self-address targets and teacher mixing at first, decay both,
then optimize final public-answer loss plus ping cost with no auxiliary target.
Hard top-k workspace packets are used during evaluation. Report teacher and
autonomous phases separately. Sparse selection is not differentiable; continuous
training and hard evaluation are distinct and both documented.

## Causal and memory experiments

Evaluation episodes use independent RNG streams and entire fresh fact graphs,
never training graphs. The target is computed by the task generator and is used
only by the trainer/scorer, never the agent runtime. Freeze theta for experiments.

1. Multi-hop held-out exact accuracy, grouped by chain length.
2. Identical current query under two different episode codebooks.
3. Delete or replace a packet; compare fast-operator drift and final answer.
4. Replace with an incorrect relation, then replay the correct packet from the
   identical pre-intervention snapshot. Recovery must be an independent branch.
5. Freeze the fast write while leaving the packet visible; narration alone must
   not move private h through the write/read channel.
6. Replace the whole state with an unrelated episode, then restore C, W, r,
   C+r, W+r, or all state. Public query metadata is retained in every condition;
   forgotten facts are not silently replayed. Record accuracy and retained scalar
   counts. Use both correct-prefix and all-prefix denominators.
7. Measure when it halts and how many nonempty packets it emits, including zero
   hop queries. No fixed correct action is injected during inference.

Comparisons are four **small controller ablations**, not four trained language
transformers: a one-read arm, textual packet feedback, continuous latent recurrence,
and the sparse self-ping loop. The recurrent arms share frozen learned weights
and the same episode facts. Text feedback serializes the same address in a strict
grammar and parses it back. No superiority claim is required or presumed. Report
actual accuracy, state sizes, and packet payload sizes. A larger transformer/CoT
comparison and learned promotion from unstructured sensory residue remain future
experiments, not mislabeled implementations.

Predeclared mechanism gate: held-out accuracy >= 90%, deleting/replacing packets
reduces accuracy by >= 20 percentage points on multi-hop queries, and replaying
the correct packet restores the intact result in >= 95% of eligible paired cases.
All pass/fail values must be reported, including a failure of any gate. These
thresholds establish the behavior of this instrument, not an AI research advance.

## Interaction and delivery

`python -m thingy demo` prints real packets and separate public output.
`python -m thingy train` saves human-readable JSON weights and training receipts.
`python -m thingy benchmark` writes JSON and Markdown evidence from frozen weights.
`python -m thingy serve` opens a local explorer: editable fact text and query,
step/run, workspace source/relation editing, delete/wrong/freeze interventions,
codebook matrix, private-state bars, trace, and pause/restore controls.

Ship a genuinely trained lightweight checkpoint and measured report so the demo
works immediately. Package static assets and checkpoint through explicit package
discovery; exclude tests/results from installed Python packages. Keep malformed
inputs, nonfinite state, incompatible checkpoints, stale snapshots, and cyclic
queries as explicit recoverable errors. Limit requests and bind to loopback.

Verification: independent hand-calculated task fixtures, finite-difference
gradient checks, training improvement, exact snapshot branching, interventions,
parser validation, HTTP integration, clean installation, full unittest suite,
and an independent whole-branch code review before GitHub publication.
