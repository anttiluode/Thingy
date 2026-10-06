# Verification record

The implementation was independently reviewed before publication. The reviewer
ran the original 31-check suite and exactly reproduced the primary 420-case JSON
receipt. Review found watch-loop races, inconsistent terminal snapshot status,
session exhaustion on reload, permissive empty interventions, numeric-import
boundaries, and unbounded length inputs. Each code issue received a failing
regression test before the fix. The final suite has **39 checks**.

The length-feature correction preserves all training inputs: original training
lengths were 0–4. The full unfinished query stays in the host residue; only the
controller feature is capped. Fresh primary receipts for all three streams were
generated after this correction. The separate post-review 32-step diagnostic
uses 100 fresh permutation worlds, seeds 0–99. It is not part of the predeclared gate.

Checks performed:

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -v`: 39 passing checks.
- Finite differences match exact BPTT gradients for each learned tensor.
- Actual curriculum training improves independent held-out task accuracy.
- `python -m compileall -q thingy tests`: passes.
- Node syntax check on the explorer script and `bash -n run.sh`: pass.
- The optional Node watch test executes the actual page script, controls timer
  wake-ups, and sends its requests to the real Python HTTP engine. Only rendering
  and DOM layout are stubbed; engine steps are real.
- Wheel build and installation into a clean target directory: passes.
- Demo from outside the repository: loads packaged trained weights and produces
  the correct packet trace/public result.
- Loopback HTTP tests execute separate sessions, edits, missing facts, malformed
  requests, snapshots, partial restoration, atomic failed restoration, and 65
  consecutive world reloads.
- Frozen paired benchmark: three streams of 420 fresh worlds, plus detailed
  multi-hop causal and interruption arms, with denominators and Wilson intervals.

Actual visual rendering/accessibility and the Windows batch launcher still need
manual platform checks. Their browser code syntax, HTTP behavior, and package
asset inclusion are verified. No visual browser binary was available in this
build environment. Wider research significance is not inferred from these small
task-family checks; text and latent recurrence also solve the primary benchmark.
