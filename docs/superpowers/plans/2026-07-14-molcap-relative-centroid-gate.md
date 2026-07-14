# MolCap Matched-Control EMA Centroid Execution Plan

> Required workflow: implement task-by-task with tests first, independent
> review at each source boundary, exact source relocking, and verification
> before completion.

**Goal:** Test the unchanged seed-7777 EMA-centroid MolCap intervention under a
prospectively frozen, matched latest-observation geometry gate, then probe and
submit every completed full run.

**Design:**
`docs/superpowers/specs/2026-07-14-molcap-relative-centroid-gate-design.md`.

**Baseline:** `213a74796e68641a852756e3afd76803ab11a367`.

## Task 1: Freeze the new config contract

- [ ] Add failing tests for `configs/molcap-ema-relative-s7777.yaml`.
- [ ] Copy the original centroid recipe and change only the declared project
  identity/output plus the new relative-gate leaves.
- [ ] Require seed `7777`, resume `null`, momentum `0.9`, latest momentum
  `0.0`, 256 permutations, the exact seed domain and frozen thresholds.
- [ ] Prove probe, FINO, DINO, target, ramp, crop, batch, and sample-budget
  leaves equal the original centroid config.
- [ ] Commit after config tests pass.

## Task 2: Add a latest-observation shadow with no training effect

- [ ] Write failing unit tests for latest-copy updates, equal-slide patient
  pooling, matched identities/counts, no gradient, and no RNG change.
- [ ] Reuse the deterministic hierarchy and grouped-reduction path.
- [ ] Add paired proposal validation so both diagnostic and EMA proposals are
  validated before either bank mutates.
- [ ] Return the shadow proposal separately; never use it as the MolCap
  forward value.
- [ ] Serialize the shadow only in pre-boundary full checkpoints used for
  calibration; probe checkpoints remain unchanged.
- [ ] Discard the shadow after a passed production boundary gate.
- [ ] Prove the old route and absolute-gate configs retain their prior paths.
- [ ] Commit after focused and integration tests pass.

## Task 3: Implement the deterministic relative audit and gate

- [ ] Write failing analytical tests for trace, spectra, effective-rank and
  participation ratios, centered alignment, CKA, raw cosine, permutation seed,
  exact p-value, inclusive/exclusive boundaries, and named failures.
- [ ] Implement CPU-float64 matched-population audit.
- [ ] Preserve the existing absolute gate for the archived config and dispatch
  by the explicit gate version.
- [ ] Emit strict JSON with complete spectra and permutation values on both
  pass and failure.
- [ ] Add checkpoint and summary provenance for gate version and shadow state.
- [ ] Commit after all focused tests pass.

## Task 4: Repair adjacent current-source contract defects

- [ ] Re-audit partial sample-prefix restoration, final gradient summaries,
  runner-cap preflight, and whole-run peak-memory reporting against current
  HEAD.
- [ ] Add the smallest failing regressions for findings that still reproduce.
- [ ] Fix only reproduced defects; do not change recipe behavior.
- [ ] Run the full local suite and independent review.
- [ ] Commit separately from the experiment implementation.

## Task 5: Relock source and operational execution

- [ ] Create an exact signed source bundle and manifest for the new commit.
- [ ] Prove the frozen probe/benchmarking manifest and target SHA are unchanged.
- [ ] Add byte-identical pre-training staging of the complete parquet dataset
  to local ephemeral storage, with source/destination file-count, byte-count,
  and deterministic-manifest equality. Preserve the configured path and keep
  every output on persistent storage.
- [ ] Extend the ignored Modal harness with new action names, distinct output
  paths, shadow-state audit, normalized checkpoint replay comparison, and
  failure archiving.
- [ ] Add harness contracts for source digest, config leaves, launch order,
  calibration stop, gate preview, replay certificate, and Labless eligibility.
- [ ] Obtain an independent exact-SHA review of the ignored harness.

## Task 6: CPU, smoke, calibration, and replay gates

- [ ] Run compilation, locked-file checks, and the complete local test suite.
- [ ] Run paired B200 mechanics smokes; require finite loss/gradients and no
  shadow effect on RNG, loss, or optimized gradients.
- [ ] Run 32,768-sample route and relative-centroid calibrations on B200 and
  exact H100 after the same validated local-data staging.
- [ ] Compare normalized training-state tensors to the four archived
  `213a747` checkpoints and require exact replay.
- [ ] Require exact EMA history digest, sample digest, target, mapping, and
  source evidence.
- [ ] Audit both B200 and H100 calibration shadow checkpoints and require both
  relative gates to pass. For each hardware independently, if maturity is
  below `0.95`, run exactly one frozen 40,960-sample preview on that hardware.
- [ ] Require relative-gate pass and centroid-minus-route peak memory at most
  `0.5 GiB` before a full launch.
- [ ] Calculate B200 and H100 timing projections without changing the recipe.

## Task 7: Full run, probes, and submission

- [ ] Recheck source, target, mapping, frozen files, replay certificate, and
  clean canonical output path.
- [ ] Run the fresh B200 seed-7777 relative-centroid arm from sample zero.
- [ ] At the first nonzero scale, persist and enforce the production relative
  gate; archive a gate failure and stop if it fails.
- [ ] If the gate passes, finish one million samples and every locked final
  probe.
- [ ] Compare against Labless route `run_sub_91ae661e33`, MiniLM, and
  `bsc-s7777-k10` using the frozen decision rules.
- [ ] Dry-run Labless validation, submit the completed result regardless of
  score, and verify the server-side source/metric record.
- [ ] Publish one durable result report with hashes, timing projection, gate,
  replay, metrics, submission IDs, and the next decision.

## Stop conditions

- A source/replay mismatch stops seed-7777 reuse and requires a separately
  preregistered paired seed-7778 experiment.
- A calibration or production relative-gate failure stops caption supervision.
- An infrastructure failure may retry the identical action; it does not permit
  a config, threshold, or seed change.
- No completed full run is withheld from Labless because its score is null or
  negative.
