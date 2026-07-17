# MolCap Paired Seed-7778 Recovery Design

Status: prospectively preregistered before any seed-7778 mapping, sample-order,
model, gate, probe, or Labless metric is observed.

## Decision and boundary

Run a new paired experiment at unseen seed `7778`:

1. a fresh probe-route control; and
2. a fresh matched-latest EMA patient-centroid arm.

Both arms use one new source commit, one staged dataset manifest, the same
hardware sequence, and the same one-million-sample schedule. The primary
causal result is the seed-matched endpoint difference:

```text
relative-centroid-s7778 - probe-route-s7778
```

This recovery is a new preregistration. It does not amend, retry, reinterpret,
or continue the stopped seed-7777 experiment.

At the time of this preregistration, the only new fact observed after the
seed-7777 stop is an operational root cause: a path validator compared a
resolved Modal volume path with the unresolved logical mount root. The stop
occurred before `train.py`, so no seed-7778 sample digest, checkpoint, model
metric, centroid geometry, probe result, or score exists.

## Why this option

The original matched-control design already specified that a replay mismatch
must move the causal comparison to two fresh arms at unseen seed `7778`. The
runtime-path false positive prevented replay from being tested and wrote an
authoritative `required_seed: 7778` stop. A paired seed-7778 recovery therefore
preserves the experiment's integrity and tests the mechanism directly.

Rejected alternatives:

- Amend or retry seed `7777`: rejected because it would overwrite the meaning
  of an immutable source-mismatch stop and reuse a seed after observing its
  operational outcome.
- Run only a new relative-centroid arm against the old public seed-7777 route:
  rejected because the result would not be a seed-matched causal test.
- Stop the research line: rejected because the user approved the clean paired
  recovery and no training result has tested the hypothesis yet.

## Preserved seed-7777 evidence

The following records remain immutable historical evidence:

- source HEAD `06679b7b61e16b402601c694cea5851f2e7bec99`;
- the seed-7777 ledger at
  `/persistent/experiments/readout-local-context/matched-latest-s7777/evidence.json`;
- the Task 6 report at `.superpowers/sdd/task-6-report.md`;
- the public seed-7777 probe-route run `run_sub_91ae661e33`, submission
  `91ae661e33`, score `0.6637154886140434`;
- the prior seed-7777 calibration, checkpoint, mapping, sample-order, timing,
  and memory evidence.

The seed-7777 ledger receives no further actions. Its CPU preflight is not
credited to seed `7778`, and its stopped B200 smoke is not retried. The public
route result may be shown as historical context, but it is not the paired
control for the new mechanism estimate.

## Frozen scientific intervention

This design carries forward the intervention and gate from
`2026-07-14-molcap-relative-centroid-gate-design.md` without changing their
scientific leaves.

```text
route:    z_R = s_p + stop_gradient(t_p - s_p)
relative: z_C = s_p + stop_gradient(e_p - s_p)
```

`s_p` is the current student hierarchy, `t_p` is the current detached teacher
hierarchy, and `e_p` is the momentum-`0.9` equal-slide patient centroid. The
matched latest-observation bank remains audit-only and never enters the MolCap
forward or loss.

The following are unchanged in both arms from their respective seed-7777
parents:

- canonical MiniLM target bytes and target SHA-256
  `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`;
- target dimension `384`, MolCap weight `0.03`, and ramp `0.50` through `0.75`;
- hierarchical readout blocks `[4, 6, 8, 11]`, route, head dimensions,
  teacher forward source, and student identity straight-through gradient;
- DINO, JEPA, KDE, FINO, crop, batch, optimizer, schedule, probe, and
  one-million-sample settings;
- EMA momentum, equal-slide pooling, maturity and population rules;
- matched-latest relative audit, exact 256-permutation algorithm, thresholds,
  and fail-before-loss behavior;
- `resume: null`.

No momentum, target, weight, ramp, pooling, head capacity, transform, gate
threshold, permutation count, or endpoint rule may be changed after any
seed-7778 evidence is observed.

For avoidance of doubt, the relative arm's frozen production gate requires:

- target, mapping, finite-state, one-GPU, and exact matched-population identity;
- sample-weighted mature coverage at least `0.95`, `min_slide_updates = 2`,
  and at least `512` exactly matched patients;
- every EMA and latest patient-centroid norm greater than `1e-6`;
- positive finite covariance traces, `trace_ratio >= 1/19`, stored as
  `0.05263157894736842`,
  `effective_rank_ratio >= 0.5`, and `participation_ratio >= 0.5`;
- centered alignment greater than `0.0`;
- the frozen one-sided 256-row-permutation value at most `0.01`, using the
  target SHA, newly computed seed-7778 mapping digest, domain
  `molcap-matched-latest-v1`, and pinned PyTorch `2.8.0` procedure.

Raw cosine and absolute rank statistics remain diagnostic only. The report
must name every failed condition and retain the complete spectra and 256
permutation alignments exactly as specified by the parent design.

## Seed-7778 scored configurations

Add exactly these two scored configuration files:

| Role | Seed-7777 parent | New configuration |
|---|---|---|
| route | `configs/molcap-probe-route-s7777.yaml` | `configs/molcap-probe-route-s7778.yaml` |
| relative centroid | `configs/molcap-ema-relative-s7777.yaml` | `configs/molcap-ema-relative-s7778.yaml` |

Each new file may differ from its own parent at exactly four YAML leaves:

| Leaf | Route value | Relative value |
|---|---|---|
| `project.name` | `molcap-probe-route-s7778` | `molcap-ema-rel-s7778` |
| `project.output_dir` | `/data/$USER/nanopath/molcap/molcap-probe-route-s7778` | `/data/$USER/nanopath/molcap/molcap-ema-rel-s7778` |
| `data.split_seed` | `7778` | `7778` |
| `train.seed` | `7778` | `7778` |

`project.recipe_id` remains unchanged in each arm. Tests must flatten each
new-parent pair and prove that the permitted four-leaf set is exact. A
route-relative comparison must separately prove that all common recipe leaves
match and that the only scientific differences are the already-preregistered
history/gate intervention leaves.

## Source and artifact lock

Implementation creates one additive tracked source commit relative to parent
HEAD `06679b7`. Every tracked path that already exists at `06679b7` must retain
that commit's exact Git blob. Additions are limited to the two seed-7778
configs, their tests, and the approved recovery specification and plan. Thus
all pre-existing training, gate, probe, config, test, and benchmark files -
including `train.py`, `molcap_relative_gate.py`, `probe.py`, and both seed-7777
parent configs - remain byte-identical to `06679b7`.

Config tests compare each new config directly with its parent bytes loaded via
`git show 06679b7:<parent-path>`, not merely with the current working-tree
parent. A tracked-tree manifest test rejects any modified or deleted
pre-existing blob and any unapproved added path.

The operational harness is a new ignored artifact, not an edit to the frozen
seed-7777 harness:

- `.superpowers/sdd/modal_molcap_paired_s7778.py`;
- `.superpowers/sdd/test_modal_molcap_paired_s7778.py`.

Before any external seed-7778 action, freeze and independently review:

- the exact tracked source commit;
- source-bundle and source-manifest SHA-256 values;
- exact operational harness and harness-test SHA-256 values;
- both scored-config SHA-256 values;
- the Labless locked snapshot manifest;
- the action registry, run root, and authoritative ledger path.

Any source, config, target, mapping, locked-file, staging, path, mechanics, or
replay mismatch after that lock is a prospective source stop. It records
`required_seed: 7779` and launches no later seed-7778 action.

A retriable infrastructure failure is limited to an explicit provider
allocation/preemption, service/network transport, container loss, or volume
I/O failure that occurs before any scientific, identity, mechanics, replay, or
gate assertion fails. Each action permits at most two fresh retries after its
initial attempt, for three total attempts, under byte-identical artifacts and
from-scratch action state. Partial output is atomically moved to a failed
archive and is never resumed.

Source/config/target/mapping/staging/path/data-integrity mismatches, code
exceptions, mechanics/replay differences, OOM, non-finite training state, and
manual cancellation after training begins are not infrastructure retries.
Source, data-integrity, code, mechanics, and replay failures create the global
seed-7779 stop. A frozen relative-gate failure stops that relative action
before caption supervision. An OOM or non-finite state stops the affected arm
as incomplete. The one allowed maturity extension is the preregistered
40,960-sample preview, not a retry.

## Alias-safe runtime-path invariant

The seed-7777 stop was caused by validating physical containment against the
unresolved string `/persistent`. In Modal, `/persistent` can resolve to a
backend path such as `/__modal/volumes/<volume-id>`, while logical paths under
`/data` resolve through symlinks into that same backend.

The recovery harness preserves all configured logical path strings. It changes
only the physical containment comparison:

```python
persistent_root_resolved = PERSISTENT_ROOT.resolve()
dataset_resolved = DATASET_ROOT.resolve()

assert dataset_resolved == Path(config["data"]["dataset_dir"]).resolve()
assert not dataset_resolved.is_relative_to(persistent_root_resolved)

for logical_path in persistent_runtime_paths:
    assert logical_path.resolve().is_relative_to(persistent_root_resolved)
```

This establishes two simultaneous invariants:

1. `/data/nanopath_parquet` is the byte-verified container-local staged
   dataset and physically remains outside the persistent volume; and
2. outputs, W&B data, target, probe roots, Torch/Hugging Face caches, and other
   declared durable paths physically resolve inside the mounted volume even
   when their logical aliases begin with `/data`.

Tests must simulate a logical `/persistent` mount resolving to a distinct
backend root, accept valid `/data/...` aliases into that backend, reject a
resolved escape, and reject a staged-dataset alias into the persistent volume.
No test may weaken exact logical configured-path checks.

## New namespace and ledger

The recovery uses a disjoint execution namespace:

- Modal app: `nanopath-molcap-paired-s7778`;
- logical run root:
  `/data/experiments/readout-local-context/matched-latest-s7778`;
- authoritative ledger:
  `/persistent/experiments/readout-local-context/matched-latest-s7778/evidence.json`;
- local recovery journal:
  `.superpowers/sdd/molcap-relative-s7778-evidence.json`.

The new ledger begins empty and imports no completion state from seed `7777`.
It may record historical hashes as lineage only. Every seed-7778 action must
read, validate, update, and durably publish this new ledger before the next
action is eligible.

## Preprocessing and frozen identities

Before Python training on each GPU action, copy the complete persistent
Parquet dataset to container-local ephemeral storage. Validate file count,
total bytes, and a deterministic manifest before exposing it at the unchanged
logical path `/data/nanopath_parquet`. This is allowed intensive preprocessing
and is excluded from training wall time.

The target bytes, target SHA, staging algorithm, gate algorithm, gate
thresholds, locked probes, and unchanged runtime/gate blobs may be reused only
after fresh byte validation. The following seed-7777 evidence may not be
reused as seed-7778 evidence:

- ledger or completed-action state;
- source bundle, source manifest, Labless snapshot, or scored-config hashes;
- mapping or sample-order digests, even if a recomputed value happens to be
  equal;
- checkpoints, optimizer state, EMA/latest histories, or gate reports;
- mechanics, replay, memory, timing, calibration, preview, or probe results.

The CPU preflight recomputes and freezes the seed-7778 patient/slide mapping
digest and the canonical persistent-source dataset manifest. Each GPU action
then performs a fresh local stage and requires its destination manifest to
match that canonical source manifest before `train.py` starts.

CPU preflight does not predict presentation order. The current locked
`train.py` uses the global Torch RNG for `DataLoader(..., shuffle=True)` after
model and head initialization, so a no-model sampler oracle would not certify
the production sequence. Instead, the action order prospectively defines two
outcome-independent anchors after all artifact and staging checks:

- Action 1 atomically anchors the route smoke's first `1,024` `sample_idx`
  values; Action 2 must match it exactly.
- Action 3 must reproduce that first-`1,024` prefix, then atomically anchors the
  route calibration's first `8,192` values; Actions 4 through 6, every required
  `40,960` preview, and both full runs must match the `8,192` anchor exactly.

Each digest is `SHA256` over the `sample_idx` values in presentation order,
encoded as one contiguous C-order array of little-endian signed 64-bit integers
(`dtype="<i8"`) with no separator, label, or count prefix. The ledger stores
the digest, count, dtype, and anchor action. No metric or gate outcome may
select or replace an anchor.

## Frozen action matrix

Actions execute serially in this order:

| Order | Action | Hardware | Sample cap | Purpose |
|---:|---|---|---:|---|
| 0 | `preflight-paired-s7778-cpu` | CPU | N/A | source, config, target, mapping, probes, staging, and ledger identity |
| 1 | `smoke-route-s7778-b200` | B200 | 1,024 | route mechanics and staged path |
| 2 | `smoke-relative-s7778-b200` | B200 | 1,024 | relative mechanics and audit-only shadow |
| 3 | `calibrate-route-s7778-b200` | B200 | 32,768 | paired B200 route checkpoint, memory, and timing |
| 4 | `calibrate-relative-s7778-b200` | B200 | 32,768 | paired B200 relative checkpoint and gate preview |
| 5 | `calibrate-route-s7778-h100` | exact H100 | 32,768 | paired maintainer-hardware projection control |
| 6 | `calibrate-relative-s7778-h100` | exact H100 | 32,768 | paired maintainer-hardware relative preview |
| 7 | `full-route-s7778-b200` | B200 | 1,000,000 | fresh paired endpoint control |
| 8 | `full-relative-s7778-b200` | B200 | 1,000,000 | fresh mechanism endpoint |

For each hardware independently, if and only if its relative calibration has
sample-weighted maturity below `0.95`, insert exactly one
`preview-relative-s7778-{b200,h100}` action at `40,960` samples immediately
after that hardware's relative calibration. Do not run a preview when maturity
already reaches `0.95`; do not extend one hardware because the other needs an
extension; do not change the gate.

The full runs start from sample zero with `resume: null`, execute `7,812`
optimizer steps and `999,936` tile presentations, and run all locked probes.
They do not resume calibration checkpoints.

## Mechanics, replay, and geometry gates

Before either full launch, all of the following must pass:

1. Both arms use the exact same staged dataset manifest, mapping digest, and
   seed-7778 sample prefix on each paired hardware.
2. At each paired 32,768-sample checkpoint, the normalized common core is
   byte-exact between route and relative: student, teacher, DINO heads,
   predictor, FINO, MolCap head, optimizer, counters, and sample prefix.
   Exclusions are limited to explicit source/config/output metadata and the
   preregistered history/shadow payloads.
3. The route arm contains no EMA history, latest shadow, matched-control gate
   state, or provenance for those structures.
4. The relative arm's EMA and latest histories independently satisfy their
   identity, update, maturity, serialization, and population invariants.
5. Synthetic and real B200 shadow-on/off mechanics checks show exact RNG,
   optimized loss, optimized gradients, primary EMA proposal, and common-core
   state equality. These checks isolate the audit-only latest shadow; they do
   not require the route and EMA proposals themselves to be equal.
6. Both B200 and exact-H100 relative previews pass every frozen matched-latest
   hard gate. A permitted `40,960` preview replaces only its own immature
   `32,768` preview for this decision.
7. Relative-minus-route peak allocated memory is no greater than `0.5 GiB` on
   each paired hardware.

After Actions 0 through 6 and any required previews pass, the harness writes
one strict-JSON aggregate launch certificate. It seals the exact source,
harness, config, target, locked-tree, canonical dataset, mapping, sample-anchor,
action-result, common-core checkpoint, history/shadow, mechanics, gate,
preview-decision, memory, hardware, and timing hashes or values. The
authoritative ledger stores the certificate SHA-256.

Both full actions must verify the same certificate bytes and SHA-256 before
launch; completing the route full may append ledger evidence but may not alter
the sealed certificate. Each full action independently restages the dataset,
revalidates destination against the certificate's canonical source manifest,
and reproduces the first-`8,192` sample anchor. Any certificate mutation or
restaging mismatch is the global seed-7779 stop.

Timing is diagnostic, not a scientific gate. Report B200 wall time, exact-H100
wall time, their observed ratio, and the H100-equivalent projection for each
full B200 arm. A projection above the maintainer's two-hour rerun window is
reported plainly but does not suppress a completed public submission.

The route full run has no centroid geometry gate. The relative full run repeats
the production matched-latest gate immediately before its first nonzero MolCap
scale. A production-gate failure stops before caption supervision and is not a
completed full run; the already completed route control remains eligible for
submission.

## Endpoint analysis and submission rule

Submit every completed full arm to Labless regardless of score or hypothesis
outcome. Preserve null and negative results in the public record. Publication
is outside the training-action dependency chain: the relative full launch must
not depend on the route Labless response, dry-run/server-side validation state,
or public score. It still requires Action 7 to be locally complete and
artifact-validated in the authoritative ledger, because that completed route
run is its paired endpoint control.

Labless transport/service failures retry the immutable completed artifact
independently and do not consume a training-action retry or change seeds. A
deterministic policy/source rejection is recorded and escalated without
changing source, metrics, or the other arm's execution. If the route completes
and the relative arm later stops, the route remains submission-eligible.

The centroid mechanism is supported by the paired seed-7778 comparison only
if all of the following hold:

- progression, mutation, and survival each improve in relative minus route;
- their unweighted mean improves by at least `0.003`;
- linear, kNN, and few-shot each decline by less than `0.003`.

Formal BSC-based promotion is out of scope for this two-arm recovery. The
historical score threshold `0.6719107210` and historical BSC/frontier metrics
may be reported descriptively, but their linear/kNN guards are not applied as
a seed-7778 promotion decision because no seed-matched BSC arm is run. No third
BSC arm is added. The only preregistered scientific decision is the paired
mechanism rule above; every completed arm is still publicly submitted.

A passed geometry gate followed by endpoint failure is a negative test of the
EMA-centroid mechanism. A relative production-gate stop is a failed eligibility
test for that arm, not an endpoint result. No post-result threshold, seed,
pooling, encoder, or submission rule may be changed.
