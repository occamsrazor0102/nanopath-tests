# MolCap Probe-Route and EMA Patient-Centroid Experiment Design

## Goal

Run a paired, seed-7777 experiment that answers two separate questions:

1. Does moving MolCap from final-layer patch means onto the exact
   block-strided CLS feature family used by the frozen probes change the
   outcome?
2. Holding that route fixed, does a historical slide-to-patient EMA target
   improve progression, mutation, and survival without materially damaging
   tile discrimination?

The first question is measured by a route-control arm. The second is measured
by an EMA-centroid arm that differs from the route control only by its
historical teacher state. Both arms reuse the exact canonical MiniLM caption
targets and every frozen training choice from `molcap-text-s7777`.

This design calls the new representation a **probe-aligned surrogate**. It is
not literally the downstream probe mean: training sees stochastic augmented
TCGA tiles, while evaluation sees deterministic external-cohort tile grids;
survival also averages slides equally within each case.

## Evidence and Hypothesis

The trained `molcap-text-s7777` arm established a useful but incomplete
mechanism:

- linear: `-0.000055` versus `bsc-s7777-k10`;
- kNN: `+0.001071`;
- progression: `+0.003138`;
- mutation: `-0.002331`;
- survival: `-0.004043`;
- segmentation: `-0.002141`;
- overall: `-0.000687`.

That arm routed each tile's final-layer patch mean through MolCap. It was
tile-safe and progression-positive, but it did not improve mutation or
survival. The two biomedical target experiments stopped at offline geometry
gates and never trained, so they do not test either biomedical semantics or
the centroid hypothesis.

The frozen evaluation path uses `model.probe_features()`: normalized CLS
vectors from blocks 4, 6, 8, and 11 are concatenated into a 1,536-dimensional
ViT-S readout. Progression and mutation average these vectors over tiles to a
slide. Survival first averages tiles to slides and then averages slides
equally within a case. The full metadata CSV contains roughly 30,324 non-null
slide barcodes across 11,120 patients, with a median of two slides per patient.
These figures are context, not bank dimensions; the exact bank universe is
constructed and reported from training-shard tile paths after the locked
split. Even so, the multi-slide prevalence shows why a flat tile-weighted
patient EMA is not an adequate surrogate for the survival pooling operator.

The experiment's primary hypothesis is that a historical, teacher-stabilized,
equal-slide patient representation is a better molecular caption carrier than
a current-tile representation, while a straight-through student path keeps
the shared encoder trainable.

## Approaches Considered

### Selected: online hierarchical teacher EMA

Maintain a raw fp32 EMA for every training slide using the EMA teacher's exact
1,536-dimensional block-strided CLS readout. Form each patient representation
by equally averaging the patient's observed slide centroids. The state adapts
as the representation changes and matches the downstream tile-to-slide and
slide-to-case hierarchy as closely as an online training objective can.

### Rejected: flat patient EMA

A flat patient EMA weights slides in proportion to sampled tiles. That differs
materially from the equal-slide survival operator because most patients have
multiple slides. It is smaller and simpler, but it would weaken the claimed
pooling alignment.

### Rejected as the primary arm: offline baseline centroids

Offline centroids are deterministic and can exactly summarize a frozen
baseline over a fixed tile population. They become stale when training begins.
Direct captioning of a fixed detached centroid gives the trunk no gradient;
distilling current features toward it constrains the new model toward the old
model. Such centroids may be useful later as an audit artifact, not as this
experiment's adaptive objective.

### Rejected: within-batch patient means alone

Under ordinary random sampling, most patients appear once in a batch. Their
two global views are augmentations of the same tile, so a nominal patient mean
usually degenerates to an augmentation mean. A patient-aware sampler would
alter DINO, Sinkhorn, KDE, and FINO batch composition and introduce a larger
confound.

## Paired Experimental Arms

### Arm R: probe-route control

For each training tile:

1. Run an additional **unmasked student** forward over both global views.
2. Extract the same normalized block-4/6/8/11 CLS concatenation as
   `probe_features()`, producing 1,536 dimensions per view.
3. Average the two global-view readouts for the tile.
4. Within the current batch, average tiles per slide and then average the
   present slides equally per patient.
5. Form the corresponding current-batch hierarchy from the unmasked EMA
   teacher readout.
6. Use the current teacher patient value in the forward pass and the current
   student patient value in the backward pass through the registered
   identity-gradient estimator.
7. Pass the result through a `1536 -> 512 -> 384` MolCap head and apply the
   existing caption cosine loss.

This arm has no historical centroid bank. It measures the effect of changing
the MolCap input route, teacher-valued forward, and aggregation while holding
the caption semantics fixed.

### Arm C: hierarchical EMA centroid

Arm C performs the same unmasked student and teacher readout collection, uses
the same helper, head, current-batch hierarchy, identity-gradient estimator,
target, and loss. It additionally:

1. proposes per-slide teacher EMA updates;
2. equally averages the patient's observed slide centroids; and
3. substitutes that historical teacher value for Arm R's current teacher value
   in the forward pass.

The two arms must execute the same student and teacher readout collection and
consume no arm-specific random numbers. Centroid grouping and bank operations
are deterministic and add no RNG consumption.

### Frozen factors

Both arms inherit from `molcap-text-s7777` without semantic changes to:

- canonical target artifact SHA-256
  `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`;
- exactly 11,428 canonical patient identifiers, captions, and 384-D targets;
- FINO coverage and metadata;
- seed 7777 and data split;
- one-million-sample budget;
- batch size, global/local crops, view count, augmentations, and sampling;
- ViT-S backbone, DINO, JEPA, KDE, and FINO objectives;
- MolCap weight `0.03`;
- sample-fraction ramp: zero through `0.50`, linear through `0.75`, then full;
- optimizer, schedules, checkpoint cadence, and frozen probe configuration.

`probe.py` and `benchmarking/` remain byte-identical. The canonical target
builder and target NPZ remain unchanged.

The two new configs are
`configs/molcap-probe-route-s7777.yaml` and
`configs/molcap-ema-centroid-s7777.yaml`. Their allowed semantic diff is exactly
four leaves:

| Leaf | Arm R | Arm C |
|---|---|---|
| `project.name` | `molcap-probe-route-s7777` | `molcap-ema-centroid-s7777` |
| `project.recipe_id` | `dinov2-vits14-reg-jepa-mask10-molcap-probe-route` | `dinov2-vits14-reg-jepa-mask10-molcap-ema-centroid` |
| `project.output_dir` | `/data/$USER/nanopath/molcap/molcap-probe-route-s7777` | `/data/$USER/nanopath/molcap/molcap-ema-centroid-s7777` |
| `molcap.history.enabled` | `false` | `true` |

Both configs explicitly contain identical values for the feature blocks,
1,536-D input, 512-D hidden width, current-teacher forward source,
identity-student gradient rule, hierarchy, momentum, maturity gates, geometry
gates, target, weight, and ramp. Arm R is not permitted to revert to the old
384-D patch route.

The common routing/history leaves are named and frozen as follows:

```yaml
molcap:
  route: probe_cls_hierarchical
  feature_blocks: [4, 6, 8, 11]
  input_dim: 1536
  head_hidden_dim: 512
  forward_source: teacher
  gradient_source: student_identity_ste
  history:
    level: slide_then_patient
    momentum: 0.9
    min_slide_updates: 2
    min_sample_weighted_coverage: 0.95
    min_geometry_patients: 512
    min_effective_rank: 32
    min_participation_ratio: 16
    max_mean_offdiag_cosine: 0.95
    min_centroid_norm: 1.0e-6
```

`molcap.history.enabled` is present in both files. All other leaves are copied
from `configs/molcap-text-s7777.yaml` unless explicitly frozen above.

## Exact Readout Contract

The training implementation must expose one shared helper for the probe
readout rather than reproducing the block selection separately. An eval-mode
regression test uses identical unmasked input and weights and must satisfy all
of the following:

- exact output shape `(N, 1536)`;
- agreement with `model.probe_features()` at absolute tolerance `2e-5`;
- agreement at the same tolerance with an independent reference implementation
  pinned to pre-experiment commit
  `01c1cdf8017a0481636a28ab58a0ddc67d6e0a06`, fixed blocks
  `(4, 6, 8, 11)`, per-block CLS normalization, and concatenation.

The independent reference must not call the newly shared helper, so a mistaken
joint refactor cannot certify itself.

The existing masked student global forward cannot satisfy this contract.
Both arms therefore add one unmasked student global forward over the same two
global-view tensors. The existing unmasked teacher global forward may expose
its intermediate CLS states without another teacher backbone pass.

The numerical equivalence gate above is intentionally eval-mode. Production
training retains the frozen train-mode student semantics, including DropPath,
so it claims the same block/normalization feature family rather than bitwise
equality to an eval-mode probe vector. The auxiliary student forward runs only
after all base-objective stochastic forwards. Its RNG is isolated in a
save/restore context and seeded deterministically from the locked run seed and
global step; CPU and CUDA RNG states outside that context must be bitwise
unchanged. Both arms use the same local seed. A test proves this isolation and
paired output, and the frozen production config keeps activation checkpointing
disabled.

The raw 1,536-D concatenation is pooled before projection. It is not
L2-normalized as a whole, because the frozen probes average the concatenated
readout itself. The MolCap head retains its existing normalized 384-D output.

Views are flattened crop-major by the training loop. Patient and slide indices
must use the corresponding crop-major repetition, never an interleaved
repetition. The two views are restored to tile identity and averaged before
any slide or patient grouping.

## Stable Training Identity Mapping

The dataset must emit deterministic contiguous slide and patient indices in
addition to its existing hashes. The training patient set is derived
exclusively from training-split tile paths. Filter the canonical target-bank
identity list to those training patients, preserve canonical NPZ order, and
renumber densely from `0` through `num_train_patients - 1`. Every training
patient must occur exactly once in the canonical list. Target-only and
validation-only identities receive no centroid index. Sort canonical raw
training slide identifiers lexicographically and renumber them densely from
`0` through `num_train_slides - 1`:

- every slide maps to exactly one patient;
- raw identities are used to validate the mapping before hashes are discarded;
- existing 63-bit hashes remain diagnostic fields, not bank indices.

The ordered patient list, ordered slide list, and dense slide-to-patient vector
are serialized into one versioned mapping digest. Duplicate identifiers,
collisions, cross-patient slide assignments, missing caption targets, or any
train/validation mutation path abort before training.

The repository remains single-GPU. `WORLD_SIZE > 1` is a hard abort. DDP bank
synchronization is outside this experiment's scope.

## Hierarchical Aggregation

For a batch with student tile features `s_i` and detached teacher tile
features `t_i`, both already averaged across global views:

1. group features by contiguous slide index and take an unweighted mean over
   current-batch tiles for each slide;
2. for the route control, group those current slide means by patient and take
   an unweighted mean over present slides;
3. for the centroid arm, propose an update for every present slide, then take
   an unweighted mean over all observed slide states belonging to the patient,
   substituting the proposed value for slides present in the current batch.

This makes repeated tiles within one slide affect the slide estimate but not
the relative weight of that slide against a second slide from the same
patient. Grouping must be invariant to batch row order.

## Centroid State and Update Rule

Arm C allocates:

- `slide_centroids`: fp32 tensor `(num_train_slides, 1536)`;
- `slide_counts`: int64 successful-update counts, incremented once per
  optimizer step in which a slide is present regardless of tile multiplicity;
- `slide_tile_presentations`: int64 number of training tiles accumulated for
  each slide;
- `centroid_state_step`: scalar step represented by the committed bank;
- mapping and configuration metadata needed to validate the state.

For current detached teacher slide mean `t_s`, prior centroid `b_s`, and fixed
momentum `mu = 0.9`:

```text
if count_s == 0:
    next_s = t_s
else:
    next_s = 0.9 * b_s + 0.1 * t_s
```

First observation copies rather than blending with zero. The bank stores raw
readouts; neither slide nor patient centroids are normalized before the
MolCap head. A patient centroid is an equal mean of its observed slide
centroids. Slides with count zero are absent, not zero-filled.

The bank warms from sample zero while the MolCap scale is zero. A proposed
update is constructed from detached teacher features and used to form the
current loss. A successful optimizer step means that the total loss passes an
explicit finite check, all optimized gradients pass an explicit finite check,
and `optimizer.step()` returns without exception. Only then do
`slide_centroids`, `slide_counts`,
`slide_tile_presentations`, and `centroid_state_step` commit atomically under
`no_grad`, exactly once. The stored features come from that iteration's
pre-optimizer teacher forward. The normal EMA-teacher update then runs.
`centroid_state_step` must equal the full-checkpoint step; disagreement fails
closed. Validation, probe, and diagnostic-only forwards never mutate the bank.

The exact feature-bank allocation is
`4 * num_train_slides * 1536` bytes and must be reported from the constructed
training mapping; the full-metadata count is not used as a dimension. Counts
and mapping add little relative to the feature tensor. Arm C's added peak
memory over Arm R, including all centroid machinery, must remain at or below
`0.5 GiB`.

## Straight-Through Gradient Contract

A detached teacher value cannot train the student trunk. Let `s_p` and `t_p`
be the current-batch hierarchical student and detached teacher patient means.
Let `h_p` be Arm C's detached historical teacher patient centroid after
substituting proposed updates for present slides. The registered estimators are:

```text
Arm R: z_R = s_p + stop_gradient(t_p - s_p)
Arm C: z_C = s_p + stop_gradient(h_p - s_p)
```

Arm R's forward value is exactly `t_p`; Arm C's is exactly `h_p`. Both local
derivatives to `s_p` are the identity. The arms therefore share a
teacher-valued forward, the same identity-gradient student path, and the same
nonlinear head. Historical substitution is the substantive treatment.

This is an explicitly chosen surrogate, not the mathematical derivative of
the EMA, which would scale a present contribution by `1 - mu = 0.1` and would
not differentiate through unavailable historical tiles. The identity
coefficient avoids building deterministic tenfold attenuation into the
centroid arm. It does **not** guarantee equal numerical gradients, because the
shared nonlinear head is evaluated at `t_p` versus `h_p`. Any result is evidence
for this registered historical-teacher-forward/identity-gradient objective,
not for exact backpropagation through a true patient mean.

`s_p` and `t_p` average only slides present in the current batch. `h_p` uses
the same proposed teacher values for those slides plus detached historical
values for previously observed absent slides. This support difference is the
registered history intervention.

Targets are applied once per unique current-batch patient. All target-present
patients participate after the ramp begins. At most the pre-registered
immature tail may still be on its first update, in which case Arm C equals the
current-teacher control for that slide rather than receiving historical
information. The maturity gate bounds this treatment dilution at the
population level, and the nonhistorical fraction is reported at every normal
diagnostic interval. No per-patient eligibility mask changes the paired loss
population.

## Checkpoint and Resume Contract

Arm C full checkpoints include:

- fp32 slide centroids, exact integer update counts, tile-presentation counts,
  and `centroid_state_step`;
- canonical target SHA-256;
- ordered patient/slide mapping digest and mapping version;
- feature blocks, feature width, momentum, hierarchy, and STE version;
- arm identity and all MolCap schedule constants.

Checkpoint round-trip must preserve centroids, update counts,
tile-presentation counts, and state step bitwise. A missing bank, missing
metadata, changed mapping digest, changed target SHA, shape mismatch, changed
constants, step disagreement, or centroid state attached to the wrong arm
fails closed. Centroid state is stored only in full training checkpoints.
Probe checkpoints remain backbone-only and return without exposing the bank or
head.

Scored paired runs start from scratch. If infrastructure interrupts an arm,
that arm restarts from scratch with the locked seed rather than resuming a
different sample position. Checkpoint restoration is still implemented and
tested for safety, but it is not used to claim bit-identical paired ordering
because the current loader does not checkpoint every worker/sampler RNG state.

## Pre-Training Gates

All gates are pre-registered. A failure stops before full caption-supervised
training; constants are not revised in response.

### Identity, readout, and hierarchy

- exact canonical target SHA, exactly 11,428 unique canonical identities and
  finite unit targets, and 100% of training-split tile patients resolving to
  exactly one canonical caption target;
- exact 1,536-D eval-mode readout agreement with both `probe_features()` and
  the independent pre-experiment reference at `atol=2e-5`;
- correct crop-major view alignment and two-view tile averaging;
- order-invariant tile-to-slide and equal-slide-to-patient aggregation;
- synthetic unequal-tile, multi-slide examples agree within `1e-6`;
- deterministic mapping digest, no identity collision, no cross-patient slide,
  and no validation-bank mutation;
- `WORLD_SIZE == 1`.

### State and gradient behavior

- first slide observation copies the teacher mean exactly;
- later update implements fixed `mu=0.9` exactly;
- bank values and teacher update path require no gradient;
- Arm R uses the current teacher patient mean in forward and the identity
  derivative to the current student hierarchy in backward;
- Arm C changes only that forward value to the historical teacher patient mean
  while retaining the same identity derivative;
- finite loss and nonzero gradients reach both the MolCap head and shared
  student trunk;
- the existing missing-target mask remains a compatibility/test path, but zero
  missing targets are permitted in either scored arm;
- checkpoint round-trip preserves centroids, update counts,
  tile-presentation counts, and state step bitwise, and every mismatch fails
  closed.

### Pairing and configuration

- both arms produce the same digest for at least the first 8,192 sampled tile
  identities;
- both arms instantiate identical readout helpers, additional forwards, head
  shapes, initial head weights, optimizer groups, and schedules;
- the semantic config diff is exactly the four declared leaves;
- before smoke, before each full run, before submission, and inside each saved
  `labless_source`, `probe.py` and `benchmarking/` match pre-experiment commit
  `01c1cdf8017a0481636a28ab58a0ddc67d6e0a06` by SHA-256;
- the canonical NPZ independently matches its declared SHA-256 at those same
  gates.

## Ramp-Boundary Bank Gates

The bank continues to warm with zero caption scale through sample fraction
`0.50`. Immediately before the first nonzero centroid scale, Arm C must pass:

- at least 95% sample-weighted mature-slide coverage, where a mature slide has
  at least two successful updates and boundary coverage is exactly
  `slide_tile_presentations[slide_counts >= 2].sum() /
  slide_tile_presentations.sum()` over the locked stream;
- at least 512 observed patient centroids available for geometry measurement;
- effective rank at least `32`;
- participation ratio at least `16`;
- mean off-diagonal cosine below `0.95`;
- every all-observed patient centroid norm greater than `1e-6`;
- finite bank, counts, metrics, and loss state.

Hard geometry thresholds use the exact all-observed patient centroids entering
the forward path: every slide with `slide_counts > 0` participates and
count-zero slides are absent. A second mature-only table, restricted to slides
with `slide_counts >= 2`, is reported diagnostically but does not replace the
forward-population gate. The audit records population size, count
distributions, slides per patient, norms, effective rank, participation ratio,
off-diagonal cosine, and centroid drift.

The hard metrics are computed deterministically on CPU in float64. For raw
patient-centroid matrix `X` with `n` rows, subtract the column mean and let
`lambda_j` be the nonnegative eigenvalues of sample covariance
`X_centered.T @ X_centered / (n - 1)`, with negative numerical roundoff clipped
to zero. Require positive total variance and define:

```text
p_j = lambda_j / sum(lambda)
effective_rank = exp(-sum(p_j * log(p_j)))  # zero p_j terms omitted
participation_ratio = sum(lambda)^2 / sum(lambda^2)
```

For cosine, L2-normalize the **uncentered** rows to `U` and compute the signed
mean over all unordered off-diagonal pairs without materializing the full Gram
matrix:

```text
mean_off_diagonal_cosine = (||sum_rows(U)||^2 - n) / (n * (n - 1))
```

Teacher centroid drift is diagnostic, not a hard geometry threshold. It is the
cosine between `b_s` and `next_s` for updates with prior `slide_counts > 0`,
reported as mean and fixed quantiles; first-copy updates are excluded.

The maturity threshold is a population-level gate. After it passes, no
per-patient or per-slide maturity mask changes the paired loss population;
count-one slide states may participate, and their prevalence is reported.

Any maturity, geometry, finite-value, or state-integrity failure aborts before
caption supervision. It does not trigger a momentum, ramp, or threshold edit.

## Diagnostics

Both arms log:

- MolCap loss, effective scale, target coverage, and unique patients;
- current slide and patient group counts;
- gradient cosine and norm ratio between base objectives and MolCap on the
  final block attention projection;
- readout/head norms, throughput, FLOP/s, and peak memory;
- initial sample-order digest and source/config/target hashes.

Arm C additionally logs:

- observed and mature slide counts and sample-weighted maturity coverage;
- slide update-count quantiles and observed slides per patient;
- teacher centroid drift cosine;
- centroid-caption cosine;
- patient-centroid geometry at the ramp boundary and normal diagnostic
  intervals;
- bank memory and checkpoint-state digest.

Diagnostics must not add stochastic forwards or mutate state outside the
normal training update.

## Hardware-Efficient Execution

Public Labless submissions have no wall-clock limit, and intensive
preprocessing before model training is excluded from training time. The
public arms therefore use the fastest compatible available single GPU, with a
Modal B200 preferred. Hardware changes execution time, not the algorithm or
configuration.

Before full runs, perform paired 32,768-sample calibrations on B200 and exact
H100 with probes disabled. Each calibration retains the one-million-sample
schedule denominator and every production constant, but an external runner
stops it after 32,768 presented tiles. It therefore exercises the production
readout, grouping, proposed bank updates, head forward, and backward graph at
the production schedule value without invoking the 50% ramp-boundary gate.
Separate mechanics tests force a nonzero local test scale to verify STE
gradients; they are not timing calibrations and never publish results. After a
B200 full run, calculate:

```text
projected_H100_train_seconds
= observed_B200_full_train_seconds *
  (steady_state_B200_throughput / steady_state_H100_throughput)
```

Report hardware identities, CUDA/PyTorch versions, train-loop time, visible
patches/s, steady-state FLOP/s, peak memory, and the projected H100 margin to
7,200 seconds. A projection over two hours does not suppress a completed public
submission; it is disclosed as reduced confidence in maintainer validation.

If B200 is unavailable or incompatible, use one declared single-GPU fallback
for both arms, preferring H200 and then exact H100. Do not add distributed
training or change batch semantics to save time.

## Execution Order

1. Run unit, integration, checkpoint, compile, and locked-file tests on CPU.
2. Run a short B200 mechanics/gradient smoke for Arm R with probes off.
3. Run the corresponding Arm C smoke with probes off.
4. Run paired 32,768-sample B200 and exact-H100 calibrations and calculate the
   H100-equivalent full-run projection.
5. Verify clean source, exact target, locked configs, and sample-order pairing.
6. Run the full one-million-sample Arm R on B200 or the declared fallback.
7. Run the full one-million-sample Arm C on the same hardware regardless of
   Arm R's score.
8. Run the complete locked final probes for both arms.
9. Compare the paired arms and both historical references using the frozen
   decision rules.
10. Dry-run and submit every completed full run to Labless regardless of score.
11. Record hashes, gate evidence, timing projection, metrics, submission IDs,
    and decisions in durable result reports.

Smoke and calibration runs are never submitted. A completed full run is not
discarded because it is null or negative.

## Decision Rules

### Primary centroid mechanism test

Arm C supports the centroid mechanism versus Arm R only if all are true:

- progression improves;
- mutation improves;
- survival improves;
- the unweighted mean of progression, mutation, and survival improves by at
  least `0.003`;
- linear declines by less than `0.003`;
- kNN declines by less than `0.003`;
- few-shot declines by less than `0.003`.

This is the causal historical-substitution comparison: route, teacher-valued
forward source, identity-gradient student path, head, schedule, target, and
training recipe are paired. The treatment changes `t_p` to the registered
historical `h_p` and adds only the deterministic state required to construct it.

### Secondary route test and historical comparisons

Arm R is compared with `molcap-text-s7777` to estimate the joint effect of the
probe-CLS route, current-batch hierarchy, and teacher-forward/student-gradient
estimator. Both Arm R and Arm C report all public components against:

- `molcap-text-s7777` at seed 7777; and
- `bsc-s7777-k10` at seed 7777, overall `0.6659107210`.

The route comparison is secondary because the input feature family, head
input width, aggregation, and estimator all change together relative to the
original patch arm.

### Promotion and stopping

An arm with overall score at least `0.6719107210`, with linear and kNN each
declining by less than `0.003` versus seed-matched `bsc-s7777-k10`, triggers two
additional locked seeds after its seed-7777 Labless submission. The primary
three-metric rule remains reported even if overall crosses this threshold.

No momentum, loss-weight, ramp, pooling, or head-capacity sweep is permitted
after observing these paired results. A negative result closes this exact
historical-forward/identity-gradient centroid arm rather than being repaired
post hoc.

## Implementation Scope

Expected implementation changes are limited to:

- deterministic training identity indices in `dataloader.py`;
- a shared exact block-strided readout path and unchanged MolCap head building
  blocks in `model.py`;
- paired routing, hierarchical grouping, bank state, gates, logging, and
  checkpoint integration in `train.py`;
- two locked seed-7777 configs;
- focused unit and integration tests;
- smoke/calibration launch plumbing and durable result reports.

`probe.py`, `benchmarking/`, existing MolCap target builders, canonical target
artifacts, and historical result records are outside the change surface.
