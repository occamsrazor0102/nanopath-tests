# MolCap Experiment Design

## Goal

Test whether a frozen generic text embedding of patient metadata can improve nanopath's slide-level mutation, progression, and survival probes without reducing tile-level linear, kNN, or few-shot performance.

The experiment starts from the exact `bsc-s7777-k10` source and recipe that scored `0.6659107210` on Labless. The existing DINO, JEPA, KDE, FINO, block-strided CLS readout, seed, data split, augmentations, and probe surface remain unchanged.

## Approaches considered

1. **Projected final CLS.** This is closest to the merged prototype, but it is not decoupled: the auxiliary gradient directly supervises the same CLS consumed by tile probes.
2. **Patch-mean routing (selected).** Project the mean final-layer patch representation to the target. Gradients still improve the shared trunk, but the auxiliary does not directly constrain the CLS activation.
3. **Gradient surgery.** Explicitly project conflicting auxiliary gradients away from the base objective. This is the cleanest conflict intervention, but it adds a second backward path and is premature until conflict is measured.

Patch-mean routing is the smallest credible first experiment. A logged gradient cosine on the final block's attention weights will determine whether gradient surgery is justified later.

## Target artifact

`build_molcap_targets.py` reads `metadata/tcga_master_dataset.csv`, aggregates deterministically by `submitter_id`, and writes a non-pickled NPZ containing:

- `patient_ids`: sorted `TCGA-XX-XXXX` Unicode identifiers;
- `targets`: finite, row-normalized `float32` vectors;
- `captions`: the rendered text used for audit;
- `mode`: `text`, `structured`, or `shuffled`.

Captions use columns that actually exist: cancer type, cBioPortal subtype, primary site and diagnosis, AJCC pathologic stage, tumor grade, binned MSI score, binned fraction genome altered, binned mutation count, and explicitly tested positive gene assays. Missing values are omitted with `pandas.notna`; repeated slide rows must agree within a patient.

Modes provide matched controls:

- `text`: encode captions with frozen `sentence-transformers/all-MiniLM-L6-v2`;
- `structured`: encode the same fields as one-hot/numeric features and apply a seeded fixed projection to 384 dimensions;
- `shuffled`: apply a seeded patient permutation to the text targets.

All modes use the same target width and L2 normalization. The first real run uses `text`; structured and shuffled modes are retained for interpretation if the text run is promising.

## Training integration

When `molcap.enabled` is true:

1. `TCGATileDataset` loads the NPZ once for the training split and emits `molcap_target` plus `molcap_present` for each tile's patient.
2. A two-layer `MolCapHead` maps the mean final-layer patch tokens to the configured target width and L2-normalizes the result.
3. The loss is masked cosine distance, repeated in crop-major order across global views.
4. FINO remains enabled and unchanged. MolCap is additive, allowing comparison to the published frontier.
5. The loss weight follows sample progress, not FLOP progress: zero through sample fraction `0.50`, linear ramp through `0.75`, then full weight `0.03`.
6. The head is checkpointed and included in optimizer and gradient clipping. It is absent from probe checkpoints and never read by `probe.py`.

The disabled path must preserve the baseline parameter set, RNG consumption, batch schema, loss value, and optimizer grouping.

## Diagnostics

Training logs must include:

- `molcap` loss and effective scale;
- target coverage in the current batch;
- FINO metadata loss;
- cosine similarity and norm ratio between the base-objective and MolCap gradients on the final block's attention projection, sampled only at normal log steps;
- existing throughput, FLOP, memory, and total-gradient metrics.

Negative gradient cosine is evidence of conflict. A projection head alone is not described as decoupling.

## Tests

Tests must fail before implementation and then cover:

- real-schema patient aggregation and omission of missing values;
- deterministic captions and target banks;
- finite, normalized text/structured/shuffled outputs;
- patient lookup, missing-patient masking, and target dimension checks;
- crop-major target repetition;
- patch-route loss gradients reaching both the MolCap head and shared trunk while not directly depending on CLS output;
- sample-based ramp endpoints;
- checkpoint round-trip;
- disabled-path parity;
- smoke config compilation and a short CPU forward/backward integration test.

## Run sequence and decision rule

1. Build and audit the text target bank; require at least 95% patient coverage against TCGA tile barcodes and non-collapsed target geometry.
2. Run the local test suite and CPU integration smoke.
3. Launch one full `molcap-text-s7777` run on the exact frontier recipe.
4. Compare against `bsc-s7777-k10` (`0.6659107210`) using all eight public metrics.
5. If the score improves by at least `0.006` while linear and kNN each decline by less than `0.003`, repeat with two additional seeds.
6. If slide metrics improve but tile metrics regress, use the logged gradient evidence to choose patch-route weight reduction or gradient surgery.
7. If text does not beat the matched structured target, drop the language claim and retain only the better dense-target geometry.

No smoke or failed run is submitted to Labless. A completed full run is submitted only if its saved source leaves `probe.py` and `benchmarking/` untouched.
