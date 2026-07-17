# MolCap Biomedical PCA-384 Retry Design

## Goal

Run one new, width- and head-capacity-controlled MolCap experiment that tests
whether biomedical sentence semantics improve molecular and survival probes.
The experiment reuses the exact canonical captions and every training choice
from `molcap-text-s7777`, but maps the pinned 768-dimensional S-PubMedBERT
embeddings to 384 dimensions before applying the existing isotropy transform.

This is a distinct pre-registration. It does not change, overwrite, or
reinterpret the failed 768-dimensional arm. That arm remains an inconclusive
target-build abort: no training ran, so biomedical semantics were not tested.

## Why the 768-D Arm Was Inconclusive

The failed gate divided effective rank and participation ratio by target
width. Biomedical targets were 768-dimensional while MiniLM targets were
384-dimensional:

```text
effective-rank ratio
= (33.188351 / 768) / (36.903330 / 384)
= (33.188351 / 36.903330) * (384 / 768)
= 0.899 * 0.5
= 0.449666

participation-ratio ratio
= (19.348371 / 768) / (22.745200 / 384)
= (19.348371 / 22.745200) * (384 / 768)
= 0.851 * 0.5
= 0.425329
```

The biomedical target passed every absolute geometry gate. The normalized
ratios failed because the denominator doubled. The proposed retry controls
both target width and MolCap-head capacity at 384.

The projected ratios are not assumed to pass. PCA, row normalization, and the
refitted isotropy transform can change the final geometry. The target build
must measure and gate the result before training.

## Approaches Considered

### Selected: centered PCA-384, row normalization, unchanged isotropy

PCA is deterministic after sign canonicalization, retains the maximum
variance available to a 384-dimensional linear projection, restores width
parity, and keeps the MolCap head shape identical to MiniLM.

### Rejected: seeded semi-orthogonal random projection

A random projection would preserve all source axes without fitting to the
caption bank, but it adds an arbitrary seed and can distort the observed
low-rank manifold more than PCA. It introduces a second unnecessary source of
variation.

### Rejected: retain 768 dimensions and change the ratio gate

Changing the failed arm's gate after observing it would be post-hoc threshold
tuning. It would also leave MolCap head capacity confounded at 768 outputs.

## Frozen Inputs and Provenance

- Canonical target artifact SHA-256:
  `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`.
- Canonical rows: exactly 11,428 patient identifiers and caption strings in
  their existing order.
- MiniLM reference:
  `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Biomedical encoder:
  `pritamdeka/S-PubMedBert-MS-MARCO@96786c7024f95c5aac7f2b9a18086c7b97b23036`.
- FINO patient count: exactly 9,389 with 100% coverage.
- PCA input width: 768; PCA output width: 384.
- Isotropy eigenvalue floor: `0.05` of the maximum eigenvalue.
- Isotropy covariance power: `-0.1`.
- NumPy/PyTorch environment remains pinned by the committed `uv.lock`.

Both encoders are loaded internally from exact validated Hugging Face snapshot
paths. Callers cannot inject an encoder object with self-attested provenance.

## Frozen PCA Procedure

The PCA transform is fitted once on all 11,428 unit-normalized biomedical
sentence embeddings, matching the population used to fit the existing
per-encoder isotropy transform.

1. Encode canonical captions with the pinned biomedical encoder and request
   row-wise L2-normalized float32 embeddings of shape `(11428, 768)`.
2. Convert to float64 and subtract the per-dimension mean.
3. Form the sample covariance with denominator `n - 1`.
4. Use `numpy.linalg.eigh` and order eigenpairs by descending eigenvalue.
5. Retain the first 384 eigenvectors.
6. Canonicalize each component sign: find the lowest-index loading among those
   tied for largest absolute magnitude and require that loading to be positive.
7. Project the centered embeddings through the canonicalized components.
8. Row-wise L2-normalize the `(11428, 384)` projected scores.
9. Apply the existing isotropy function unchanged: mean removal, covariance
   eigendecomposition, `max(eigenvalue, max_eigenvalue * 0.05) ** -0.1`, rotate
   back, and final row-wise L2 normalization.

Sign canonicalization does not mathematically resolve arbitrary rotations
inside exactly degenerate eigenspaces. Therefore the implementation also pins
the numerical environment and requires two independent builds in the same
environment to produce identical target and report hashes.

## PCA Audit and Gate

The report records:

- fit rows, input/output widths, centering and covariance denominator;
- solver, ordering, and sign convention;
- all 768 eigenvalues in descending order and their deterministic hash;
- retained-384 component hash and centered-mean hash;
- total, retained, and discarded variance;
- retained explained-variance fraction;
- discarded Frobenius-energy fraction;
- eigenvalues at components 384 and 385 and their eigengap;
- geometry before PCA, after PCA normalization, and after isotropy.

The PCA target is training-eligible only if:

- retained explained variance is at least `0.99`;
- the projected matrix is exactly `(11428, 384)` and finite;
- every projected row is nonzero before normalization;
- maximum post-projection row-norm error is at most `1e-5`;
- component, mean, and audit hashes are deterministic across the independent
  rebuild.

The prior aggregate effective-rank values do not prove 99% retention. This
gate establishes or rejects the negligible-loss claim directly.

## Existing Target Gates Remain Frozen

After PCA normalization and isotropy, all existing gates apply unchanged:

- exactly 11,428 unique patient identifiers and exact caption/order equality;
- exact 384 target dimensions, all finite, unit-norm error at most `1e-5`;
- strict non-pickled deterministic NPZ serialization;
- pinned MiniLM replay agrees with the canonical targets at `atol=2e-5`,
  `rtol=0`;
- exact 9,389/9,389 FINO coverage and no missing patients;
- absolute mean off-diagonal cosine at most `0.01`;
- effective rank at least `32`;
- participation ratio at least `16`;
- per-dimension variance CV at most `0.75`;
- biomedical/reference normalized-effective-rank ratio in `[0.5, 2.0]`;
- biomedical/reference normalized-participation-ratio in `[0.5, 2.0]`.

Any failure writes a strict-JSON `status=failed` report, clears staging and any
stale PCA-384 target at the fixed path, and stops before training. Thresholds,
component count, or ordering are not revised after observing the target.

## Artifact and Configuration Isolation

The prior files remain immutable:

- `/data/repo-data/molcap_biomed_768.geometry.json`;
- the absent `/data/repo-data/molcap_biomed_768.npz`;
- `configs/molcap-biomed-s7777.yaml`;
- `docs/results/2026-07-12-molcap-biomed-s7777.md`.

The retry uses distinct names:

- target: `/data/repo-data/molcap_biomed_pca384.npz`;
- report: `/data/repo-data/molcap_biomed_pca384.geometry.json`;
- config: `configs/molcap-biomed-pca384-s7777.yaml`;
- smoke: `smoke/molcap-biomed-pca384-s7777`;
- calibration: `calibration/molcap-biomed-pca384-s7777/<gpu>`;
- full: `full/molcap-biomed-pca384-s7777`.

The new config is copied from `configs/molcap-text-s7777.yaml`. Its semantic
diff is exactly:

- `project.name`;
- `project.output_dir`;
- `molcap.targets`.

`molcap.target_dim` remains 384. Seed 7777, split, batch size, crops, views,
DINO, JEPA, KDE, FINO, MolCap weight `0.03`, ramp `[0.50, 0.75]`, and every
probe mapping remain identical. No change is permitted in `model.py`,
`dataloader.py`, `train.py`, `probe.py`, or `benchmarking/`.

## Hardware-Efficient Execution

Public submissions have no wall-clock limit, and preprocessing before training
is excluded from training time. Hardware is therefore an execution choice,
not an experimental hyperparameter.

The preferred public-run accelerator is one Modal B200. The existing stack is
PyTorch 2.8 with CUDA 12.9; Modal supports B200 directly. B300 and `B200+` are
excluded because B300 currently requires CUDA 13.1 or later. If B200 fails a
compatibility smoke for infrastructure reasons, the fallback order is H200,
then exact H100. The algorithm and configuration do not change. Modal's GPU
contract is documented at <https://modal.com/docs/guide/gpu>.

Multi-GPU execution is rejected. The current training loop is single-GPU, and
adding distributed training would change implementation risk and potentially
batch semantics solely to save wall time.

### Existing two-hour evidence

The completed `molcap-text-s7777` run used the same 384-dimensional MolCap
head, batch 128, views, sample budget, and training path:

- H100 train-loop time: `3594.60417402` seconds = 59.91 minutes;
- final locked probe: `1727.07658274` seconds = 28.78 minutes;
- complete Modal function: `5439.499483931` seconds = 90.66 minutes.

Training already has 60.09 minutes of margin under the maintainer's two-hour
limit. Even the complete training-plus-probe function has 29.34 minutes of
margin. The historical evidence is retained in
`/data/experiments/readout-local-context/full/molcap-text-s7777/summary.json`
and `modal_result.json`.

### Current hardware calibration

After the target passes, run one representative 32,768-sample calibration on
each of B200 and exact `H100!`:

- full batch size 128, precision, crops, two global views, eight local views,
  optimizer, losses, routing, and MolCap ramp fractions;
- probes disabled and no checkpoint during the short calibration;
- identical source commit and PCA target;
- exclude throughput log entries before step 60 and use the median logged
  FLOP/s from step 60 onward;
- record GPU name, CUDA/PyTorch versions, train-loop seconds, visible patches/s,
  FLOP/s, memory peak, and target coverage.

Let `F_B200` and `F_H100` be the median steady-state calibration throughputs.
After the B200 full run, compute:

```text
projected_H100_train_seconds
= observed_B200_full_train_seconds * (F_B200 / F_H100)
```

Report the projected H100 training time, its margin to 7,200 seconds, and the
historical 3,594.6-second cross-check. A projection over two hours does not
invalidate or suppress a completed public run; it only lowers confidence that
maintainer validation will meet its separate time criterion.

If B200 is unavailable or fails compatibility, no B200/H100 ratio is invented.
Run the declared H200 or H100 fallback, report its observed training time, and
use the current exact-H100 calibration plus the historical full H100 run as the
maintainer-time evidence.

## Execution Order

1. Unit tests and compilation.
2. Deterministic PCA self-test on synthetic matrices.
3. Full target build on the canonical captions.
4. Independent target/report download and audit.
5. CPU integration through the real target-bank/head/loss/checkpoint path.
6. B200 compatibility and gradient-diagnostic smoke.
7. B200 and exact-H100 32,768-sample throughput calibration.
8. Fresh local verification and clean-source check.
9. One-million-sample locked seed-7777 full run on B200, or the declared
   infrastructure fallback.
10. Complete locked final probe.
11. Compare against MiniLM and `bsc-s7777-k10` using the predeclared rules.
12. Dry-run and submit the completed full run to Labless regardless of score.
13. Record target hashes, PCA audit, hardware calibration, timing projection,
   full metrics, submission ID, and decision in a durable result report.

Preprocessing and target construction are never included in reported training
time. Smoke and calibration runs are never submitted.

## Decision Rule

Biomedical semantics are supported only if all are true versus
`molcap-text-s7777`:

- molecular AUC is higher;
- survival mean c-index is higher;
- their two-metric mean improves by at least `0.003`;
- linear mean F1 declines by less than `0.003`;
- kNN mean F1 declines by less than `0.003`.

Overall, few-shot, slide, segmentation, and robustness are secondary endpoints
reported against both MiniLM and `bsc-s7777-k10`.

A score of at least `0.6719107210`, with both tile guards satisfied, triggers
maintainer-style validation interest and two additional local seeds only after
the seed-7777 public submission is complete. A completed full run is submitted
even when null or negative.

If the target aborts or the completed run fails the primary endpoint, no second
width-control retry is attempted. The next experiment is the separately
pre-registered EMA patient-centroid hypothesis.
