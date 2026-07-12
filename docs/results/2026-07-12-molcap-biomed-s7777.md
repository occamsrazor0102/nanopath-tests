# MolCap biomedical target, seed 7777

## Decision

Do not train or submit this arm. The frozen S-PubMedBERT target build failed
the normalized effective-rank ratio gate before an NPZ was published:
`0.449666078410`, below the inclusive lower bound `0.5` by
`0.050333921590`. The normalized participation-ratio comparison also fails at
`0.425328676587`, below the same lower bound by `0.074671323413`.

All absolute geometry, identity, finiteness, norm, and FINO coverage gates
passed. The failure is specifically the predeclared relative-geometry guard;
its thresholds and frozen isotropy parameters were not changed. No smoke,
full training, probe, score comparison, or Labless submission was run.

## Target provenance and artifacts

- Diagnostic source commit: `b531d0c80e8b72c46d0c500ec8a2d8011c2d349d`
- Pre-diagnostic experiment source: `b706a581dc971cc7c973de205791b21ad20f6f66`
- Canonical MiniLM NPZ SHA-256:
  `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`
- FINO metadata SHA-256:
  `45b49a11891c6889f2f240de5ffaecd5ded57f2146c151c73cb43ca3872a5d55`
- Failed geometry-report SHA-256:
  `204ad37ca1644c15259e56e08b95fedbd86a063f8da401f1c551e0e4426e7817`
- Persisted report:
  `/data/repo-data/molcap_biomed_768.geometry.json`
- Proposed target: `/data/repo-data/molcap_biomed_768.npz` — **not published**
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2` at
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Biomedical: `pritamdeka/S-PubMedBert-MS-MARCO` at
  `96786c7024f95c5aac7f2b9a18086c7b97b23036`
- Isotropy: floor `0.05`, power `0.1`, independently fitted per encoder

The retrieved report has `status=failed`, names `effective rank ratio` as the
first failed gate, and records `artifact.published=false`. A direct volume
listing returned one geometry report and zero biomedical NPZs.

## Raw and corrected geometry

| Metric | MiniLM raw | MiniLM corrected | Biomedical raw | Biomedical corrected |
|---|---:|---:|---:|---:|
| Rows | 11,428 | 11,428 | 11,428 | 11,428 |
| Width | 384 | 384 | 768 | 768 |
| Mean off-diagonal cosine | 0.674885362348 | 0.000816930455 | 0.948815334848 | 0.000679637555 |
| Effective rank | 31.306104129830 | 36.903329651010 | 28.491657307591 | 33.188351048901 |
| Normalized effective rank | 0.081526312838 | 0.096102420966 | 0.037098512119 | 0.043213998762 |
| Participation ratio | 19.159624833565 | 22.745199631436 | 16.405428610810 | 19.348371315879 |
| Normalized participation ratio | 0.049894856337 | 0.059232290707 | 0.021361235170 | 0.025193191818 |
| Variance minimum | 1.98437409846e-66 | 2.62591898772e-35 | 1.53577029039e-05 | 0.000338385951 |
| Variance median | 0.000809614464 | 0.002518697665 | 0.000058673738 | 0.001160275821 |
| Variance maximum | 0.002184759932 | 0.006192894998 | 0.000234398630 | 0.004321736660 |
| Variance CV | 0.342424931894 | 0.317418497005 | 0.484119851678 | 0.464038012167 |
| Maximum unit-norm error | 1.46691096825e-07 | 9.79518561817e-08 | 2.92692890591e-07 | 1.24818715475e-07 |

## Frozen gate audit

| Gate | Observed | Requirement | Result |
|---|---:|---:|:---:|
| Rows | 11,428 | 11,428 | Pass |
| Unique patient IDs | 11,428 | 11,428 | Pass |
| Width | 768 | 768 | Pass |
| Finite geometry | true | true | Pass |
| Maximum unit-norm error | 1.24818715475e-07 | <= 1e-05 | Pass |
| Absolute mean off-diagonal cosine | 0.000679637555 | <= 0.01 | Pass |
| Effective rank | 33.188351048901 | >= 32 | Pass |
| Participation ratio | 19.348371315879 | >= 16 | Pass |
| Variance CV | 0.464038012167 | <= 0.75 | Pass |
| Normalized effective-rank ratio | 0.449666078410 | 0.5 to 2.0 | **Fail** |
| Normalized participation-ratio ratio | 0.425328676587 | 0.5 to 2.0 | **Fail** |
| FINO patient count | 9,389 | 9,389 | Pass |
| FINO coverage | 9,389 / 9,389 (1.0) | 1.0 | Pass |

The helper reports the first failure in frozen assertion order, so the named
error is effective-rank ratio. Independent arithmetic from the corrected
geometry reproduced both ratios exactly. Independent loading of the canonical
bank confirmed the four expected non-pickled keys, 11,428 patient IDs and
captions, 11,428 x 384 finite targets, scalar `text` mode, and 11,428 unique
patients. Recomputed canonical geometry matched the report with maximum
absolute error `0`.

## Execution metadata

- Modal app: `ap-ifp5FXOViI2GIsY2opxgMm`
- Modal container: `ta-01KXANJZBGSAYETR8NKB44QN2R`
- Allocation: exact `H100!`, 16 CPUs, 131,072 MiB memory
- Timeout: four hours
- Encoder replay device: CPU
- W&B mode: offline (no training run was initialized)
- Rerun wall time: 210.2 seconds
- App interval: 2026-07-12 04:02:28-04:00 to 04:05:56-04:00

The strict MiniLM replay (`atol=2e-5`, `rtol=0`) completed before geometry
validation. The report-only diagnostic commit changes the ignored preprocessing
helper and its tests; it does not modify `probe.py`, `benchmarking/`, or the
training path.

## Predeclared experiment comparisons

The biomedical arm has no checkpoint or probe metrics, so values and deltas
are intentionally `N/A` rather than inferred from a failed target build.

| Category | MiniLM | Frontier | Biomedical | Delta vs MiniLM | Delta vs frontier |
|---|---:|---:|---:|---:|---:|
| Overall | 0.665223919386 | 0.665910721008 | N/A | N/A | N/A |
| Linear | 0.807588787549 | 0.807644278100 | N/A | N/A | N/A |
| kNN | 0.752106627086 | 0.751035390218 | N/A | N/A | N/A |
| Few-shot | 0.689874342935 | 0.690418943516 | N/A | N/A | N/A |
| Slide | 0.670255183413 | 0.667117077205 | N/A | N/A | N/A |
| Segmentation | 0.331968268739 | 0.334109633151 | N/A | N/A | N/A |
| Molecular | 0.611648962395 | 0.613979651293 | N/A | N/A | N/A |
| Survival | 0.578469383594 | 0.582511896731 | N/A | N/A | N/A |
| Robustness | 0.879879799378 | 0.880468897851 | N/A | N/A | N/A |

The predeclared MiniLM molecular/survival mean is `0.595059172995`; the
frontier mean is `0.598245774012`. No biomedical primary endpoint, tile guard,
or `0.6719107210` promotion test exists. Additional seeds are therefore not
authorized.

## Next action

Advance the already-approved EMA patient-centroid hypothesis. Do not tune the
frozen isotropy or geometry gates post hoc, and do not submit this untrained
encoder arm to Labless.
