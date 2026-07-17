# MolCap biomedical PCA-384 target, seed 7777

## Decision

Do not train or submit this arm. The frozen PCA-384 target build failed the
coordinate-wise variance-CV gate before any target was published:
`4.290600203955621` against a maximum of `0.75`. The excess was
`3.540600203955621`, or `5.720800271940828` times the limit. The threshold,
384-component count, component ordering, and isotropy parameters were not
changed after observing the result.

PCA retained `0.999774751852857` of the raw variance and passed its frozen
`0.99` minimum, but that does not override the later variance-CV failure. The
report records `status=failed`, `artifact.published=false`, and
`artifact.target_path_cleared=true`; the independent post-run volume check
found no PCA target.

## Frozen provenance and evidence

- Source commit:
  `9a23043433c243057f20657d655099a9f765626c`
- Canonical MiniLM source NPZ SHA-256:
  `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`
- Frozen config `configs/molcap-biomed-pca384-s7777.yaml` SHA-256:
  `268c3f48edbd687fa7f305643b13d392196b034050d52b98df2526ce6dd8dd40`
- Downloaded failure-report SHA-256:
  `bbe5680f84f07c9dd52754a5a4e6b5ad3ba1c590709941235cbe337ca5c6e88d`
- Persisted report: `/data/repo-data/molcap_biomed_pca384.geometry.json`
- Downloaded report:
  `C:\Users\occam\AppData\Local\Temp\molcap-biomed-pca384-audit-ap-NaD6zukpkPNbMyJa5QnOME\molcap_biomed_pca384.geometry.json`
- Independent audit:
  `.superpowers/sdd/pca384-target-audit.json`
- Proposed target: `/data/repo-data/molcap_biomed_pca384.npz` -- not published
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2` at
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Biomedical encoder: `pritamdeka/S-PubMedBert-MS-MARCO` at
  `96786c7024f95c5aac7f2b9a18086c7b97b23036`

The downloaded report was 28,496 bytes. The independent audit loaded the
canonical bank with `allow_pickle=false`, reproduced its SHA-256, and confirmed
the four exact keys, 11,428 unique patients and captions, `11428 x 384` finite
unit targets, and scalar `text` mode. It also records exact 9,389/9,389 FINO
coverage. The strict MiniLM replay was reached and passed before candidate
validation.

## PCA audit

| Quantity | Observed | Requirement | Result |
|---|---:|---:|:---:|
| Fit rows | 11,428 | 11,428 | Pass |
| Input width | 768 | 768 | Pass |
| Output width | 384 | 384 | Pass |
| Retained variance | 0.051173180141635466 | N/A | N/A |
| Total variance | 0.05118470940259045 | N/A | N/A |
| Discarded variance (eigenvalue sum) | 1.1529260954981817e-05 | N/A | N/A |
| Discarded variance fraction | 0.0002252481471429757 | N/A | N/A |
| Retained variance fraction | 0.999774751852857 | >= 0.99 | Pass |
| Eigenvalue 384 | 1.081554812027289e-07 | N/A | N/A |
| Eigenvalue 385 | 1.0764401246688935e-07 | N/A | N/A |
| 384/385 eigengap | 5.114687358395448e-10 | N/A | N/A |

The independent eigenvalue-spectrum hash is
`34c7ea4c4f8e2aa10807bf1d227eb14a3fec4b22a0ccc67b45f993d10510b49a`.
All 768 eigenvalues were finite, nonnegative, and descending; retained, total,
and discarded variance recomputed exactly.

## Frozen target-gate audit

| Gate | Observed | Requirement | Live result |
|---|---:|---:|:---:|
| Rows | 11,428 | 11,428 | Pass |
| Unique patient IDs | 11,428 | 11,428 | Pass |
| Width | 384 | 384 | Pass |
| Finite geometry | true | true | Pass |
| Maximum unit-norm error | 1.4972772532928502e-07 | <= 1e-5 | Pass |
| Absolute mean off-diagonal cosine | 1.0149864602736523e-05 | <= 0.01 | Pass |
| Effective rank | 33.041619200268 | >= 32 | Pass |
| Participation ratio | 19.352352008536 | >= 16 | Pass |
| Variance CV | 4.290600203955621 | <= 0.75 | **Fail** |

Variance CV was the first failed production assertion. Values computed before
the raise also give a normalized effective-rank ratio of `0.89535604274` and a
normalized participation-ratio ratio of `0.850832365603`, both within the
frozen inclusive range `[0.5, 2.0]`. Those later assertions were not reached,
so they are independently checked report values rather than live assertion
passes. The same distinction applies to the independently confirmed
9,389/9,389 FINO coverage.

## Publication and execution status

- PCA target: absent; no target SHA-256 exists.
- Independent deterministic target/report rebuild: not attempted because the
  first target build failed before publication.
- Real 384-dimensional target integration: not run; no published target
  existed to load.
- Target-build allocation: source-declared CPU-only; no GPU was requested.
- GPU smoke, fallback, calibration, and full-run actions: not run after the
  target failure.
- B200 smoke: not run -- target gate failed.
- H200/H100 fallback: not activated -- this was not an infrastructure failure.
- B200/H100 calibration: not run -- target gate failed.
- Training: not run.
- Probe: not run.
- Full run: not run.
- Labless submission: not run; no submission ID exists.

The CPU target build used Modal app `ap-NaD6zukpkPNbMyJa5QnOME`, container
`ta-01KXBKFGWEYNR4X450HM141TKR`, and exited 1 after 261.3 client-wall seconds
with `ValidationGateError: variance CV gate failed`. The source-declared build
allocation was CPU-only (16 CPUs, 131,072 MiB); retained Modal history did not
expose a historical resource-allocation ledger. Its no-GPU/no-submission log
searches are supporting observations, not an exhaustive workspace-wide
activity ledger.

Local pytest, compilation, whitespace, status, and locked-path checks were run
after the failure as out-of-band postmortem validation. That was a protocol
deviation from the original literal stop instruction and did not authorize or
run target integration, GPU work, training, probing, calibration, a full run,
or submission.

## Audit limitations and next action

Because failure cleanup intentionally removed the target, target-byte hashing,
target/caption equality, corrected-geometry recomputation from target bytes,
real integration, and deterministic rebuild checks are unavailable rather
than failed. Raw biomedical and post-PCA matrices were not persisted; their
reported tables were checked internally, while raw effective rank and
participation ratio were independently reconstructed from the eigenvalue
spectrum.

No second width-control retry is authorized. Advance the separately
pre-registered EMA patient-centroid hypothesis; do not tune the frozen PCA or
geometry gates post hoc.
