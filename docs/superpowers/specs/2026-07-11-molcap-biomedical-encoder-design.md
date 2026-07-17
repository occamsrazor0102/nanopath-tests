# MolCap Biomedical Encoder A/B Design

## Goal

Determine whether biomedical sentence semantics, rather than the patch-routed MolCap mechanism, caused the MiniLM arm to miss molecular and survival endpoints. The experiment is a paired seed-7777 A/B against `molcap-text-s7777`: reuse its exact 11,428 patient identifiers and caption strings, change only the frozen text encoder and resulting target width, and retain every training and probe choice.

The completed biomedical run is submitted to Labless even if it is null or negative. The draft PR #4 remains untouched as historical context; its guessed-schema caption builder is not used.

## Approaches Considered

1. **Strict re-embedding of canonical captions (selected).** Read `patient_ids` and `captions` from the deterministic MiniLM NPZ, encode those strings with a pinned biomedical sentence encoder, independently fit the identical isotropy procedure, and change the MolCap head width from 384 to 768. This isolates sentence-encoder semantics after comparable geometry normalization.
2. **PR #4 caption builder.** Rejected because it expects nonexistent columns and changes caption text as well as the encoder.
3. **EMA patient-centroid routing.** Deferred because changing pooling and encoder geometry together would make any result unattributable. It becomes the next experiment if the strict encoder A/B does not recover the intended endpoints.

## Frozen Inputs and Model Provenance

- Canonical source artifact SHA-256: `2F6648A4155B96757A136335A253E3FAEB6029A92A7E6356380CE80805011577`.
- Canonical rows: 11,428, sorted by `submitter_id`.
- Generic reference encoder: `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Biomedical encoder: `pritamdeka/S-PubMedBert-MS-MARCO` at revision `96786c7024f95c5aac7f2b9a18086c7b97b23036`.
- The biomedical model is a 768-dimensional sentence-transformer derived from PubMedBERT and fine-tuned on MS-MARCO. It is text-only, ungated, and published under CC-BY-NC-2.0. Its use remains research/noncommercial.

The re-embedding tool must fail if the canonical artifact hash, patient identifiers, caption count, caption order, encoder revision, or expected output width differs.

## Isotropy Isolation

The comparison tests semantic content after geometry normalization. It does not reuse a fitted 384-dimensional MiniLM transform. Instead, each encoder is independently processed by the exact same procedure and constants:

1. L2-normalize raw sentence embeddings.
2. Subtract that encoder's 11,428-vector mean.
3. Eigendecompose that encoder's sample covariance.
4. Apply `max(eigenvalue, max_eigenvalue * 0.05) ** -0.1`.
5. Rotate back and L2-normalize again.

There is no PubMedBERT-specific tuning. If these frozen constants fail the geometry gates, the experiment stops rather than changing them after observing the target.

The existing corrected MiniLM artifact establishes the predeclared reference:

| Metric | MiniLM corrected target |
|---|---:|
| Mean off-diagonal cosine | 0.0008169305 |
| Effective rank | 36.9033 |
| Effective rank / width | 0.0961024 |
| Participation ratio | 22.7452 |
| Participation ratio / width | 0.0592323 |
| Per-dimension variance CV | 0.3174185 |
| Maximum unit-norm error | 9.80e-8 |

The geometry report records, for raw and corrected outputs from both encoders: row count, width, mean off-diagonal cosine, effective rank, normalized effective rank, participation ratio, normalized participation ratio, minimum/median/maximum per-dimension variance, variance coefficient of variation, and maximum unit-norm error.

## Hard Target Gates

The biomedical artifact is eligible for training only if all gates pass:

- exactly 11,428 unique patient identifiers and exact elementwise equality to the canonical identifiers and captions;
- exactly 768 target dimensions, all finite, with maximum row-norm error at most `1e-5`;
- deterministic non-pickled NPZ serialization: two writes from the same computed arrays have identical SHA-256 hashes;
- exact coverage of all 9,389 FINO/tile patients (`100%`); missing-patient count is zero;
- corrected absolute mean off-diagonal cosine at most `0.01`;
- corrected effective rank at least `32` and participation ratio at least `16`;
- corrected per-dimension variance CV at most `0.75`;
- biomedical/reference normalized-effective-rank ratio in `[0.5, 2.0]`;
- biomedical/reference normalized-participation-ratio in `[0.5, 2.0]`.

The tool also regenerates corrected MiniLM embeddings from the pinned revision and requires elementwise agreement with the canonical targets within absolute tolerance `2e-5`. This confirms that caption order and the shared isotropy implementation reproduce the original arm.

## Components and Data Flow

`reembed_molcap_targets.py` is an offline-only helper:

1. Load and validate the canonical MiniLM NPZ.
2. Re-encode the canonical captions with pinned MiniLM and PubMedBERT revisions.
3. Compute raw geometry, independently fit the shared isotropy procedure, and compute corrected geometry.
4. Apply every hard gate before writing the biomedical NPZ.
5. Write a deterministic NPZ plus a JSON report containing provenance, geometry, coverage, thresholds, and hashes.

The helper is excluded from the Labless training source snapshot, like `build_molcap_targets.py` and tests. Training reads only the resulting NPZ through the already-tested target-bank interface.

`configs/molcap-biomed-s7777.yaml` is copied from `configs/molcap-text-s7777.yaml`. Permitted differences are limited to project/output labels, target path, and `molcap.target_dim: 768`. Seed, data split, crops, DINO, JEPA, KDE, FINO, routing, `weight: 0.03`, `ramp_start: 0.5`, `ramp_len: 0.25`, and the locked probe mapping remain identical.

## Tests and Execution

Test-first coverage must prove:

- canonical strings and patient order are copied rather than rendered;
- model names, revisions, widths, and isotropy constants are pinned;
- geometry formulas match hand-computable examples;
- every hard gate accepts a valid fixture and rejects its specific failure mode;
- deterministic metadata-rich NPZ and JSON output;
- the biomedical config differs from MiniLM only on the permitted paths and values;
- 768-dimensional target loading, forward/backward, head checkpointing, and disabled-path parity continue to work;
- development helpers remain outside the Labless source snapshot.

Execution order is tests and compilation, target self-test, full target build, geometry/coverage audit, CPU integration, H100 smoke with gradient diagnostics, then the exact one-million-sample locked seed-7777 run. The full run is submitted to Labless regardless of score if policy checks pass.

## Decision Rule

The primary encoder-semantics endpoint is the mean of molecular AUC and survival c-index. Biomedical semantics are supported only if:

- molecular AUC and survival are each higher than the MiniLM arm;
- their two-metric mean improves by at least `0.003` over MiniLM;
- linear and kNN each decline by less than `0.003` versus MiniLM.

Overall score, slide AUC, segmentation, few-shot, and robustness are reported as secondary endpoints against both `molcap-text-s7777` and `bsc-s7777-k10`. A score at least `0.6719107210` with the tile-metric guard triggers two additional seeds. Failure of the primary endpoint advances the already-approved EMA patient-centroid hypothesis without changing this experiment post hoc.
