# MolCap biomedical-encoder A/B (strict, paired seed-7777)

Companion to `MOLCAP_PROPOSAL.md §8`. Tests whether **biomedical sentence semantics** — not the patch-routed MolCap mechanism — caused the MiniLM arm to miss molecular/survival. Paired against `molcap-text-s7777`: **reuse the exact 11,428 captions + patient ids, change only the frozen encoder and target width, hold everything else.** The completed run is submitted to Labless regardless of outcome. PR #4's guessed-schema builder is *not* used and stays as history.

## Approach (selected: strict re-embedding of canonical captions)
Read `patient_ids` + `captions` from the deterministic MiniLM NPZ, re-encode those exact strings with a pinned biomedical sentence encoder, independently re-fit the identical isotropy procedure, set the head width 384→768. Rejected alternatives: the PR #4 builder (re-renders captions → confounds text with encoder; wrong schema) and jumping to EMA patient-centroids (confounds pooling with encoder geometry — that's the *next* experiment if this one fails the primary endpoint).

## Frozen provenance (in `reembed_molcap_targets.py`; the tool refuses to run on drift)
- Canonical NPZ SHA-256 `2f6648…011577`; 11,428 rows sorted by `submitter_id`.
- Reference encoder `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243…` (384-d).
- Biomedical encoder `pritamdeka/S-PubMedBert-MS-MARCO` @ `96786c70…` (768-d) — PubMedBERT fine-tuned on MS-MARCO; text-only, ungated, CC-BY-NC-2.0 (research/noncommercial). A *literature* text encoder is public non-image info, not a pathology image model — worth a one-line maintainer confirm since it's domain-adjacent.

## Isotropy isolation (semantic content after comparable geometry)
Not a reused 384-d transform — each encoder gets the **same procedure and constants**: L2-normalize → subtract that encoder's mean → eigendecompose that encoder's covariance → scale by `max(λ, λ_max·0.05)^-0.1` → rotate back → L2-normalize. Constants are frozen; if they fail the gates the experiment stops rather than retuning post-hoc. MiniLM-corrected reference (predeclared): mean off-diag cosine 0.000817, effective rank 36.90 (÷width 0.0961), participation 22.75 (÷width 0.0592), per-dim var CV 0.317.

## Hard gates (all must pass before the artifact is training-eligible)
Exactly 11,428 unique ids with elementwise equality to canonical ids **and** captions · width 768, all finite, max row-norm error ≤1e-5 · deterministic non-pickled NPZ (two writes → identical SHA-256) · 100% coverage of the 9,389 tile patients · corrected |mean off-diag cosine| ≤0.01 · corrected effective rank ≥32 and participation ≥16 · corrected per-dim var CV ≤0.75 · biomedical/reference normalized-effective-rank and normalized-participation ratios both in [0.5, 2.0]. Plus a **MiniLM reproduction check**: re-encoding MiniLM at the pinned revision + the shared isotropy must match the canonical targets within 2e-5 (confirms caption order + isotropy implementation reproduce the original arm).

## Permitted config diff (`molcap-biomed-s7777.yaml` vs `molcap-text-s7777.yaml`)
Only: project/output labels, the `molcap.targets` path, and `molcap.target_dim: 768`. **Held identical:** seed 7777, data split, crops, DINO, JEPA, KDE, FINO, routing, `molcap.weight: 0.03`, `molcap.ramp_start: 0.5`, `molcap.ramp_len: 0.25`, and the locked probe mapping. (A test asserting "differs only on permitted paths" runs in the fork, where both configs live.)

## Decision rule (pre-registered)
Primary endpoint = mean(molecular AUC, survival c-index). Biomedical semantics are **supported** only if: molecular AUC *and* survival each exceed the MiniLM arm; their two-metric mean improves by ≥0.003; and linear *and* kNN each decline by <0.003. Secondary (overall, slide AUC, seg, few-shot, robustness) reported vs both `molcap-text-s7777` and `bsc-s7777-k10`. Overall ≥0.6719107210 with the tile-metric guard → run two more seeds. Primary failure → advance the already-approved EMA patient-centroid hypothesis (no post-hoc changes here).

## What runs where
- **Validated in this repo (CPU, no models):** the geometry metrics (effective rank / participation / off-diag cosine on hand values), the isotropy transform (unit-norm + decorrelation + determinism), deterministic NPZ hashing, and all nine gates' accept/reject logic — `tests/test_reembed_molcap_targets.py`, 6 tests green.
- **Runs in the fork (GPU + data):** `python reembed_molcap_targets.py canonical=… biomed_out=… report_out=…` (needs the canonical NPZ + both pinned encoders + the tile-patient list), then tests → CPU integration → H100 smoke with gradient diagnostics → the locked 1,000,000-sample seed-7777 probe → submit to Labless if policy passes.

`reembed_molcap_targets.py` and `tests/` are development helpers — exclude them from the Labless training source snapshot (like `build_molcap_targets.py`).
