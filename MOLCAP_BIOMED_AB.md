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

## Result (2026-07-12): aborted at the width-confounded ratio gate — encoder hypothesis UNTESTED

The S-PubMedBERT build passed every absolute gate (identity, finiteness, unit norm, off-diag cosine 0.00068, effective rank 33.19 ≥ 32, participation 19.35 ≥ 16, var CV 0.46, FINO coverage 9,389/9,389) but failed the **normalized** ratio gates and correctly published nothing:

| gate | biomedical | MiniLM ref | ratio | bound |
|---|---:|---:|---:|:---:|
| norm effective-rank | 0.043214 | 0.096102 | **0.4497** | [0.5, 2.0] ✗ |
| norm participation | 0.025193 | 0.059232 | **0.4253** | [0.5, 2.0] ✗ |

**Diagnosis — the failure is width-confounded, not degenerate semantics.** `norm_effrank = effective_rank / width`, and biomedical width is 768 vs MiniLM 384. So `ratio = (eff_bio/eff_mini) × (384/768) = 0.899 × 0.5 = 0.450`. In **absolute** terms the biomedical corrected geometry (eff rank 33.2, participation 19.3) is **85–90%** of MiniLM's — well inside [0.5, 2.0]. The gate fired essentially because the encoder is 2× wider, not because its semantic manifold is collapsed. **Therefore the biomedical-semantics hypothesis is untested, not falsified** — no training ran.

The disciplined response is a **new, width-controlled pre-registration** (not a post-hoc change to this frozen gate): PCA-reduce the biomedical embedding 768→384 *before* isotropy, so (a) width parity makes the normalized-ratio gate meaningful, (b) the MolCap head shape (→384) is held constant with the MiniLM arm — making it a truer encoder-only A/B. Reducing 768→384 loses negligible signal (corrected effective rank is only ~33). Alternatively the *next* experiment is the already-approved EMA patient-centroid variant.

**Failure-path hardening (this PR).** Mirrors the fork's post-run review: the artifact is written to a staging path and **published only on a full pass**; on failure staging is removed and any stale target at the fixed path is cleared (`publish_or_clear`), a `status=failed`/`published=false` report with the full gate audit is still written, and non-finite audit values serialize as strict-JSON `null` with recorded field paths (`json_safe`). Covered by `test_publish_on_pass_and_clear_on_fail`, `test_json_safe_*`, and `test_gates_reproduce_biomed_width_confounded_failure`.

## Pre-registration 2: width-controlled biomedical A/B (the actual encoder test)

The strict 768 arm above is aborted and **frozen** as history — this is a **new, separately pre-registered** experiment, not a post-hoc edit to it. One declared change vs. strict: **PCA-reduce the biomedical embedding 768→384 before the identical isotropy** (`reembed_molcap_targets.py target_width=384`). Rationale:
1. **Width parity** makes the normalized-ratio gate apples-to-apples — the strict arm failed it only because `norm_rank = rank/width` halves at 2× width (predicted ratio ≈ 0.90 → passes; verified in `test_gates_width_matched_pass`).
2. **Head-shape parity** — a 384-d target holds the `MolCapHead` output width constant with the MiniLM arm, so this is a *truer* encoder-only A/B (the strict arm also varied head capacity, a second confound).
3. **Negligible loss** — the biomedical corrected effective rank is only ~33, so the top 384 principal dims retain essentially all signal (the run records `pca_variance_retained`; expect ≈1.0).

Everything else is unchanged and still frozen: same captions/ids, same isotropy procedure + constants, all nine gates (with the width gate now checking 384), the MiniLM reproduction precondition, and the failure-path hygiene. The fork config becomes `molcap-biomed384-s7777.yaml` with `molcap.target_dim: 384`; still holds seed 7777, weight 0.03, ramp, routing, FINO, and the locked probe mapping identical to the MiniLM arm.

**Decision rule is unchanged** (§Decision rule): does mean(molecular AUC, survival c-index) rise ≥0.003 while linear/kNN each decline <0.003, judged paired against `molcap-text-s7777`. This is the run that actually answers the encoder-semantics question the strict arm left inconclusive. If it too fails the primary endpoint, advance the EMA patient-centroid hypothesis.

Run in the fork: `python reembed_molcap_targets.py canonical=… biomed_out=molcap_biomed384.npz report_out=… target_width=384` → confirm `status=passed`, `published=true`, `pca_variance_retained≈1.0` → tests → CPU integration → H100 smoke → locked seed-7777 probe → submit regardless of outcome.
