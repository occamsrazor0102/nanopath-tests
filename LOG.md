# JEPA × FINO campaign log

**Goal:** maximize mean_probe_score. Beat the live Labless leader **jepa-mask25 = 0.6471** (I-JEPA contig-patch,
NimaAsh). Hypothesis: FINO metadata-guidance acts on the CLS token and is **orthogonal** to the JEPA patch
objective (DINO CLS + smooth-L1 latent regression + KDE), so it should *stack* — unlike on the DINO+iBOT base
where FINO's best (subtype+expr512) only reached 0.6351 (+0.0074 over 0.6277, below the +0.01 bar).

**Setup:** worktree `nanopath-jepafino` (branch `jepa-fino`), FINO cherry-picked onto `origin/leader`. JEPA
objective untouched; FINO grafted as a 4th loss term `meta` on the CLS token (GradScale DANN gate, EMA prototype
banks for discrete factors, MLP regressors for continuous, lambda_meta=0.03/branch). Smoke (job 81946) validated:
dino+jepa+kde+meta all flow, val + checkpoint(JEPA-predictor + FINO protos/predictors) OK. NOT submitting to labless.

Reference points (live board, same probe suite): leader **0.6471** (jepa family) · curation `lr-and-curation`
0.6357 · `dinov2-s-kde` 0.6277. FINO-on-DINO best: `abl_sub_expr512` 0.6351, `abl_subtype` 0.6323.

## Wave 1 (launched 2026-06-11 ~06:18, 4-wide, pinned n-1/3/4/8)
| id | job | recipe Δ vs JEPA leader (main.yaml) | M+ / M- | final | Δ vs 0.6471 | decision |
|----|-----|------------------------------------|---------|------:|------------:|----------|
| W1-base | 81947 | none — JEPA leader control (harness reproducibility) | — | running | — | — |
| W1-se   | 81948 | + FINO subtype + expr512 | subtype, expr512 M+ | running | — | — |
| W1-s    | 81949 | + FINO subtype | subtype M+ | running | — | — |
| W1-se-rr| 81950 | + FINO subtype + expr512, ramp:run | subtype, expr512 M+ | running | — | — |

Hypotheses: W1-base confirms we reproduce 0.6471 (control for all FINO deltas). W1-se = best DINO-era FINO lever
on JEPA. W1-s isolates the histotype anchor (most robust single factor in the DINO sweep). W1-se-rr tests the
run-keyed DANN ramp (nanopath is sample-capped at ~19% of FLOP budget; flop-keyed gamma stalls at ~0.74).
Results + Wave 2 design pending (~1 hr).

## Literature (scan 2026-06-11, ingested AdvDINO -> geist)
- **Novelty:** FINO M+/M- on a JEPA objective in pathology is novel. Precedents: AdvDINO (2508.04955, DANN on
  slide-ID atop DINOv2 — gate stacks, slide-ARI 0.663->0.037, survival +0.010); JEPA-T (2510.00974, metadata-
  conditioned JEPA predictor, T2I not pathology); GenBio-PathFM (JEPA as stage-2 on frozen DINO). None combine all.
- **JEPA framing:** "Pretext Matters" (2603.22649) — JEAs > JEPAs for spatially-localized signal (tile pathology);
  JEPA earns its keep only as an AUXILIARY on a DINO spine. Our leader recipe IS that shape -> keep DINO-CLS spine.
- **Weighting:** don't import lambda blindly (AdvDINO 50 vs FINO 0.03; magnitudes differ). GRADNORM-MATCH the meta
  branch to the SSL gradient. We log grad_norm -> use it. Wave-2 lever: sweep gamma_max / lambda_meta for JEPA.
- **Entanglement gotcha:** U(cancer|TSS)=1.0 -> solo TSS/scanner-M- reverses cancer signal; suppression must be
  PAIRED with an M+ anchor (subtype/cancer). Pure-suppression alone ~+0.01 (borderline). M+ anchor carries the EV.
- **Mechanism check:** for any M- run, measure slide/TSS-clustering ARI drop ("adversary working" signal) before
  expecting probe gains. **Novel idea (Wave 2/3):** condition JEPAPredictor on cancer-type embedding (JEPA-T style).

## Wave 2 candidate directions (finalize FROM Wave-1 results)
- if jf_sub_expr512 > jepa_base by >+0.01: push M+ stacking (subtype+expr512+morphology / +fga) + gamma_max sweep.
- if FINO ~flat on JEPA: gradnorm-match meta (raise gamma_max), OR try the JEPA-T metadata-conditioned predictor.
- paired cancer-M+/TSS-M- (entanglement-safe suppression) as a separate track once an M+ anchor is confirmed.

## Wave 1 RESULTS (2026-06-11 07:39) — FINO stacks on JEPA
| id | job | final | Δ vs base | decision |
|----|-----|------:|----------:|----------|
| W1-base | jepa_base | 0.6436 | — | control (≈ board I-JEPA-contig 0.6444 ref; mask10 recipe, not the 0.6471 mask25) |
| W1-se | jf_sub_expr512 | **0.6473** | **+0.0037** | KEEP — best; ≈ live leader 0.6471 |
| W1-s | jf_sub | 0.6454 | +0.0018 | subtype alone weaker than +expr512 |
| W1-se-rr | jf_sub_expr512_rr | 0.6453 | +0.0017 | ramp:run no better than flop -> DROP ramp:run |

Components (jf_sub_expr512 vs base): knn +0.022, survival +0.025, fewshot +0.008 (FINO M+ wins) | seg -0.023,
slide -0.006 (CLS steering hurts dense/local). Net +0.0037 (below +0.01 bar). **Seg loss is the drag.** FINO helps
LESS on JEPA (+0.0037) than on DINO+iBOT (+0.0074) — JEPA's patch objective already captures some of what FINO adds.

## Wave 2 (launched 07:39, 4-wide) — map gamma_max + anchor stacking on the best recipe
| id | job | recipe Δ vs jf_sub_expr512 | hypothesis |
|----|-----|---------------------------|-----------|
| W2-g05 | jf_se_g05 | gamma_max 1.0->0.5 | gentler guidance preserves seg/slide, keeps some knn/surv -> better net |
| W2-g15 | jf_se_g15 | gamma_max 1.0->1.5 | stronger (gradnorm-match) -> more M+ gain if seg loss sublinear |
| W2-g20 | jf_se_g20 | gamma_max 1.0->2.0 | upper bound of the gamma curve |
| W2-morph | jf_se_morph | + morphology M+ (gamma 1.0) | 2nd-best DINO anchor; does stacking add on JEPA |

## Wave 2 RESULTS (08:50) — gentler gamma wins (monotonic)
| id | gamma | score | knn | seg | surv | note |
|----|------:|------:|----:|----:|-----:|------|
| W2-g05 | 0.5 | **0.6482** | 0.714 | 0.304 | 0.591 | NEW BEST; +0.0046 vs base, +0.0011 vs leader 0.6471 |
| W2-morph | 1.0 | 0.6477 | 0.723 | 0.298 | 0.579 | +morph: knn up, surv down -> wash |
| (W1-se) | 1.0 | 0.6473 | 0.718 | 0.285 | 0.597 | prior best |
| W2-g15 | 1.5 | 0.6469 | 0.720 | 0.293 | 0.590 | |
| W2-g20 | 2.0 | 0.6462 | 0.718 | 0.288 | 0.583 | too strong; seg loss dominates |
Trend: gamma 0.5>1.0>1.5>2.0 monotonic. Seg loss scales with gamma; gentle gamma keeps M+ knn/surv gains w/o the
seg cost. Optimum at low gamma. NOTE deltas now ~noise floor (DINO-era 3-seed std 0.0005) -> reseed best.

## Wave 3 (launched 08:50, 4-wide) — find gamma floor + confirm best + stack at good gamma
| id | job | recipe | hypothesis |
|----|-----|--------|-----------|
| W3-g03 | jf_se_g03 | subtype+expr512, gamma 0.3 | even gentler — is optimum <0.5 or does it decay to control? |
| W3-g05s2 | jf_se_g05_s2 | gamma 0.5, seed 1337 | RESEED of best — is 0.6482 real vs noise? |
| W3-morph-g05 | jf_morph_g05 | +morphology, gamma 0.5 | best stack at gentle gamma (less seg damage) |
| W3-fga-g05 | jf_se_fga_g05 | +fga (continuous), gamma 0.5 | DINO-era #3 stack at gentle gamma |
