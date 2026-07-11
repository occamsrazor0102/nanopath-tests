# nanopath score-improvement plan (OccamsTrustyRazor)

_Generated 2026-07-11 from a full crawl of the Labless ledger (611 runs) + your 3 runs + the clean MedARC `main` recipe. `mean_probe_score` = mean of {linear, knn, 16-shot, segmentation, progression, mutation, survival, robustness}._

## TL;DR

You are **already at the frontier and ~0.0003 from the validation bar.** Your best run scores **0.6649**; the public leader is **0.6592**; the bar to *claim* the leaderboard is leader **+0.006 = 0.6652**. The single board-topping run (Jeremy's `bsc-s7777-k10`, **0.6659**) uses **byte-for-byte the same model code as yours** — it beats you on three config knobs. Close that gap, then stack two orthogonal levers to build validation margin.

**Do first:** raise `kde_loss_weight` 0.005 → **0.05** and train the **full 1,000,000** tiles (drop your 850k early-stop). Expected ≈ **0.666**, clearing the bar. Then harden with curation + more local views + multi-seed to push the *median* to **≥0.668** so it survives the maintainer's different-seed rerun.

---

## 1. Where you stand

| Your run | score | state | linear | knn | 16-shot | seg | **prog** | mut | surv | **robu** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lctx10x128-s7777` | **0.6649** | unvalidated | 0.817 | 0.752 | 0.676 | 0.334 | 0.647 | 0.617 | 0.576 | **0.900** |
| `localctx10x128` | 0.6645 | unvalidated | 0.819 | 0.751 | 0.680 | 0.328 | 0.645 | 0.619 | 0.575 | 0.898 |
| `lc10x128-s271828` | 0.6597 | unvalidated | 0.809 | 0.752 | 0.672 | 0.337 | 0.631 | 0.615 | 0.568 | 0.893 |

Reference points:

| Run / baseline | score | prog | knn | 16-shot | robu | note |
|---|---:|---:|---:|---:|---:|---|
| **Bar to claim leader** | **0.6652** | | | | | 0.6592 + 0.006 |
| `bsc-s7777-k10` (JeremyKalfus) | **0.6659** | 0.667 | 0.751 | 0.690 | 0.880 | best on board, same code as you |
| `curation-unbalanced` (PaulScotti) | 0.6653 | 0.659 | 0.743 | 0.685 | 0.901 | curation lever, kde 0.005 |
| `block-strided-cls` (RyanKim17920) | 0.6592 | 0.630 | 0.747 | 0.681 | 0.892 | current **validated leader** |
| `lr-and-curation` (nevasini1) | 0.6357 | 0.649 | 0.700 | 0.612 | 0.861 | current **main** |
| _GenBio-PathFM (ViT-G baseline)_ | _0.6917_ | _0.768_ | _0.763_ | _0.697_ | _0.941_ | ceiling reference (not attainable init) |

**Two facts that define the strategy:**
1. Your recipe is essentially the frontier recipe. Your `model.py`/`train.py`/`dataloader.py` diff is identical to Jeremy's — you both run **DINOv2-S/14 + JEPA + FINO(subtype+expr512+fga) + the (4,6,8,11) multi-block CLS readout + 10 local views**. This is a solved, high-performing stack; you don't need a new architecture.
2. **Progression is your biggest gap and biggest opportunity.** You sit at 0.647; the frontier reaches 0.667–0.699; the ViT-G baseline hits 0.768. Progression is ⅛ of the score, so 0.647→0.68 alone is **+0.004 overall** — more than the entire validation margin.

---

## 2. The exact gap to the top run

Your `lctx10x128-s7777` (0.6649) vs Jeremy's `bsc-s7777-k10` (0.6659) — identical model code, three config deltas:

| knob | you | Jeremy | effect |
|---|---|---|---|
| `dino.kde_loss_weight` | **0.005** | **0.05** (10×) | lifts prog/mutation/survival via CLS-sphere uniformity |
| `train.local_size` | 128 | 112 | minor; 112 is the tested frontier value |
| training length | **850k** (`target_train_samples`) | **full 1M** | you leave ~15% of the tile budget unused |

Your compensating asset: **robustness 0.900 vs his 0.880.** Robustness and KDE trade off (more uniformity pressure → lower pathorob), so expect your robustness to ease toward ~0.89 when you raise KDE — the net is still up.

---

## 3. What the 611-run ledger proves

**KDE weight has a clear optimum at 0.05** (best overall score per weight; higher collapses knn/16-shot):

| `kde_loss_weight` | best score | prog | knn | 16-shot | robu |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.6518 | 0.671 | 0.709 | 0.641 | 0.898 |
| 0.005 _(yours)_ | 0.6653 | 0.659 | 0.743 | 0.685 | 0.901 |
| **0.05** | **0.6659** | 0.667 | **0.751** | **0.690** | 0.880 |
| 0.35 | 0.6634 | 0.653 | 0.734 | 0.689 | 0.885 |
| 0.7 | 0.5977 | 0.608 | 0.565 | 0.597 | 0.856 |

- **ViT-B is a dead end.** ~7 runs at 86M params were tried; the best (`dinov2b-fino`) reached only **0.6462**, far below the 0.665 ViT-S frontier. 1M tiles under the caps underfits a from-bigger model; the pretrained DINOv2-**S** init wins. **Do not scale the backbone.**
- **FLOP budget is ~80% unused.** Top runs use only ~20% of the 1e18 FLOP cap because the 1,000,000-tile cap binds first. Wall-clock at the top is ~53–79 min vs the **2 h maintainer-validation limit** → roughly **2× compute headroom.** Spend it on *more signal per tile* (local views, predictor depth), **not** a bigger model.
- **Curation is orthogonal and unexplored at kde 0.05.** PaulScotti's *unbalanced* curation (H-optimus-0 tile embeddings, cluster-balancing OFF, preprocessing — allowed) hit **0.6653** while keeping robu 0.901; the *balanced* variant hurt (0.6437). Nobody has combined good curation with kde 0.05.
- **Seed variance ≈ 0.005.** Your own three runs span 0.6597–0.6649. A single lucky seed is fragile: the maintainer re-runs with a *different* seed and must still clear +0.006. Plan for margin, not a single peak.

---

## 4. Prioritized plan

Goal: get the recipe's **median** score to **≥0.668**, so a different-seed validation rerun clears 0.6652 even on an unlucky −0.005 draw.

### Experiment A — KDE uplift + full budget _(do first; high confidence; ≈+0.001)_
From your `lctx10x128-s7777.yaml`, change only:
```yaml
dino:
  kde_loss_weight: 0.05        # was 0.005  (proven overall optimum)
train:
  target_train_samples: 1000000  # was 850000 — train the full tile budget
                                 # (or delete the key entirely to use max_train_samples)
```
Expected ≈ 0.666 (Jeremy's level), clearing the bar. Keep `local_size: 128` for this run — your knn (0.752) is already frontier-best and 128 preserves it; run a paired `local_size: 112` variant to confirm which your recipe prefers.

### Experiment B — spend the unused FLOPs on local context _(medium confidence)_
You have ~2× wall-clock headroom before the 2 h cap. More/larger local crops is the lever that historically lifts progression (RyanKim's 10×128 local-crop runs reached prog 0.689). On top of A:
```yaml
train:
  local_views: 14              # was 10  (watch wall-clock stays < ~110 min)
```
Optionally nudge `dino.jepa_pred_depth: 4→6`. Re-measure wall time; back off if it approaches 2 h.

### Experiment C — curation × kde 0.05 _(medium confidence; the genuinely new combination)_
Reproduce PaulScotti's *unbalanced* H-optimus-0 tile curation as a **preprocessing** pass (embed TCGA tiles, resample toward tissue-rich / diverse tiles, cluster-balancing **OFF**), then train Experiment-A settings on the curated shards. This stacks an orthogonal +prog/+robustness lever onto the KDE gain — never tried together. Curation runs before the capped training and does not count against FLOPs or the 2 h limit.

### Experiment D — validation hardening _(do before claiming)_
Run your best candidate across **≥3 seeds** (e.g. 7777 / 314159 / 271828) and report the **median**. Only claim once the median is ≥0.668. This directly de-risks the maintainer rerun, which is where an unvalidated 0.6659 would otherwise be a coin flip.

### Do NOT touch
`probe.py`, anything in `benchmarking/`, or the locked probe keys in the config (dataset lists, probe counts) — any change there is auto-marked invalid. Backbone scaling (ViT-B/DINOv3-B) — proven worse here.

---

## 5. Expected trajectory

| Step | change | expected median | vs bar (0.6652) |
|---|---|---:|---:|
| now | your best single run | 0.6649 | −0.0003 |
| + A | kde 0.05, full 1M | ~0.666 | +0.001 |
| + B | 14 local views | ~0.667 | +0.002 |
| + C | curation × kde 0.05 | ~0.668–0.670 | +0.003–0.005 |
| + D | median of 3 seeds | robust ≥0.668 | validation-safe |

Sequence A → D. A is one-line and settles the "am I over the bar" question immediately; B and C build the margin that makes validation reliable; D is the discipline that turns an unvalidated peak into a validated leader.
