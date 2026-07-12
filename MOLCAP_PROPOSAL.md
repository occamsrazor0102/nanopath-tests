# MolCap: molecular captioning for nanopath

_A genuinely untried direction, derived from the 611-run Labless ledger. Companion to `IMPROVEMENT_PLAN.md` (incremental levers). This is the "next attempt" swing after the config tweaks plateaued._

## 1. Why the frontier is stuck (the evidence)

The whole field has converged on one recipe family — DINOv2-S + DINO/iBOT + JEPA + FINO + KDE + block-strided CLS readout + local views — and is now fighting over the 4th decimal (~0.664–0.666). The ledger shows **why**: the molecular/slide-level signal your weakest probes need is *achievable*, but every method that captures it **collapses the tile-level probes**:

| run | mutation | progression | overall |
|---|---:|---:|---:|
| `training-multi-294` (CLS molecular-diversity loss) | **0.681** | — | **0.5935** 💥 |
| `curation-halfnhalf` | — | **0.699** | 0.6431 💥 |
| `survival-rank-168` (scalar aux risk head on CLS) | 0.644 | — | 0.6473 |
| `off-trunk-bank-243` (separate survival head) | — | — | 0.6451 |
| frontier `bsc-s7777-k10` | 0.614 | 0.667 | 0.6659 |

**Root cause:** FINO and friends force molecular supervision and self-supervised tile discrimination through the **same CLS token**, where they are antagonistic. And in all 611 runs, **nobody has ever used a text encoder, a caption, or a language-space target** — everyone regresses raw scalars.

## 2. What `probe.py` actually reads (this shaped the design)

Reading the frozen (untouchable) probe code:
- **Linear / kNN / few-shot** → `model.probe_features()` = the **CLS token** (probe.py:325,331).
- **Progression / mutation / survival (slide-level)** → also `probe_features()` = CLS, then **mean-pooled across a slide's tiles** (probe.py:271,304).
- **Segmentation** → `encode_image()` patch tokens, registers dropped (probe.py:422).

So a literally *separate token* would never be read by the probes. The decoupling therefore has to live in two places the probes actually consume:
1. **Feature subspace** — supervise a *projection* of the CLS, so the CLS need only *contain* molecular semantics in a linear subspace (which the probes' own linear/CoxPH heads can read), not *become* the text vector. Tile discrimination keeps the rest of the CLS.
2. **Pooling frequency** — the slide probes read the *mean-pooled* CLS; molecular structure that survives averaging is exactly what they need, and it barely perturbs per-tile discrimination.

## 3. The idea — MolCap

Two untried pillars, aimed at the measured failure mode:

**(1) Supervise in language space, not scalar space.** Offline, render each patient's structured TCGA metadata into a sentence — *"Colon adenocarcinoma, stage III, MSI-high, KRAS-mutant, high lymphocytic infiltration"* — and embed it once with a **frozen, non-pathology text encoder** (MiniLM/BGE — text-only, so allowed). One dense, L2-normalized vector per patient: a smoother, correlation-aware target that encodes molecular *semantics* the scalar heads throw away.

**(2) Align a projection of the CLS to that vector, ramped in late, low weight.** A small `MolCapHead` (CLS → MLP → text_dim, L2-normed) trained with a cosine loss against the caption. Gradients flow through the shared trunk (so the trunk learns molecular structure and the mean-pooled CLS improves) but only reach the CLS through a projection (so linear/kNN stay clean).

The two closest prior attempts each had **one** pillar and stalled: `off-trunk-bank-243` decoupled a survival head (0.6451); `survival-rank-168` used a scalar aux on the CLS (0.6473). **Neither used language-space targets, and neither combined decoupling with a rich semantic target.** Their combination is the open space.

## 4. What this branch implements

Off by default — `main.yaml` has no `molcap` block, so `cfg.get("molcap", {})` disables every path and the base recipe is byte-identical. 66 new lines across three files:

- **`model.py`** — `MolCapHead` (Linear→GELU→Linear→L2-norm). Discarded at probe time.
- **`dataloader.py`** — loads an `.npz` caption bank (train split only); attaches `caption` + `has_caption` per tile (missing patient → zero vector, 0 weight, no gradient).
- **`train.py`** — builds the head when enabled, adds it to the optimizer/grad-clip, and adds one loss term inside `compute_losses`: cosine-align the global-crop student CLS projection to the caption, masked and view-tiled, scaled by a late linear ramp.
- **`configs/molcap.yaml`** — the runnable config with the `molcap` block.

**Validated here (CPU smoke test, `model.py` path):** head shape + L2-norm; loss finite in [0, 2·weight]; caption/weight tiling matches the crop-major `gf` order; gradients reach **both** trunk and head; all-missing-caption batch → exactly 0 (guarded divide). Not yet trained end-to-end (needs your GPU + data).

## 5. Build the caption bank (offline, before training)

```python
# build_captions.py — run once in your fork. Non-pathology text encoder = allowed.
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

meta = pd.read_csv("metadata/tcga_master_dataset.csv")          # your existing metadata
enc = SentenceTransformer("all-MiniLM-L6-v2")                   # 384-dim, non-pathology
def caption(r):
    bits = [r.get("cancer_type"), r.get("subtype"),
            f"stage {r['stage']}" if r.get("stage") else None,
            r.get("msi_status"),
            f"{r['top_mutations']} mutant" if str(r.get("top_mutations","")).strip() else None]
    return ", ".join(str(b) for b in bits if b and str(b) != "nan")
caps = {r["patient_barcode"]: caption(r) for _, r in meta.iterrows()}
bc = list(caps)
emb = enc.encode([caps[b] for b in bc], normalize_embeddings=True)   # L2-normed
np.savez("molcap_captions.npz", **{b: emb[i].astype("float32") for i, b in enumerate(bc)})
```
Point `molcap.caption_embeds` at the `.npz` and set `molcap.text_dim` to the encoder width (MiniLM = 384). Patient barcodes must match `patient_id_from_relpath` (first three dash-parts of the tile path, e.g. `TCGA-XX-XXXX`).

## 6. The ablation ladder (learn *which* pillar pays)

Run on top of your current frontier recipe (fork), not vanilla main:

- **C0 — decoupling alone:** move FINO's *existing* scalar targets onto the projected-CLS aux (drop the direct-on-CLS FINO heads). Tests "does decoupling stop the CLS collapse?"
- **C1 — language alone:** caption target, but weight it like FINO (no late ramp / higher weight). Tests "is language-space richer than scalars?"
- **C2 — MolCap (both):** `configs/molcap.yaml` settings on the fork recipe.

**Decisive read:** does the mutation/survival/progression trio rise while **linear/kNN hold**? No prior run achieved that combination — if C2 shows it, you're through the plateau by mechanism, not by a lucky seed. Judge on the usual ≥0.006 bar, across ≥3 seeds (median), per `IMPROVEMENT_PLAN.md`.

## 7. Risks & knobs

- **Caption template + encoder choice matter most.** Start minimal (subtype/stage/top mutations); a richer, well-ordered sentence usually helps. Try one biomedical text encoder vs one generic.
- **Weight / ramp is the antagonism dial.** Too high re-introduces the collapse; `weight 0.1`, `ramp_start 0.15` is a conservative start — sweep `weight ∈ {0.05, 0.1, 0.2}`.
- **Batch composition:** with i.i.d. tile sampling most patients appear once per batch, so this MVP supervises per-tile (toward the patient caption). A follow-up — an **EMA per-patient CLS-centroid bank** aligned to the caption — would supervise the exact mean-pooled quantity the slide probes read, at ~20 more lines; worth trying if C2 is promising.
- **Honest framing:** this is a research bet, not a guaranteed +0.006. But it is the one direction the field left untouched, aimed squarely at the failure mode the data reveals.

## 8. Result: `molcap-text-s7777` (2026-07-11) — mechanism confirmed, target rejected

First real run (per-patient MiniLM caption, weight 0.03, patch-routed late ramp) scored **0.66522**, −0.00069 vs same-seed `bsc-s7777-k10` (0.66591). Do not submit — a paired loss to a higher unvalidated run with no real margin. But the split is the informative part:

| | linear | kNN | slide/prog | seg | molecular | survival | overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Δ vs bsc-s7777-k10 | −0.0001 | **+0.0011** | **+0.0031** | −0.0021 | −0.0023 | −0.0040 | −0.0007 |

**The decoupling worked** — linear held flat, kNN and slide AUC *rose*. No prior ledger run lifted slide signal without collapsing tile metrics; the antagonism is breakable. **The target failed**: molecular and survival (the categories it should help) dropped, and that plus seg erased the slide gain. Likely causes: (a) per-tile alignment flattens intra-slide heterogeneity mutation/survival need; (b) MiniLM is pathology-blind, so "KRAS-mutant" vs "wild-type" sit nearly collinear and get compressed; (c) shared-trunk gradients perturb patch tokens → seg −0.0021. Survival is also the noisiest probe, so part of −0.0040 is variance.

**Next arm (biomedical encoder — tests cause b).** Routing is encoder-agnostic, so this is offline-only: rebuild captions with a biomedical **sentence** encoder (`pritamdeka/S-PubMedBert-MS-MARCO`, 768-d — *not* raw PubMedBERT, which embeds sentences poorly) via `build_captions.py`, set `molcap.text_dim: 768` and `caption_embeds` to the new `.npz`, and **keep caption text / weight 0.03 / ramp / seed 7777 identical** so the encoder is the only changed variable. Judge paired against this run: does molecular/survival recover while linear/kNN hold? Allowed — a text encoder trained on biomedical *literature* is public non-image info, not a pathology image model (worth a one-line maintainer confirm since it's domain-adjacent). If it doesn't recover, cause (a) dominates → build the EMA per-patient-centroid variant (§7) next.
