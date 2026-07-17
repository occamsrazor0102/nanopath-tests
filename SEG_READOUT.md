# Maximizing segmentation: strided block fusion in the seg readout

Goal: raise the segmentation probe as high as possible with the **minimal** change, above the highest
seg ever recorded (Labless: `jepa-fino-densefuse6` = **0.3384**). Segmentation is the one probe that
reads **patch tokens** — `probe.py:422` feeds it `model.encode_image(x)[:, model.registers:]` — so the
seg readout in `encode_image` is an **isolated, segmentation-only** lever (the CLS/`probe_features`
path is untouched, verified). `probe.py:975` builds `DinoV2ViT(variant=...)` with no extra args, so the
readout is set by module constants in `model.py`, not YAML.

## The change (2 lines, same block count, same feature dim)

`encode_image` fused the **last 4 contiguous** blocks. Change it to the **strided (4,6,8,11)** set —
the same depths already validated for the CLS readout (`PROBE_FEATURE_BLOCKS`):

```
SEG_FUSE_BLOCKS = (4, 6, 8, 11)     # was: last-4 contiguous (i >= len(blocks) - 4)
   ... if i in SEG_FUSE_BLOCKS ...  # was: if i >= len(self.blocks) - SEG_FUSE_BLOCKS
```

Nothing else moves: still 4 fused blocks → identical feature dimension (4·384), still JBU-upsampled to
G=32, still only the seg probe sees it. No retraining is even required for the readout itself — it is
eval-time, so it re-scores an existing checkpoint's seg probe.

## Why strided — the experiment (real DINOv2-S, controlled H&E nuclei seg)

`bench/seg_readout_sweep.py` loads the **real pretrained DINOv2-S**, generates a controlled H&E-like
nuclei task (small purple ellipses on eosin-pink tissue), and mirrors the locked probe flow:
`encode_image(x)[:, registers:]` → per-patch linear head → upsample prediction to full res → foreground
IoU. The backbone runs once/image (all blocks cached) and each readout is built with the exact JBU code
from `encode_image`, so this isolates the *readout*, not backbone noise. 3-seed means:

| seg readout (G=32) | IoU (mean ± sd) |
|---|---:|
| last-4 (contiguous, prior `main`) | 0.4959 ± 0.0053 |
| last-6 (the 0.3384 record's readout) | 0.4996 ± 0.0015 |
| **strided (4,6,8,11)** | **0.5234 ± 0.0013** |
| wide (2,5,8,11) | 0.5231 ± 0.0039 |

Strided beats the record's exact last-6 readout by **+0.024 IoU** — ~15× the seed noise. Mechanism:
strided blocks span depth, fusing **fine spatial detail** (block 4) with **late semantics** (block 11);
contiguous late blocks are all semantically coarse and spatially redundant, so they localize sub-patch
nuclei boundaries worse. Two knobs I also swept and left alone because they don't help:
- **Upsample grid G**: saturates by 32 (16→0.38, 24→0.47, **32→0.52**, 40≈32, 48 regresses). Keep 32.
- **Fuse count** (last-N): 4/6/8/12 all sit ~0.50–0.52, within noise. The *selection* matters, not the count.

## Caveat & how to get the real number

This is a **proxy** (synthetic nuclei, real backbone) — use it for the *relative ranking* of readout
choices, which is robust. The absolute nanopath seg number comes from the locked pannuke/monusac/consep
probe in the fork. To confirm it clears 0.3384: run the standard recipe (the seg readout is eval-only,
so training is byte-identical and all other probes are unchanged), and read `seg_mean_jaccard`. Because
the change beats the record's *own* readout by +0.024 IoU on the real backbone at the same block count
and resolution, it is the highest-confidence minimal lever available for segmentation. Reproduce the
sweep with `python bench/seg_readout_sweep.py`.
