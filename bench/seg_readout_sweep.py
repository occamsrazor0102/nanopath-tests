#!/usr/bin/env python3
# seg_readout_sweep.py — offline evidence for the segmentation encode_image readout (NOT training).
# Loads the real pretrained DINOv2-S backbone and measures foreground IoU of the segmentation readout
# on a controlled H&E-like nuclei task, mirroring the locked probe flow: features =
# encode_image(x)[:, registers:] -> per-patch linear head -> upsample prediction to full res -> IoU.
# Backbone runs once/image (all block outputs cached); each readout config is built from the cache with
# the exact JBU code from model.encode_image, so this isolates the readout knobs from backbone noise.
# It is a proxy (synthetic nuclei), not the locked pannuke/monusac/consep probe — use it for the
# relative ranking of readout choices; the absolute nanopath seg number comes from the real probe.
#   python bench/seg_readout_sweep.py
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from model import DinoV2ViT, load_dinov2_pretrained

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
H = W = 16   # 224 / patch 14
N_TR, N_VA = 120, 48


def make_image(rng):   # eosin-pink background + hematoxylin-purple nuclei ellipses -> per-pixel mask
    bg, nuc = np.array([0.82, 0.60, 0.78]), np.array([0.42, 0.24, 0.52])
    low = np.array([np.kron(c, np.ones((28, 28))) for c in rng.normal(0, 1, (3, 8, 8))])
    img, mask = bg[:, None, None] + 0.06 * low, np.zeros((224, 224), np.float32)
    yy, xx = np.mgrid[0:224, 0:224]
    for _ in range(rng.integers(25, 55)):
        cy, cx = rng.integers(8, 216, 2); ry, rx = rng.integers(4, 12, 2); th = rng.uniform(0, np.pi)
        a = np.cos(th) * (xx - cx) + np.sin(th) * (yy - cy)
        b = -np.sin(th) * (xx - cx) + np.cos(th) * (yy - cy)
        e = (a / rx) ** 2 + (b / ry) ** 2 <= 1.0
        img[:, e] = nuc[:, None] + 0.05 * rng.normal(0, 1, (3, e.sum())); mask[e] = 1.0
    return np.clip(img + 0.02 * rng.normal(0, 1, img.shape), 0, 1).astype(np.float32), mask


def jbu(patches, guide, G):   # exact joint-bilateral upsample from model.encode_image, parameterized by G
    B = patches.shape[0]
    up = F.interpolate(patches.transpose(1, 2).reshape(B, patches.shape[-1], H, W).float(), size=(G, G), mode="bilinear", align_corners=False)
    glr, ghr = F.interpolate(guide, size=(H, W), mode="area"), F.interpolate(guide, size=(G, G), mode="area")
    wr = torch.exp(-((ghr - F.interpolate(glr, size=(G, G), mode="nearest")).abs() ** 2) / 0.02)
    blur = F.avg_pool2d(F.pad(up, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return (up + (1 - wr) * (up - blur)).flatten(2).transpose(1, 2)


def iou(cache, guide, GT, reg, blocks, G, seed):
    rng = np.random.default_rng(seed)
    Fa = jbu(torch.cat([cache[i][:, reg:] for i in blocks], -1), guide, G).numpy()
    gtG = F.interpolate(torch.tensor(GT)[:, None], size=(G, G), mode="area").squeeze(1).numpy()
    lab = (gtG.reshape(len(GT), -1) > 0.5).astype(np.float32)
    Xtr, ytr = Fa[:N_TR].reshape(-1, Fa.shape[-1]), lab[:N_TR].reshape(-1)
    idx = rng.choice(len(Xtr), 30000, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt, yt = torch.tensor((Xtr - mu) / sd), torch.tensor(ytr)
    Wt, bt = torch.zeros(Xt.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([Wt, bt], max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(Xt @ Wt + bt, yt) + 1e-3 * (Wt * Wt).sum(); loss.backward(); return loss
    opt.step(closure)
    z = (torch.tensor((Fa[N_TR:] - mu) / sd) @ Wt.detach() + bt.detach()).numpy().reshape(N_VA, G, G)
    pr = F.interpolate(torch.tensor((z > 0).astype(np.float32))[:, None], size=(224, 224), mode="nearest")[:, 0].numpy() > 0.5
    gtv = GT[N_TR:] > 0.5
    inter, uni = (pr & gtv).reshape(N_VA, -1).sum(1), (pr | gtv).reshape(N_VA, -1).sum(1)
    return float(np.mean(inter / (uni + 1e-9)))


def main():
    rng = np.random.default_rng(0); torch.manual_seed(0)
    model = DinoV2ViT(variant="dinov2_vits14_reg"); load_dinov2_pretrained(model); model.eval()
    imgs, masks = zip(*(make_image(rng) for _ in range(N_TR + N_VA)))
    X, GT = torch.tensor(np.stack(imgs)), np.stack(masks).astype(np.float32)
    guide = X.mean(1, keepdim=True)
    guide = (guide - guide.amin((2, 3), keepdim=True)) / (guide.amax((2, 3), keepdim=True) - guide.amin((2, 3), keepdim=True) + 1e-6)
    cache = [[] for _ in range(len(model.blocks))]
    with torch.no_grad():
        for s in range(0, len(X), 8):
            xt = model._prepare_tokens((X[s:s + 8] - MEAN) / STD)
            for i, blk in enumerate(model.blocks):
                xt = blk(xt); cache[i].append(model.norm(xt)[:, 1:])
    cache = [torch.cat(cs, 0) for cs in cache]
    sets = {"last-4 (contiguous)": (8, 9, 10, 11), "last-6 (prior seg record readout)": (6, 7, 8, 9, 10, 11),
            "strided (4,6,8,11)": (4, 6, 8, 11), "wide (2,5,8,11)": (2, 5, 8, 11)}
    print(f"{'seg readout (G=32)':<36}{'IoU  mean ± sd (3 seeds)':>26}")
    print("-" * 62)
    for name, blocks in sets.items():
        vals = [iou(cache, guide, GT, model.registers, blocks, 32, s) for s in (1, 2, 3)]
        print(f"{name:<36}{np.mean(vals):>16.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
