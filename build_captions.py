#!/usr/bin/env python3
# build_captions.py — offline MolCap caption builder (run once, before training; NOT a training dep).
# Renders each TCGA patient's metadata into one short clinical sentence and embeds it with a frozen,
# NON-pathology TEXT encoder (allowed: text is public non-image info; this is not a pathology IMAGE
# model). Writes patient_barcode -> L2-normalized float32[text_dim] into an .npz that
# molcap.caption_embeds points at, and prints the text_dim to copy into molcap.text_dim.
#
# Biomedical-encoder arm: swaps the general MiniLM sentence encoder for a biomedical SENTENCE encoder
# to test whether MiniLM's pathology-blind geometry is what distorted molecular/survival directions
# in the 2026-07-11 run. Raw PubMedBERT/BioBERT embed sentences poorly, so we use one FINE-TUNED for
# sentence similarity — that keeps "is it a real sentence encoder" fixed and changes only the domain,
# a clean A/B. Keep caption text, molcap.weight, ramp, and seed identical to the MiniLM arm.
#
# Usage (key=value overrides, no argparse per AGENTS.md):
#   python build_captions.py                          # uses the constants below
#   python build_captions.py out=molcap_captions_bio.npz
#   python build_captions.py selftest=1               # embed two dummy captions, print text_dim

import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Biomedical SENTENCE encoder (768-d). Alternatives, all 768-d sentence models:
#   pritamdeka/S-BioBert-snli-multinli-stsb , NeuML/pubmedbert-base-embeddings
ENCODER = "pritamdeka/S-PubMedBert-MS-MARCO"
META = "metadata/tcga_master_dataset.csv"
OUT = "molcap_captions_bio.npz"
# Map these to your metadata's column names.
C = dict(barcode="patient_barcode", cancer="cancer_type", subtype="subtype",
         stage="stage", msi="msi_status", mutations="top_mutations")


# One short clinical sentence per patient; keep this identical across encoder arms so only the
# encoder changes. Empty/NaN fields are dropped so the sentence stays clean.
def caption(r):
    bits = [r.get(C["cancer"]), r.get(C["subtype"]),
            f"stage {r[C['stage']]}" if r.get(C["stage"]) else None, r.get(C["msi"]),
            f"{r[C['mutations']]} mutant" if str(r.get(C["mutations"], "")).strip() else None]
    return ", ".join(str(b) for b in bits if b and str(b) != "nan")


args = dict(a.split("=", 1) for a in sys.argv[1:])
enc = SentenceTransformer(args.get("encoder", ENCODER))
if args.get("selftest"):
    v = enc.encode(["Colon adenocarcinoma, stage III, MSI-high, KRAS mutant", "Benign tissue"], normalize_embeddings=True)
    print(f"encoder ok: text_dim={v.shape[1]}, |v0|={np.linalg.norm(v[0]):.4f}")
    sys.exit()
meta = pd.read_csv(args.get("meta", META))
caps = {str(r[C["barcode"]]): caption(r) for _, r in meta.iterrows()}
bc = list(caps)
# normalize_embeddings=True so the .npz vectors are unit-norm and the training cosine loss is direct.
emb = enc.encode([caps[b] for b in bc], normalize_embeddings=True, show_progress_bar=True)
np.savez(args.get("out", OUT), **{b: emb[i].astype("float32") for i, b in enumerate(bc)})
print(f"wrote {len(bc)} captions -> {args.get('out', OUT)}; set molcap.text_dim={emb.shape[1]}")
