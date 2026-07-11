# Build deterministic patient-level MolCap target banks from the committed TCGA metadata.
# Usage: python build_molcap_targets.py <metadata.csv> <output.npz> <text|structured|shuffled>

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL = ["cancer_type", "disease_type", "primary_site", "primary_diagnosis", "cbio_subtype", "ajcc_pathologic_stage", "tumor_grade"]
CONTINUOUS = ["cbio_msi_score", "cbio_fraction_genome_altered", "cbio_mutation_count"]
GENES = ["mol_kras_", "mol_braf_", "mol_tp53_", "mol_egfr_"]
COLUMNS = ["submitter_id", *CATEGORICAL, *CONTINUOUS, *GENES]


def aggregate_patients(path):
    raw = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    for column in COLUMNS[1:]:
        conflicts = raw.groupby("submitter_id")[column].nunique(dropna=True)
        assert not (conflicts > 1).any(), f"conflicting patient values for {column}: {conflicts[conflicts > 1].index.tolist()[:5]}"
    return raw.sort_values("submitter_id").groupby("submitter_id", as_index=False, sort=True).first()


def render_captions(frame):
    cuts = {column: frame[column].quantile([1 / 3, 2 / 3]).tolist() for column in CONTINUOUS}
    captions = []
    for _, row in frame.iterrows():
        bits = []
        if pd.notna(row["disease_type"]): bits.append(str(row["disease_type"]))
        if pd.notna(row["primary_site"]): bits.append(f"primary site {row['primary_site']}")
        if pd.notna(row["primary_diagnosis"]): bits.append(f"diagnosis {row['primary_diagnosis']}")
        if pd.notna(row["cancer_type"]): bits.append(f"TCGA cancer type {row['cancer_type']}")
        if pd.notna(row["cbio_subtype"]): bits.append(f"molecular subtype {row['cbio_subtype']}")
        if pd.notna(row["ajcc_pathologic_stage"]): bits.append(f"pathologic {row['ajcc_pathologic_stage']}")
        if pd.notna(row["tumor_grade"]): bits.append(f"tumor grade {row['tumor_grade']}")
        for column, label in zip(CONTINUOUS, ("microsatellite instability", "fraction genome altered", "mutation burden")):
            if pd.notna(row[column]):
                level = "low" if row[column] <= cuts[column][0] else "intermediate" if row[column] <= cuts[column][1] else "high"
                bits.append(f"{level} {label}")
        for column in GENES:
            value = str(row[column]).lower() if pd.notna(row[column]) else ""
            if value in ("positive", "amplified"):
                bits.append(f"{column[4:-1].upper()} {value}")
        captions.append(", ".join(bits))
    return captions


def _normalize(targets):
    targets = np.asarray(targets, dtype=np.float32)
    assert np.isfinite(targets).all(), "MolCap targets contain non-finite values"
    norms = np.linalg.norm(targets, axis=1, keepdims=True)
    assert (norms > 0).all(), "MolCap targets contain a zero vector"
    return targets / norms


def structured_targets(frame, dim=384, seed=7777):
    categorical = pd.get_dummies(frame[CATEGORICAL].astype("string").fillna("missing"), dtype=np.float32).to_numpy()
    numeric = frame[CONTINUOUS].astype(np.float32)
    missing = numeric.isna().to_numpy(dtype=np.float32)
    numeric = numeric.fillna(numeric.median()).to_numpy(dtype=np.float32)
    scale = numeric.std(0); scale[scale == 0] = 1
    features = np.concatenate([categorical, (numeric - numeric.mean(0)) / scale, missing], axis=1)
    projection = np.random.default_rng(seed).standard_normal((features.shape[1], dim), dtype=np.float32) / np.sqrt(dim)
    return _normalize(features @ projection)


def encode_text(captions, encoder=None):
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _normalize(encoder.encode(captions, normalize_embeddings=True))


def save_target_bank(path, patient_ids, targets, captions, mode):
    arrays = {
        "patient_ids": np.asarray(patient_ids, dtype=str),
        "targets": _normalize(targets).astype(np.float32),
        "captions": np.asarray(captions, dtype=str),
        "mode": np.asarray(mode, dtype=str),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO(); np.save(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def audit_targets(targets):
    centered = targets - targets.mean(0, keepdims=True)
    values = np.linalg.eigvalsh(centered.T @ centered / max(1, len(targets) - 1)).clip(0)
    weights = values / values.sum()
    effective_rank = float(np.exp(-(weights[weights > 0] * np.log(weights[weights > 0])).sum()))
    std = float(targets.std())
    assert std > 0.01 and effective_rank > 32, f"collapsed MolCap geometry: std={std:.4f}, effective_rank={effective_rank:.1f}"
    return std, effective_rank


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in ("text", "structured", "shuffled"):
        raise ValueError("usage: python build_molcap_targets.py <metadata.csv> <output.npz> <text|structured|shuffled>")
    source, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    patients = aggregate_patients(source); captions = render_captions(patients)
    targets = structured_targets(patients) if mode == "structured" else encode_text(captions)
    if mode == "shuffled": targets = targets[np.random.default_rng(7777).permutation(len(targets))]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_target_bank(output, patients["submitter_id"], targets, captions, mode)
    std, rank = audit_targets(targets)
    print(f"wrote {len(targets)} {mode} targets to {output}  dim={targets.shape[1]}  std={std:.4f}  effective_rank={rank:.1f}")


if __name__ == "__main__":
    main()
