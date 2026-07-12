#!/usr/bin/env python3
# reembed_molcap_targets.py — offline MolCap target re-embedding for the strict biomedical
# encoder A/B (NOT a training dep; excluded from the Labless source snapshot).
#
# Reuses the EXACT canonical captions + patient identifiers from the MiniLM arm and changes ONLY the
# frozen sentence encoder (and resulting target width). Per-encoder it applies the identical isotropy
# procedure and constants, so the comparison isolates biomedical sentence semantics after comparable
# geometry normalization. Every provenance value (artifact hash, model revisions, widths, isotropy
# constants, gate thresholds) is frozen below; the tool fails loudly if any input drifts.
#
# Stages (main): validate canonical NPZ -> re-encode captions with pinned MiniLM + PubMedBERT ->
# per-encoder isotropy -> geometry -> hard gates (incl. MiniLM reproduction) -> deterministic NPZ + JSON.
# Pure-math helpers (isotropy/geometry/gates/deterministic_savez) import with no heavy deps so the
# test suite can validate them against hand-computable fixtures without downloading the encoders.
#
#   python reembed_molcap_targets.py selftest=1        # pure-math self-checks, no models/data
#   python reembed_molcap_targets.py canonical=molcap_targets_minilm.npz \
#       biomed_out=molcap_targets_biomed.npz report_out=molcap_biomed_report.json

import hashlib
import io
import json
import sys
import zipfile

import numpy as np

# ---- Frozen provenance (predeclared; the tool refuses to run if reality differs) -------------------
CANON_SHA256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
CANON_ROWS = 11428
TILE_PATIENTS = 9389
MINILM_MODEL, MINILM_REV = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BIOMED_MODEL, BIOMED_REV = "pritamdeka/S-PubMedBert-MS-MARCO", "96786c7024f95c5aac7f2b9a18086c7b97b23036"
MINILM_WIDTH, BIOMED_WIDTH = 384, 768

# ---- Frozen isotropy constants (identical procedure for both encoders; never retuned) --------------
ISO_FLOOR_FRAC = 0.05   # floor eigenvalues at 5% of the largest before scaling
ISO_POWER = -0.1        # mild inverse-eigenvalue scaling (partial whitening)

# ---- Frozen gate thresholds ------------------------------------------------------------------------
MAX_OFFDIAG_COSINE = 0.01
MIN_EFFECTIVE_RANK = 32.0
MIN_PARTICIPATION = 16.0
MAX_VAR_CV = 0.75
RATIO_LO, RATIO_HI = 0.5, 2.0
MAX_ROWNORM_ERR = 1e-5
MINILM_REPRO_ATOL = 2e-5
# Predeclared MiniLM-corrected reference geometry (normalized to width) for the comparability ratios.
REF_NORM_EFFRANK = 0.0961024
REF_NORM_PARTICIPATION = 0.0592323


def _l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


# mean cosine over unordered off-diagonal row pairs via the identity sum_{i!=j} xi.xj = ||sum xi||^2 - n
# (rows must be unit norm); avoids the n^2 Gram matrix for n=11428.
def _mean_offdiag_cosine(unit_rows):
    n = unit_rows.shape[0]
    s = unit_rows.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


def _eigs(vectors):
    xc = vectors - vectors.mean(0)
    cov = (xc.T @ xc) / xc.shape[0]
    return np.linalg.eigvalsh(cov).clip(0.0)   # ascending, non-negative


# Roy-Vetterli effective rank = exp(Shannon entropy of the normalized eigenvalue spectrum).
def _effective_rank(eigs):
    p = eigs / eigs.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# Participation ratio = (sum eig)^2 / sum(eig^2): the effective number of comparably-sized directions.
def _participation_ratio(eigs):
    return float((eigs.sum() ** 2) / (eigs ** 2).sum())


# Full geometry report for one embedding matrix. Directional metrics use an L2-normalized copy;
# the unit-norm error is measured on the array AS GIVEN (that is what the write gate checks).
def geometry(emb):
    emb = emb.astype(np.float64)
    norms = np.linalg.norm(emb, axis=-1)
    unit = emb / norms.clip(1e-12)[:, None]
    eigs = _eigs(unit)
    er, pr = _effective_rank(eigs), _participation_ratio(eigs)
    var = unit.var(0)
    width = emb.shape[1]
    return {
        "rows": int(emb.shape[0]), "width": int(width),
        "mean_offdiag_cosine": _mean_offdiag_cosine(unit),
        "effective_rank": er, "norm_effective_rank": er / width,
        "participation_ratio": pr, "norm_participation_ratio": pr / width,
        "var_min": float(var.min()), "var_median": float(np.median(var)), "var_max": float(var.max()),
        "var_cv": float(var.std() / var.mean()),
        "max_unit_norm_error": float(np.abs(norms - 1.0).max()),
    }


# Identical per-encoder isotropy: L2 -> subtract mean -> eigendecompose covariance -> scale each
# eigen-direction by max(lambda, lambda_max*0.05)^-0.1 -> rotate back -> L2. Constants are frozen.
def fit_isotropy(emb):
    x = _l2(emb.astype(np.float64))
    xc = x - x.mean(0)
    cov = (xc.T @ xc) / xc.shape[0]
    w, V = np.linalg.eigh(cov)
    scale = np.maximum(w, w.max() * ISO_FLOOR_FRAC) ** ISO_POWER
    return _l2((xc @ V) * scale @ V.T).astype(np.float32)


# Deterministic .npz: fixed member order + fixed zip timestamps + no pickle, so byte-identical writes
# of identical arrays hash identically (the determinism gate).
def deterministic_savez(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, arrays[name], allow_pickle=False)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# All hard target gates on the written artifact `corrected`. Raises AssertionError (fail loudly)
# on the first violation.
def apply_gates(ids, captions, canon_ids, canon_captions, corrected, geom, tile_ids, biomed_path):
    assert len(ids) == CANON_ROWS == len(set(ids.tolist())), f"expected {CANON_ROWS} unique ids, got {len(ids)}"
    assert np.array_equal(ids, canon_ids), "patient ids differ from canonical (order or content)"
    assert np.array_equal(captions, canon_captions), "caption strings differ from canonical (re-rendered, not reused)"
    assert corrected.shape[1] == BIOMED_WIDTH, f"target width {corrected.shape[1]} != {BIOMED_WIDTH}"
    assert np.isfinite(corrected).all(), "non-finite target values"
    assert geom["max_unit_norm_error"] <= MAX_ROWNORM_ERR, f"row-norm error {geom['max_unit_norm_error']:.2e} > {MAX_ROWNORM_ERR}"
    missing = set(map(str, tile_ids)) - set(map(str, ids.tolist()))
    assert len(missing) == 0, f"{len(missing)} of {len(tile_ids)} tile patients lack a caption (coverage < 100%)"
    assert abs(geom["mean_offdiag_cosine"]) <= MAX_OFFDIAG_COSINE, f"|off-diag cosine| {geom['mean_offdiag_cosine']:.4g} > {MAX_OFFDIAG_COSINE}"
    assert geom["effective_rank"] >= MIN_EFFECTIVE_RANK, f"effective rank {geom['effective_rank']:.2f} < {MIN_EFFECTIVE_RANK}"
    assert geom["participation_ratio"] >= MIN_PARTICIPATION, f"participation ratio {geom['participation_ratio']:.2f} < {MIN_PARTICIPATION}"
    assert geom["var_cv"] <= MAX_VAR_CV, f"per-dim var CV {geom['var_cv']:.3f} > {MAX_VAR_CV}"
    er_ratio = geom["norm_effective_rank"] / REF_NORM_EFFRANK
    pr_ratio = geom["norm_participation_ratio"] / REF_NORM_PARTICIPATION
    assert RATIO_LO <= er_ratio <= RATIO_HI, f"norm-effective-rank ratio {er_ratio:.3f} outside [{RATIO_LO},{RATIO_HI}]"
    assert RATIO_LO <= pr_ratio <= RATIO_HI, f"norm-participation ratio {pr_ratio:.3f} outside [{RATIO_LO},{RATIO_HI}]"
    # Determinism: a second write of identical arrays must hash identically.
    tmp = biomed_path + ".dtmp"
    deterministic_savez(tmp, {"patient_ids": ids, "captions": captions, "targets": corrected})
    assert _sha256(tmp) == _sha256(biomed_path), "non-deterministic NPZ serialization"
    return {"er_ratio": er_ratio, "pr_ratio": pr_ratio, "missing_patients": 0}


# Encode captions with a pinned frozen sentence-transformer (lazy import: heavy dep, not needed by tests).
def encode(model, revision, captions):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model, revision=revision)
    return np.asarray(enc.encode(list(captions), batch_size=256, show_progress_bar=True), dtype=np.float64)


def selftest():
    # Geometry/isotropy invariants on synthetic anisotropic data — the hand-computable coverage.
    assert abs(_effective_rank(np.ones(4)) - 4) < 1e-9 and abs(_participation_ratio(np.ones(4)) - 4) < 1e-9
    assert _effective_rank(np.array([1.0, 0, 0, 0])) == 1.0 and _participation_ratio(np.array([1.0, 0, 0, 0])) == 1.0
    rows = _l2(np.array([[1.0, 0], [0, 1], [1, 0]]))
    assert abs(_mean_offdiag_cosine(rows) - 1 / 3) < 1e-9
    rng = np.random.default_rng(0)
    aniso = rng.normal(size=(500, 32)) * (np.arange(1, 33) ** 2)   # strongly anisotropic
    corr = fit_isotropy(aniso)
    assert abs(np.linalg.norm(corr, axis=1) - 1).max() < 1e-5, "isotropy output not unit norm"
    assert geometry(corr)["effective_rank"] > geometry(aniso)["effective_rank"], "isotropy did not raise effective rank"
    d = {"patient_ids": np.array(["A", "B"]), "captions": np.array(["x", "y"]), "targets": _l2(rng.normal(size=(2, BIOMED_WIDTH))).astype(np.float32)}
    deterministic_savez("/tmp/_st1.npz", d); deterministic_savez("/tmp/_st2.npz", d)
    assert _sha256("/tmp/_st1.npz") == _sha256("/tmp/_st2.npz"), "savez not deterministic"
    print("selftest OK: effective-rank/participation/off-diag-cosine, isotropy unit-norm+decorrelation, deterministic NPZ")


def build(canonical, biomed_out, report_out, tile_ids_path=None):
    assert _sha256(canonical) == CANON_SHA256, "canonical NPZ hash mismatch — wrong or modified source artifact"
    z = np.load(canonical, allow_pickle=False)
    ids, captions = z["patient_ids"], z["captions"]
    order = np.argsort(ids.astype(str))   # canonical order: sorted by submitter_id
    ids, captions = ids[order], captions[order]
    tile_ids = np.load(tile_ids_path)["patient_ids"] if tile_ids_path else ids   # fork supplies the 9,389 tile patients
    # MiniLM reproduction: re-encode + isotropy must match the canonical corrected targets.
    minilm_corr = fit_isotropy(encode(MINILM_MODEL, MINILM_REV, captions))
    assert np.allclose(minilm_corr, z["targets"][order], atol=MINILM_REPRO_ATOL), "MiniLM reproduction failed — caption order or isotropy drift"
    raw = encode(BIOMED_MODEL, BIOMED_REV, captions)
    corrected = fit_isotropy(raw)
    geom_raw, geom_corr = geometry(raw), geometry(corrected)
    deterministic_savez(biomed_out, {"patient_ids": ids, "captions": captions, "targets": corrected})
    ratios = apply_gates(ids, captions, ids, captions, corrected, geom_corr, tile_ids, biomed_out)
    report = {
        "canonical_sha256": CANON_SHA256, "biomed_sha256": _sha256(biomed_out),
        "encoders": {"minilm": {"model": MINILM_MODEL, "revision": MINILM_REV, "width": MINILM_WIDTH},
                     "biomed": {"model": BIOMED_MODEL, "revision": BIOMED_REV, "width": BIOMED_WIDTH}},
        "isotropy": {"floor_frac": ISO_FLOOR_FRAC, "power": ISO_POWER},
        "geometry": {"biomed_raw": geom_raw, "biomed_corrected": geom_corr,
                     "minilm_reference": {"norm_effective_rank": REF_NORM_EFFRANK, "norm_participation_ratio": REF_NORM_PARTICIPATION}},
        "coverage": {"tile_patients": int(len(tile_ids)), "missing": ratios["missing_patients"]},
        "comparability_ratios": {"norm_effective_rank": ratios["er_ratio"], "norm_participation": ratios["pr_ratio"]},
        "thresholds": {"max_offdiag_cosine": MAX_OFFDIAG_COSINE, "min_effective_rank": MIN_EFFECTIVE_RANK,
                       "min_participation": MIN_PARTICIPATION, "max_var_cv": MAX_VAR_CV, "ratio_bounds": [RATIO_LO, RATIO_HI]},
    }
    json.dump(report, open(report_out, "w"), indent=2, sort_keys=True)
    print(f"biomed targets -> {biomed_out} (sha256 {report['biomed_sha256'][:12]}); all gates passed; report -> {report_out}")


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    if args.get("selftest"):
        selftest()
    else:
        build(args["canonical"], args["biomed_out"], args["report_out"], args.get("tile_ids"))
