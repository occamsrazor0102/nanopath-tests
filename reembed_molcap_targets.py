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
# Failure hygiene: the biomedical NPZ is written to a staging path, every gate is evaluated into an
# audit table, and the artifact is PUBLISHED only on a full pass. On failure the staging file is
# removed AND any stale target at the fixed path is cleared, and a status=failed report is still
# written — so a failed run can never leave a target beside a failed report.
#
# Pure-math helpers (isotropy/geometry/gates/publish/json) import with no heavy deps so the test
# suite validates them against hand-computable fixtures without downloading the encoders.
#
#   python reembed_molcap_targets.py selftest=1        # pure-math self-checks, no models/data
#   python reembed_molcap_targets.py canonical=molcap_targets_minilm.npz \
#       biomed_out=molcap_targets_biomed.npz report_out=molcap_biomed_report.json

import hashlib
import io
import json
import os
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


# Width-controlled A/B: PCA-reduce a wider encoder to k dims (center -> project onto top-k principal
# directions) BEFORE isotropy, so a 768-d biomedical encoder is width-matched to MiniLM's 384. This
# makes the normalized-ratio gate apples-to-apples and holds the MolCap head shape (->384) constant
# with the MiniLM arm. The effective rank (~33) is far below 384, so essentially all signal is kept.
def pca_reduce(emb, k):
    xc = emb.astype(np.float64) - emb.astype(np.float64).mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    reduced = xc @ vt[:k].T
    retained = float(reduced.var(0).sum() / xc.var(0).sum())
    return reduced, retained


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


# Evaluate every hard gate on the staged artifact `corrected`, returning an audit table plus the
# verdict and first failing gate (frozen order). No exceptions: the caller writes a status report and
# publishes the NPZ only on a full pass.
def evaluate_gates(ids, captions, canon_ids, canon_captions, corrected, geom, tile_ids, target_width, staging_path):
    missing = len(set(map(str, tile_ids.tolist())) - set(map(str, ids.tolist())))
    dtmp = staging_path + ".dtmp"
    deterministic_savez(dtmp, {"patient_ids": ids, "captions": captions, "targets": corrected})
    deterministic = _sha256(dtmp) == _sha256(staging_path)
    os.remove(dtmp)
    er = geom["norm_effective_rank"] / REF_NORM_EFFRANK
    pr = geom["norm_participation_ratio"] / REF_NORM_PARTICIPATION
    checks = [
        ("rows", len(ids), CANON_ROWS, len(ids) == CANON_ROWS),
        ("unique_ids", len(set(ids.tolist())), CANON_ROWS, len(set(ids.tolist())) == CANON_ROWS),
        ("ids_match_canonical", "elementwise", "equal", bool(np.array_equal(ids, canon_ids))),
        ("captions_match_canonical", "elementwise", "equal", bool(np.array_equal(captions, canon_captions))),
        ("width", int(corrected.shape[1]), target_width, corrected.shape[1] == target_width),
        ("finite", bool(np.isfinite(corrected).all()), True, bool(np.isfinite(corrected).all())),
        ("max_unit_norm_error", geom["max_unit_norm_error"], MAX_ROWNORM_ERR, geom["max_unit_norm_error"] <= MAX_ROWNORM_ERR),
        ("deterministic_npz", deterministic, True, deterministic),
        ("fino_coverage_missing", missing, 0, missing == 0),
        ("mean_offdiag_cosine", abs(geom["mean_offdiag_cosine"]), MAX_OFFDIAG_COSINE, abs(geom["mean_offdiag_cosine"]) <= MAX_OFFDIAG_COSINE),
        ("effective_rank", geom["effective_rank"], MIN_EFFECTIVE_RANK, geom["effective_rank"] >= MIN_EFFECTIVE_RANK),
        ("participation_ratio", geom["participation_ratio"], MIN_PARTICIPATION, geom["participation_ratio"] >= MIN_PARTICIPATION),
        ("var_cv", geom["var_cv"], MAX_VAR_CV, geom["var_cv"] <= MAX_VAR_CV),
        ("norm_effrank_ratio", er, [RATIO_LO, RATIO_HI], RATIO_LO <= er <= RATIO_HI),
        ("norm_participation_ratio", pr, [RATIO_LO, RATIO_HI], RATIO_LO <= pr <= RATIO_HI),
    ]
    audit = [{"gate": n, "observed": o, "requirement": r, "passed": bool(p)} for n, o, r, p in checks]
    first_failed = next((c["gate"] for c in audit if not c["passed"]), None)
    return audit, first_failed is None, first_failed


# Publish the staged NPZ only on a full pass; on failure remove staging AND clear any stale target at
# the fixed path, so a failed run can never leave a target beside a failed report.
def publish_or_clear(passed, staging, final):
    if passed:
        os.replace(staging, final)
        return True, False
    if os.path.exists(staging):
        os.remove(staging)
    stale = os.path.exists(final)
    if stale:
        os.remove(final)
    return False, stale


# Replace non-finite floats with None (recording their paths) so a degenerate geometry audit still
# serializes as strict JSON.
def json_safe(obj, nonfinite, path=""):
    if isinstance(obj, dict):
        return {k: json_safe(v, nonfinite, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v, nonfinite, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, float) and not np.isfinite(obj):
        nonfinite.append(path or "<root>")
        return None
    return obj


# Encode captions with a pinned frozen sentence-transformer (lazy import: heavy dep, not needed by tests).
def encode(model, revision, captions):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model, revision=revision)
    return np.asarray(enc.encode(list(captions), batch_size=256, show_progress_bar=True), dtype=np.float64)


def selftest():
    assert abs(_effective_rank(np.ones(4)) - 4) < 1e-9 and abs(_participation_ratio(np.ones(4)) - 4) < 1e-9
    assert _effective_rank(np.array([1.0, 0, 0, 0])) == 1.0 and _participation_ratio(np.array([1.0, 0, 0, 0])) == 1.0
    rows = _l2(np.array([[1.0, 0], [0, 1], [1, 0]]))
    assert abs(_mean_offdiag_cosine(rows) - 1 / 3) < 1e-9
    rng = np.random.default_rng(0)
    aniso = rng.normal(size=(500, 32)) * (np.arange(1, 33) ** 2)
    corr = fit_isotropy(aniso)
    assert abs(np.linalg.norm(corr, axis=1) - 1).max() < 1e-5
    assert geometry(corr)["effective_rank"] > geometry(aniso)["effective_rank"]
    d = {"patient_ids": np.array(["A", "B"]), "captions": np.array(["x", "y"]), "targets": _l2(rng.normal(size=(2, BIOMED_WIDTH))).astype(np.float32)}
    deterministic_savez("/tmp/_st1.npz", d); deterministic_savez("/tmp/_st2.npz", d)
    assert _sha256("/tmp/_st1.npz") == _sha256("/tmp/_st2.npz")
    lowrank = rng.normal(size=(300, 5)) @ rng.normal(size=(5, 40))   # rank-5 signal embedded in 40-d
    red, ret = pca_reduce(lowrank, 8)
    assert red.shape == (300, 8) and ret > 0.999   # top-8 dims keep the rank-5 signal (width-controlled A/B)
    print("selftest OK: effective-rank/participation/off-diag-cosine, isotropy unit-norm+decorrelation, PCA reduction, deterministic NPZ")


def build(canonical, biomed_out, report_out, tile_ids_path=None, target_width=BIOMED_WIDTH):
    target_width = int(target_width)   # BIOMED_WIDTH (768) = strict A/B; 384 = width-controlled A/B
    assert _sha256(canonical) == CANON_SHA256, "canonical NPZ hash mismatch — wrong or modified source artifact"
    z = np.load(canonical, allow_pickle=False)
    order = np.argsort(z["patient_ids"].astype(str))   # canonical order: sorted by submitter_id
    ids, captions = z["patient_ids"][order], z["captions"][order]
    tile_ids = np.load(tile_ids_path)["patient_ids"] if tile_ids_path else ids   # fork supplies the 9,389 tile patients
    # Provenance precondition (not a tunable gate): MiniLM re-encode + shared isotropy must reproduce
    # the canonical corrected targets, or caption order / isotropy drifted.
    assert np.allclose(fit_isotropy(encode(MINILM_MODEL, MINILM_REV, captions)), z["targets"][order], atol=MINILM_REPRO_ATOL), "MiniLM reproduction failed — caption order or isotropy drift"
    raw = encode(BIOMED_MODEL, BIOMED_REV, captions)
    # Width-controlled A/B (a distinct pre-registration): PCA-reduce to target_width BEFORE isotropy so
    # the encoder is width-matched to MiniLM. Strict mode (target_width == native) leaves raw untouched.
    pca_retained = 1.0
    if target_width < raw.shape[1]:
        raw, pca_retained = pca_reduce(raw, target_width)
    corrected = fit_isotropy(raw)
    geom_raw, geom_corr = geometry(raw), geometry(corrected)
    staging = biomed_out + ".staging"
    deterministic_savez(staging, {"patient_ids": ids, "captions": captions, "targets": corrected})
    audit, passed, first_failed = evaluate_gates(ids, captions, ids, captions, corrected, geom_corr, tile_ids, target_width, staging)
    published, stale_cleared = publish_or_clear(passed, staging, biomed_out)
    nonfinite = []
    report = json_safe({
        "status": "passed" if passed else "failed", "published": published,
        "first_failed_gate": first_failed, "stale_target_cleared": stale_cleared,
        "artifact_sha256": _sha256(biomed_out) if published else None,
        "canonical_sha256": CANON_SHA256, "target_width": target_width, "pca_variance_retained": pca_retained,
        "encoders": {"minilm": {"model": MINILM_MODEL, "revision": MINILM_REV, "width": MINILM_WIDTH},
                     "biomed": {"model": BIOMED_MODEL, "revision": BIOMED_REV, "width": BIOMED_WIDTH}},
        "isotropy": {"floor_frac": ISO_FLOOR_FRAC, "power": ISO_POWER},
        "geometry": {"biomed_raw": geom_raw, "biomed_corrected": geom_corr,
                     "minilm_reference": {"norm_effective_rank": REF_NORM_EFFRANK, "norm_participation_ratio": REF_NORM_PARTICIPATION}},
        "coverage": {"tile_patients": int(len(tile_ids))},
        "gate_audit": audit,
    }, nonfinite)
    report["nonfinite_fields"] = nonfinite
    json.dump(report, open(report_out, "w"), indent=2, sort_keys=True, allow_nan=False)
    print(f"status={'passed' if passed else 'failed'} published={published} target_width={target_width} first_failed={first_failed}; report -> {report_out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    if args.get("selftest"):
        selftest()
    else:
        build(args["canonical"], args["biomed_out"], args["report_out"], args.get("tile_ids"), args.get("target_width", BIOMED_WIDTH))
