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

import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

# ---- Frozen provenance (predeclared; the tool refuses to run if reality differs) -------------------
CANON_SHA256 = CANONICAL_SHA256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
CANON_ROWS = CANONICAL_ROWS = 11428
TILE_PATIENTS = FINO_PATIENT_COUNT = 9389
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REV = MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BIOMED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
BIOMED_REV = BIOMED_REVISION = "96786c7024f95c5aac7f2b9a18086c7b97b23036"
MINILM_WIDTH = 384
BIOMED_WIDTH = BIOMED_DIM = 768
RAW768 = "raw768"
PCA384 = "pca384"
PCA_DIM = 384
PCA_MIN_VARIANCE = 0.99
VARIANT_SPECS = {
    RAW768: {"target_width": BIOMED_WIDTH, "artifact_mode": "biomedical"},
    PCA384: {"target_width": PCA_DIM, "artifact_mode": "biomedical-pca384"},
}

# ---- Frozen isotropy constants (identical procedure for both encoders; never retuned) --------------
ISOTROPY_FLOOR = ISO_FLOOR_FRAC = 0.05
ISOTROPY_POWER = 0.1
ISO_POWER = -ISOTROPY_POWER

# ---- Frozen gate thresholds ------------------------------------------------------------------------
MAX_OFFDIAG_COSINE = 0.01
MIN_EFFECTIVE_RANK = 32.0
MIN_PARTICIPATION = 16.0
MAX_VAR_CV = 0.75
RATIO_LO, RATIO_HI = 0.5, 2.0
MAX_ROWNORM_ERR = 1e-5
MINILM_REPRO_ATOL = 2e-5
REF_NORM_EFFRANK = 0.0961024
REF_NORM_PARTICIPATION = 0.0592323


def _l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def _mean_offdiag_cosine(unit_rows):
    n = unit_rows.shape[0]
    s = unit_rows.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


def _eigs(vectors):
    xc = vectors - vectors.mean(0)
    cov = (xc.T @ xc) / xc.shape[0]
    return np.linalg.eigvalsh(cov).clip(0.0)


def _effective_rank(eigs):
    p = eigs / eigs.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _participation_ratio(eigs):
    return float((eigs.sum() ** 2) / (eigs ** 2).sum())


def geometry_metrics(targets):
    x = np.asarray(targets, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    centered = x - x.mean(0, keepdims=True)
    values = np.linalg.eigvalsh(centered.T @ centered / max(1, len(x) - 1)).clip(0)
    weights = values / values.sum()
    effective_rank = round(float(np.exp(-(weights[weights > 0] * np.log(weights[weights > 0])).sum())), 12)
    participation = round(float(values.sum() ** 2 / np.square(values).sum()), 12)
    variance = x.var(0, ddof=1)
    off_diagonal = (np.square(x.sum(0)).sum() - len(x)) / (len(x) * (len(x) - 1))
    return {
        "rows": len(x),
        "width": x.shape[1],
        "mean_off_diagonal_cosine": float(off_diagonal),
        "effective_rank": float(effective_rank),
        "normalized_effective_rank": float(effective_rank / x.shape[1]),
        "participation_ratio": float(participation),
        "normalized_participation_ratio": float(participation / x.shape[1]),
        "variance_min": float(variance.min()),
        "variance_median": float(np.median(variance)),
        "variance_max": float(variance.max()),
        "variance_cv": float(variance.std() / variance.mean()),
        "max_unit_norm_error": float(np.abs(norms - 1).max()),
    }


def geometry(emb):
    metrics = geometry_metrics(_l2(np.asarray(emb, dtype=np.float64)))
    return {
        "rows": metrics["rows"],
        "width": metrics["width"],
        "mean_offdiag_cosine": metrics["mean_off_diagonal_cosine"],
        "effective_rank": metrics["effective_rank"],
        "norm_effective_rank": metrics["normalized_effective_rank"],
        "participation_ratio": metrics["participation_ratio"],
        "norm_participation_ratio": metrics["normalized_participation_ratio"],
        "var_min": metrics["variance_min"],
        "var_median": metrics["variance_median"],
        "var_max": metrics["variance_max"],
        "var_cv": metrics["variance_cv"],
        "max_unit_norm_error": metrics["max_unit_norm_error"],
    }


def fit_isotropy(emb):
    x = _l2(np.asarray(emb, dtype=np.float64))
    xc = x - x.mean(0)
    cov = (xc.T @ xc) / xc.shape[0]
    w, V = np.linalg.eigh(cov)
    scale = np.maximum(w, w.max() * ISO_FLOOR_FRAC) ** ISO_POWER
    return _l2((xc @ V) * scale @ V.T).astype(np.float32)


def canonicalize_component_signs(components):
    fixed = np.asarray(components, dtype=np.float64).copy()
    pivots = np.argmax(np.abs(fixed), axis=1)
    signs = np.where(fixed[np.arange(len(fixed)), pivots] < 0, -1.0, 1.0)
    return fixed * signs[:, None]


def array_sha256(array):
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def pca_project_unit(raw, n_components=PCA_DIM):
    values = np.asarray(raw, dtype=np.float64)
    assert values.ndim == 2 and 0 < n_components < values.shape[1]
    assert np.isfinite(values).all(), "PCA input finite gate failed"
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.lexsort((np.arange(len(eigenvalues)), -eigenvalues))
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = canonicalize_component_signs(eigenvectors[:, order[:n_components]].T)
    scores = centered @ components.T
    expected_shape = (len(values), n_components)
    assert scores.shape == expected_shape, f"PCA projected shape gate failed: {scores.shape} != {expected_shape}"
    assert np.isfinite(scores).all(), "PCA projected finite gate failed"
    norms = np.linalg.norm(scores, axis=1, keepdims=True)
    min_pre_normalization_row_norm = float(norms.min())
    assert min_pre_normalization_row_norm > 0, "PCA projection zero-row gate failed"
    projected = (scores / norms).astype(np.float32)
    assert projected.shape == expected_shape, f"PCA projected shape gate failed: {projected.shape} != {expected_shape}"
    projected_finite = bool(np.isfinite(projected).all())
    assert projected_finite, "PCA projected finite gate failed"
    max_post_normalization_unit_norm_error = float(np.abs(np.linalg.norm(projected, axis=1) - 1.0).max())
    assert max_post_normalization_unit_norm_error <= 1e-5, (
        "PCA post-normalization unit-norm gate failed: "
        f"{max_post_normalization_unit_norm_error} > 1e-05"
    )
    total = float(eigenvalues.sum())
    retained = float(eigenvalues[:n_components].sum())
    discarded = float(eigenvalues[n_components:].sum())
    return projected, {
        "fit_rows": len(values),
        "input_width": values.shape[1],
        "output_width": n_components,
        "centering": "per-dimension mean subtraction",
        "solver": "numpy.linalg.eigh",
        "eigenvalue_ordering": "descending",
        "covariance_denominator": "n-1",
        "sign_rule": "lowest-index largest-absolute loading positive",
        "projected_shape": list(projected.shape),
        "projected_finite": projected_finite,
        "min_pre_normalization_row_norm": min_pre_normalization_row_norm,
        "max_post_normalization_unit_norm_error": max_post_normalization_unit_norm_error,
        "eligibility_thresholds": {
            "projected_shape": list(expected_shape),
            "projected_finite": True,
            "min_pre_normalization_row_norm_exclusive": 0.0,
            "max_post_normalization_unit_norm_error": 1e-5,
        },
        "eigenvalues_descending": eigenvalues.tolist(),
        "eigenvalues_sha256": array_sha256(eigenvalues),
        "component_sha256": array_sha256(components),
        "retained_variance": retained,
        "discarded_variance": discarded,
        "total_variance": total,
        "retained_variance_fraction": retained / total,
        "discarded_variance_fraction": discarded / total,
    }


def pca_reduce(emb, k):
    xc = np.asarray(emb, dtype=np.float64) - np.asarray(emb, dtype=np.float64).mean(0)
    values, vectors = np.linalg.eigh(xc.T @ xc / max(1, len(xc) - 1))
    order = np.lexsort((np.arange(len(values)), -values))[:k]
    components = canonicalize_component_signs(vectors[:, order].T)
    reduced = xc @ components.T
    retained = float(np.clip(values[order], 0.0, None).sum() / np.clip(values, 0.0, None).sum())
    return reduced, retained


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


class ValidationGateError(AssertionError):
    def __init__(self, gate, validation, message=None):
        self.gate = gate
        self.validation = validation
        super().__init__(message or f"{gate} gate failed")


def fino_patient_ids(path):
    payload = json.loads(Path(path).read_text())
    return {
        patient
        for group in ("discrete", "continuous")
        for mapping in payload[group].values()
        for patient in mapping
    }


def validate_candidate(reference, candidate, patient_ids, fino_ids, expected_fino_count=FINO_PATIENT_COUNT, expected_width=BIOMED_WIDTH):
    patient_set = set(np.asarray(patient_ids, dtype=str).tolist())
    numeric = np.asarray(list(candidate.values()), dtype=np.float64)
    effective_ratio = round(candidate["normalized_effective_rank"] / reference["normalized_effective_rank"], 12)
    participation_ratio = round(candidate["normalized_participation_ratio"] / reference["normalized_participation_ratio"], 12)
    missing = sorted(set(fino_ids) - patient_set)
    coverage_count = len(fino_ids) - len(missing)
    coverage_fraction = 1.0 if not fino_ids else coverage_count / len(fino_ids)
    validation = {
        "coverage_count": coverage_count,
        "coverage_total": len(fino_ids),
        "coverage_fraction": coverage_fraction,
        "missing_patient_count": len(missing),
        "missing_patient_ids": missing,
        "normalized_effective_rank_ratio": effective_ratio,
        "normalized_participation_ratio_ratio": participation_ratio,
        "gate_values": {
            "rows": candidate["rows"],
            "unique_patient_ids": len(patient_set),
            "width": candidate["width"],
            "finite_geometry": bool(np.isfinite(numeric).all()),
            "max_unit_norm_error": candidate["max_unit_norm_error"],
            "absolute_mean_off_diagonal_cosine": abs(candidate["mean_off_diagonal_cosine"]),
            "effective_rank": candidate["effective_rank"],
            "participation_ratio": candidate["participation_ratio"],
            "variance_cv": candidate["variance_cv"],
            "normalized_effective_rank_ratio": effective_ratio,
            "normalized_participation_ratio_ratio": participation_ratio,
            "fino_patient_count": len(fino_ids),
            "coverage_fraction": coverage_fraction,
        },
        "thresholds": {
            "rows": len(patient_ids),
            "width": expected_width,
            "max_unit_norm_error": MAX_ROWNORM_ERR,
            "max_absolute_mean_off_diagonal_cosine": MAX_OFFDIAG_COSINE,
            "min_effective_rank": MIN_EFFECTIVE_RANK,
            "min_participation_ratio": MIN_PARTICIPATION,
            "max_variance_cv": MAX_VAR_CV,
            "normalized_rank_ratio_range": [RATIO_LO, RATIO_HI],
            "required_fino_count": expected_fino_count,
            "required_coverage_fraction": 1.0,
        },
    }

    def require(condition, gate, message=None):
        if not condition:
            raise ValidationGateError(gate, validation, message)

    require(candidate["rows"] == len(patient_ids), "row count")
    require(len(patient_set) == len(patient_ids), "unique patient IDs")
    require(candidate["width"] == expected_width, "width")
    require(validation["gate_values"]["finite_geometry"], "finite")
    require(candidate["max_unit_norm_error"] <= MAX_ROWNORM_ERR, "unit norm")
    require(abs(candidate["mean_off_diagonal_cosine"]) <= MAX_OFFDIAG_COSINE, "cosine")
    require(candidate["effective_rank"] >= MIN_EFFECTIVE_RANK, "effective rank")
    require(candidate["participation_ratio"] >= MIN_PARTICIPATION, "participation ratio")
    require(candidate["variance_cv"] <= MAX_VAR_CV, "variance CV")
    require(RATIO_LO <= effective_ratio <= RATIO_HI, "effective rank ratio")
    require(RATIO_LO <= participation_ratio <= RATIO_HI, "participation ratio ratio")
    require(len(fino_ids) == expected_fino_count, "FINO count", f"FINO count gate failed: {len(fino_ids)} != {expected_fino_count}")
    require(not missing, "FINO coverage", f"FINO coverage gate failed: {len(missing)} missing")
    return validation


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


def json_safe(obj, nonfinite, path=""):
    if isinstance(obj, dict):
        return {k: json_safe(v, nonfinite, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v, nonfinite, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.generic):
        return json_safe(obj.item(), nonfinite, path)
    if isinstance(obj, float) and not np.isfinite(obj):
        nonfinite.append(path or "<root>")
        return None
    return obj


def json_safe_payload(payload):
    non_finite_values = {}

    def convert(value, path):
        if isinstance(value, dict):
            return {k: convert(v, f"{path}.{k}" if path else str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, np.generic):
            return convert(value.item(), path)
        if isinstance(value, float) and not np.isfinite(value):
            non_finite_values[path] = "nan" if np.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"
            return None
        return value

    return convert(payload, ""), non_finite_values


def _encode_model(model, revision, captions):
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, revision=revision)
    return np.asarray(enc.encode(list(captions), batch_size=256, show_progress_bar=True), dtype=np.float64)


def encode(*args):
    if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], str):
        return _encode_model(args[0], args[1], args[2])
    if len(args) == 3:
        encoder, captions, expected_dim = args
        raw = encoder.encode(
            captions,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
        )
        raw = np.asarray(raw, dtype=np.float32)
        assert raw.shape == (len(captions), expected_dim), raw.shape
        return raw
    raise TypeError("encode expects either (model, revision, captions) or (encoder, captions, expected_dim)")


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
    deterministic_savez("/tmp/_st1.npz", d)
    deterministic_savez("/tmp/_st2.npz", d)
    assert _sha256("/tmp/_st1.npz") == _sha256("/tmp/_st2.npz")
    lowrank = rng.normal(size=(300, 5)) @ rng.normal(size=(5, 40))
    red, ret = pca_reduce(lowrank, 8)
    assert red.shape == (300, 8) and ret > 0.999
    print("selftest OK: effective-rank/participation/off-diag-cosine, isotropy unit-norm+decorrelation, PCA reduction, deterministic NPZ")


def build(canonical, biomed_out, report_out, tile_ids_path=None, target_width=BIOMED_WIDTH):
    target_width = int(target_width)
    assert _sha256(canonical) == CANON_SHA256, "canonical NPZ hash mismatch — wrong or modified source artifact"
    z = np.load(canonical, allow_pickle=False)
    order = np.argsort(z["patient_ids"].astype(str))
    ids, captions = z["patient_ids"][order], z["captions"][order]
    tile_ids = np.load(tile_ids_path)["patient_ids"] if tile_ids_path else ids
    assert np.allclose(fit_isotropy(encode(MINILM_MODEL, MINILM_REV, captions)), z["targets"][order], atol=MINILM_REPRO_ATOL), (
        "MiniLM reproduction failed — caption order or isotropy drift"
    )
    raw = encode(BIOMED_MODEL, BIOMED_REV, captions)
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
        "status": "passed" if passed else "failed",
        "published": published,
        "first_failed_gate": first_failed,
        "stale_target_cleared": stale_cleared,
        "artifact_sha256": _sha256(biomed_out) if published else None,
        "canonical_sha256": CANON_SHA256,
        "target_width": target_width,
        "pca_variance_retained": pca_retained,
        "encoders": {
            "minilm": {"model": MINILM_MODEL, "revision": MINILM_REV, "width": MINILM_WIDTH},
            "biomed": {"model": BIOMED_MODEL, "revision": BIOMED_REV, "width": BIOMED_WIDTH},
        },
        "isotropy": {"floor_frac": ISO_FLOOR_FRAC, "power": ISO_POWER},
        "geometry": {
            "biomed_raw": geom_raw,
            "biomed_corrected": geom_corr,
            "minilm_reference": {
                "norm_effective_rank": REF_NORM_EFFRANK,
                "norm_participation_ratio": REF_NORM_PARTICIPATION,
            },
        },
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
