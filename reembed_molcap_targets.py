# Re-embed the canonical MolCap caption bank with pinned text encoders.
# Geometry and provenance gates keep the biomedical target bank auditable.

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_molcap_targets import isotropize, save_target_bank


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BIOMED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
BIOMED_REVISION = "96786c7024f95c5aac7f2b9a18086c7b97b23036"
BIOMED_DIM = 768
RAW768 = "raw768"
PCA384 = "pca384"
PCA_DIM = 384
PCA_MIN_VARIANCE = 0.99
VARIANT_SPECS = {
    RAW768: {"target_width": 768, "artifact_mode": "biomedical"},
    PCA384: {"target_width": 384, "artifact_mode": "biomedical-pca384"},
}
ISOTROPY_FLOOR = 0.05
ISOTROPY_POWER = 0.1
CANONICAL_SHA256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
CANONICAL_ROWS = 11_428
FINO_PATIENT_COUNT = 9_389


@dataclass(frozen=True)
class EncoderBinding:
    model: str
    revision: str
    snapshot_path: Path


class ValidationGateError(AssertionError):
    def __init__(self, gate, validation, message=None):
        self.gate = gate
        self.validation = validation
        super().__init__(message or f"{gate} gate failed")


def json_safe_payload(payload):
    non_finite_values = {}

    def convert(value, path):
        if isinstance(value, dict):
            return {
                key: convert(item, f"{path}.{key}" if path else str(key))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [convert(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            return convert(value.item(), path)
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value):
                classification = "nan"
            elif value > 0:
                classification = "positive_infinity"
            else:
                classification = "negative_infinity"
            non_finite_values[path] = classification
            return None
        return value

    return convert(payload, ""), non_finite_values


def resolve_build_paths(source, output, report, fino_path):
    resolved = {
        "source": Path(source).resolve(),
        "output": Path(output).resolve(),
        "report": Path(report).resolve(),
        "fino": Path(fino_path).resolve(),
    }
    output_path, report_path = resolved["output"], resolved["report"]
    resolved.update(
        {
            "output_tmp": output_path.with_name(output_path.name + ".tmp"),
            "output_check": output_path.with_name(output_path.name + ".check"),
            "output_backup": output_path.with_name(output_path.name + ".bak"),
            "report_tmp": report_path.with_name(report_path.name + ".tmp"),
            "report_backup": report_path.with_name(report_path.name + ".bak"),
        }
    )
    by_path = {}
    for label, path in resolved.items():
        by_path.setdefault(path, []).append(label)
    collisions = {str(path): labels for path, labels in by_path.items() if len(labels) > 1}
    assert not collisions, f"path collision gate failed: {collisions}"
    return resolved["source"], resolved["output"], resolved["report"], resolved["fino"]


def validate_encoder_binding(binding, expected_model, expected_revision, label):
    assert isinstance(binding, EncoderBinding), f"{label} binding provenance gate failed"
    assert binding.model == expected_model, f"{label} model provenance gate failed"
    assert binding.revision == expected_revision, f"{label} revision provenance gate failed"
    snapshot = Path(binding.snapshot_path).resolve()
    expected_cache_name = f"models--{expected_model.replace('/', '--')}"
    assert snapshot.is_dir(), f"{label} snapshot path provenance gate failed"
    assert snapshot.name == expected_revision, f"{label} snapshot revision provenance gate failed"
    assert snapshot.parent.name == "snapshots", f"{label} snapshot layout provenance gate failed"
    assert snapshot.parent.parent.name == expected_cache_name, f"{label} snapshot model provenance gate failed"
    return snapshot


def load_snapshot_encoder(snapshot_path, device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(Path(snapshot_path).resolve()), device=device, local_files_only=True)


def matrix_audit(values):
    array = np.asarray(values)
    finite = np.isfinite(array)
    return {
        "rows": int(array.shape[0]),
        "width": int(array.shape[1]),
        "finite": bool(finite.all()),
        "non_finite_count": int(array.size - finite.sum()),
    }


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
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.lexsort((np.arange(len(eigenvalues)), -eigenvalues))
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = canonicalize_component_signs(eigenvectors[:, order[:n_components]].T)
    scores = centered @ components.T
    expected_shape = (len(values), n_components)
    assert scores.shape == expected_shape, (
        f"PCA projected shape gate failed: {scores.shape} != {expected_shape}"
    )
    assert np.isfinite(scores).all(), "PCA projected finite gate failed"
    norms = np.linalg.norm(scores, axis=1, keepdims=True)
    min_pre_normalization_row_norm = float(norms.min())
    assert min_pre_normalization_row_norm > 0, "PCA projection zero-row gate failed"
    projected = (scores / norms).astype(np.float32)
    assert projected.shape == expected_shape, (
        f"PCA projected shape gate failed: {projected.shape} != {expected_shape}"
    )
    projected_finite = bool(np.isfinite(projected).all())
    assert projected_finite, "PCA projected finite gate failed"
    max_post_normalization_unit_norm_error = float(
        np.abs(np.linalg.norm(projected, axis=1) - 1.0).max()
    )
    assert max_post_normalization_unit_norm_error <= 1e-5, (
        "PCA post-normalization unit-norm gate failed: "
        f"{max_post_normalization_unit_norm_error} > 1e-05"
    )
    total = float(eigenvalues.sum())
    retained = float(eigenvalues[:n_components].sum())
    discarded = float(eigenvalues[n_components:].sum())
    audit = {
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
        "mean_sha256": array_sha256(mean),
        "component_sha256": array_sha256(components),
        "retained_variance": retained,
        "discarded_variance": discarded,
        "total_variance": total,
        "retained_variance_fraction": retained / total,
        "discarded_variance_fraction": discarded / total,
        "discarded_energy_fraction": 1.0 - retained / total,
        "eigenvalue_384": float(eigenvalues[n_components - 1]),
        "eigenvalue_385": float(eigenvalues[n_components]),
        "eigengap_384_385": float(eigenvalues[n_components - 1] - eigenvalues[n_components]),
    }
    return projected, audit


def persist_validation_failure(
    error,
    source_sha,
    patient_ids,
    mode,
    output,
    report,
    model_payload,
    artifact_width,
    artifact_mode,
    extra_payload=None,
):
    preexisting_target_detected = output.exists()
    target_clear_error = None
    try:
        output.unlink(missing_ok=True)
    except OSError as clear_error:
        target_clear_error = clear_error
    failure_payload = {
        "status": "failed",
        "source": {"sha256": source_sha, "rows": len(patient_ids), "mode": mode.item()},
        "artifact": {
            "published": False,
            "preexisting_target_detected": preexisting_target_detected,
            "target_path_cleared": not output.exists(),
            "rows": len(patient_ids),
            "width": artifact_width,
            "mode": artifact_mode,
        },
        "models": model_payload,
        "validation": error.validation,
        "gate_error": {"gate": error.gate, "message": str(error)},
    }
    if extra_payload is not None:
        failure_payload.update(extra_payload)
    failure_payload, non_finite_values = json_safe_payload(failure_payload)
    if non_finite_values:
        failure_payload["non_finite_values"] = non_finite_values
    report.parent.mkdir(parents=True, exist_ok=True)
    failure_report_tmp = report.with_name(report.name + ".tmp")
    assert not failure_report_tmp.exists(), "staging artifact gate failed"
    try:
        failure_report_tmp.write_text(
            json.dumps(failure_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        failure_report_tmp.replace(report)
    finally:
        failure_report_tmp.unlink(missing_ok=True)
    if target_clear_error is not None:
        error.add_note(f"failed to clear pre-existing target path: {target_clear_error}")
        raise error from target_clear_error
    return failure_payload


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


def fino_patient_ids(path):
    payload = json.loads(Path(path).read_text())
    return {patient for group in ("discrete", "continuous") for mapping in payload[group].values() for patient in mapping}


def validate_candidate(
    reference,
    candidate,
    patient_ids,
    fino_ids,
    expected_fino_count=FINO_PATIENT_COUNT,
    expected_width=768,
):
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
            "max_unit_norm_error": 1e-5,
            "max_absolute_mean_off_diagonal_cosine": 0.01,
            "min_effective_rank": 32,
            "min_participation_ratio": 16,
            "max_variance_cv": 0.75,
            "normalized_rank_ratio_range": [0.5, 2.0],
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
    require(validation["gate_values"]["finite_geometry"], "finite geometry")
    require(candidate["max_unit_norm_error"] <= 1e-5, "unit norm")
    require(abs(candidate["mean_off_diagonal_cosine"]) <= 0.01, "cosine")
    require(candidate["effective_rank"] >= 32, "effective rank")
    require(candidate["participation_ratio"] >= 16, "participation ratio")
    require(candidate["variance_cv"] <= 0.75, "variance CV")
    require(0.5 <= effective_ratio <= 2.0, "effective rank ratio")
    require(0.5 <= participation_ratio <= 2.0, "participation ratio ratio")
    require(
        len(fino_ids) == expected_fino_count,
        "FINO count",
        f"FINO count gate failed: {len(fino_ids)} != {expected_fino_count}",
    )
    require(not missing, "FINO coverage", f"FINO coverage gate failed: {len(missing)} missing")
    return validation


def encode(encoder, captions, expected_dim):
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


def pca_staging_paths(output, report):
    output, report = Path(output).resolve(), Path(report).resolve()
    return (
        output.with_name(output.name + ".tmp"),
        output.with_name(output.name + ".check"),
        output.with_name(output.name + ".bak"),
        report.with_name(report.name + ".tmp"),
        report.with_name(report.name + ".bak"),
    )


def persist_pca_build_failure(
    error,
    source,
    output,
    report,
    fino_path,
    artifact_width,
    artifact_mode,
    preexisting_target_detected,
    preexisting_report_sha256,
):
    source = Path(source).resolve()
    output = Path(output).resolve()
    report = Path(report).resolve()
    fino_path = Path(fino_path).resolve()
    protected_paths = {source, fino_path}
    if output in protected_paths or report in protected_paths or output == report:
        error.add_note("PCA failure boundary skipped unsafe colliding artifact paths")
        return None

    existing_failure = None
    if report.is_file():
        try:
            def reject_nonstandard_constant(value):
                raise ValueError(f"non-standard JSON constant: {value}")

            report_bytes = report.read_bytes()
            candidate = json.loads(
                report_bytes.decode("utf-8"),
                parse_constant=reject_nonstandard_constant,
            )
            current_report_sha256 = hashlib.sha256(report_bytes).hexdigest()
            if (
                current_report_sha256 != preexisting_report_sha256
                and isinstance(candidate, dict)
                and candidate.get("status") == "failed"
            ):
                existing_failure = candidate
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            pass

    staging_paths = pca_staging_paths(output, report)
    cleanup_errors = {}
    for path in (output, *staging_paths):
        try:
            path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            cleanup_errors[str(path)] = str(cleanup_error)
    target_path_cleared = not output.exists()
    staging_paths_cleared = not any(path.exists() for path in staging_paths)

    exception_message = str(error)
    gate = error.gate if isinstance(error, ValidationGateError) else "PCA build/publication"
    validation = (
        error.validation
        if isinstance(error, ValidationGateError)
        else {
            "gate_values": {
                "exception_type": type(error).__name__,
                "exception_message": exception_message,
            },
            "thresholds": {"pca_build_and_publication_completed": True},
        }
    )
    source_sha = None
    try:
        if source.is_file():
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError:
        pass

    payload = existing_failure or {
        "source": {"sha256": source_sha, "rows": None, "mode": None},
        "artifact": {
            "rows": None,
            "width": artifact_width,
            "mode": artifact_mode,
        },
        "models": {},
        "validation": validation,
        "gate_error": {"gate": gate, "message": exception_message},
        "pca": None,
    }
    artifact = payload.setdefault("artifact", {})
    artifact.update(
        {
            "published": False,
            "preexisting_target_detected": bool(
                preexisting_target_detected or artifact.get("preexisting_target_detected", False)
            ),
            "target_path_cleared": target_path_cleared,
            "width": artifact_width,
            "mode": artifact_mode,
        }
    )
    payload["status"] = "failed"
    payload.setdefault("validation", validation)
    payload.setdefault("gate_error", {"gate": gate, "message": exception_message})
    payload.setdefault("pca", None)
    payload["failure_boundary"] = {
        "variant": PCA384,
        "exception_type": type(error).__name__,
        "exception_message": exception_message,
        "target_path_cleared": target_path_cleared,
        "staging_paths_cleared": staging_paths_cleared,
    }
    if cleanup_errors:
        payload["failure_boundary"]["cleanup_errors"] = cleanup_errors
    payload, non_finite_values = json_safe_payload(payload)
    if non_finite_values:
        payload.setdefault("non_finite_values", {}).update(non_finite_values)

    report.parent.mkdir(parents=True, exist_ok=True)
    failure_report_tmp = report.with_name(report.name + ".tmp")
    try:
        failure_report_tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        failure_report_tmp.replace(report)
    finally:
        failure_report_tmp.unlink(missing_ok=True)
    return payload


def _build_reembedded_bank(
    source,
    output,
    report,
    fino_path,
    minilm_binding,
    biomedical_binding,
    expected_source_sha,
    device="cpu",
    variant=RAW768,
):
    assert variant in VARIANT_SPECS, f"variant gate failed: {variant}"
    variant_spec = VARIANT_SPECS[variant]
    artifact_width = BIOMED_DIM if variant == RAW768 else variant_spec["target_width"]
    artifact_mode = variant_spec["artifact_mode"]
    pca_audit = None
    source, output, report, fino_path = resolve_build_paths(source, output, report, fino_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_source_sha is not None:
        assert source_sha == expected_source_sha.lower(), f"source SHA-256 gate failed: {source_sha}"
    with np.load(source, allow_pickle=False) as bank:
        assert set(bank.files) == {"patient_ids", "targets", "captions", "mode"}, f"canonical keys gate failed: {bank.files}"
        patient_ids = bank["patient_ids"].copy()
        canonical_targets = bank["targets"].copy()
        captions = bank["captions"].copy()
        mode = bank["mode"].copy()
    assert len(patient_ids) == CANONICAL_ROWS, f"canonical row count gate failed: {len(patient_ids)}"
    assert canonical_targets.shape[0] == len(patient_ids), "canonical target row count gate failed"
    assert captions.shape == patient_ids.shape, "canonical caption count gate failed"
    assert len(set(patient_ids.tolist())) == len(patient_ids), "canonical unique patient gate failed"
    assert mode.shape == () and mode.item() == "text", "canonical mode gate failed"
    minilm_snapshot = validate_encoder_binding(minilm_binding, MINILM_MODEL, MINILM_REVISION, "MiniLM")
    biomedical_snapshot = validate_encoder_binding(
        biomedical_binding, BIOMED_MODEL, BIOMED_REVISION, "biomedical"
    )

    minilm_encoder = load_snapshot_encoder(minilm_snapshot, device)
    biomedical_encoder = load_snapshot_encoder(biomedical_snapshot, device)
    minilm_raw = encode(minilm_encoder, captions, canonical_targets.shape[1])
    biomedical_raw = encode(biomedical_encoder, captions, BIOMED_DIM)
    minilm_raw_audit = matrix_audit(minilm_raw)
    biomedical_raw_audit = matrix_audit(biomedical_raw)
    provenance = {
        "minilm": {
            "model": minilm_binding.model,
            "revision": minilm_binding.revision,
            "snapshot_commit": minilm_snapshot.name,
        },
        "biomedical": {
            "model": biomedical_binding.model,
            "revision": biomedical_binding.revision,
            "snapshot_commit": biomedical_snapshot.name,
        },
    }
    if not (minilm_raw_audit["finite"] and biomedical_raw_audit["finite"]):
        error = ValidationGateError(
            "raw embedding finite",
            {
                "gate_values": {
                    "minilm_raw_finite": minilm_raw_audit["finite"],
                    "biomedical_raw_finite": biomedical_raw_audit["finite"],
                },
                "thresholds": {"raw_embeddings_finite": True},
            },
        )
        model_payload = {
            "minilm": provenance["minilm"]
            | {"raw_matrix_audit": minilm_raw_audit, "raw_geometry": None, "corrected_geometry": None},
            "biomedical": provenance["biomedical"]
            | {
                "raw_matrix_audit": biomedical_raw_audit,
                "raw_geometry": None,
                "corrected_geometry": None,
            },
        }
        if variant == PCA384:
            model_payload["biomedical"]["post_pca_geometry"] = None
        persist_validation_failure(
            error,
            source_sha,
            patient_ids,
            mode,
            output,
            report,
            model_payload,
            artifact_width,
            artifact_mode,
            extra_payload={"pca": pca_audit} if variant == PCA384 else None,
        )
        raise error
    biomedical_pre_isotropy = biomedical_raw
    try:
        with np.errstate(all="ignore"):
            minilm_targets = isotropize(minilm_raw)
            if variant == PCA384:
                try:
                    biomedical_pre_isotropy, pca_audit = pca_project_unit(
                        biomedical_raw, artifact_width
                    )
                except Exception as pca_error:
                    error = ValidationGateError(
                        "PCA projection",
                        {
                            "gate_values": {
                                "exception_type": type(pca_error).__name__,
                                "exception_message": str(pca_error),
                            },
                            "thresholds": {"pca_projection_completed": True},
                        },
                    )
                    model_payload = {
                        "minilm": provenance["minilm"]
                        | {
                            "raw_matrix_audit": minilm_raw_audit,
                            "raw_geometry": None,
                            "corrected_geometry": None,
                        },
                        "biomedical": provenance["biomedical"]
                        | {
                            "raw_matrix_audit": biomedical_raw_audit,
                            "raw_geometry": None,
                            "post_pca_geometry": None,
                            "corrected_geometry": None,
                        },
                    }
                    persist_validation_failure(
                        error,
                        source_sha,
                        patient_ids,
                        mode,
                        output,
                        report,
                        model_payload,
                        artifact_width,
                        artifact_mode,
                        extra_payload={"pca": pca_audit},
                    )
                    raise error from pca_error
                if pca_audit["retained_variance_fraction"] < PCA_MIN_VARIANCE:
                    error = ValidationGateError(
                        "PCA variance retention",
                        {
                            "gate_values": {
                                "retained_variance_fraction": pca_audit[
                                    "retained_variance_fraction"
                                ],
                            },
                            "thresholds": {
                                "min_retained_variance_fraction": PCA_MIN_VARIANCE,
                            },
                        },
                    )
                    model_payload = {
                        "minilm": provenance["minilm"]
                        | {
                            "raw_geometry": geometry_metrics(minilm_raw),
                            "corrected_geometry": geometry_metrics(minilm_targets),
                        },
                        "biomedical": provenance["biomedical"]
                        | {
                            "raw_geometry": geometry_metrics(biomedical_raw),
                            "post_pca_geometry": geometry_metrics(biomedical_pre_isotropy),
                            "corrected_geometry": None,
                        },
                    }
                    persist_validation_failure(
                        error,
                        source_sha,
                        patient_ids,
                        mode,
                        output,
                        report,
                        model_payload,
                        artifact_width,
                        artifact_mode,
                        extra_payload={"pca": pca_audit},
                    )
                    raise error
            biomedical_targets = isotropize(biomedical_pre_isotropy)
    except ValidationGateError:
        raise
    except Exception as isotropy_error:
        error = ValidationGateError(
            "isotropy",
            {
                "gate_values": {
                    "exception_type": type(isotropy_error).__name__,
                    "exception_message": str(isotropy_error),
                },
                "thresholds": {"isotropy_completed": True},
            },
        )
        model_payload = {
            "minilm": provenance["minilm"]
            | {"raw_matrix_audit": minilm_raw_audit, "raw_geometry": None, "corrected_geometry": None},
            "biomedical": provenance["biomedical"]
            | {
                "raw_matrix_audit": biomedical_raw_audit,
                "raw_geometry": None,
                "corrected_geometry": None,
            },
        }
        if variant == PCA384:
            model_payload["biomedical"]["post_pca_geometry"] = (
                geometry_metrics(biomedical_pre_isotropy) if pca_audit is not None else None
            )
        persist_validation_failure(
            error,
            source_sha,
            patient_ids,
            mode,
            output,
            report,
            model_payload,
            artifact_width,
            artifact_mode,
            extra_payload={"pca": pca_audit} if variant == PCA384 else None,
        )
        raise error from isotropy_error
    minilm_corrected_audit = matrix_audit(minilm_targets)
    biomedical_corrected_audit = matrix_audit(biomedical_targets)
    if not (minilm_corrected_audit["finite"] and biomedical_corrected_audit["finite"]):
        error = ValidationGateError(
            "corrected embedding finite",
            {
                "gate_values": {
                    "minilm_corrected_finite": minilm_corrected_audit["finite"],
                    "biomedical_corrected_finite": biomedical_corrected_audit["finite"],
                },
                "thresholds": {"corrected_embeddings_finite": True},
            },
        )
        model_payload = {
            "minilm": provenance["minilm"]
            | {
                "raw_matrix_audit": minilm_raw_audit,
                "corrected_matrix_audit": minilm_corrected_audit,
                "raw_geometry": geometry_metrics(minilm_raw),
                "corrected_geometry": None,
            },
            "biomedical": provenance["biomedical"]
            | {
                "raw_matrix_audit": biomedical_raw_audit,
                "corrected_matrix_audit": biomedical_corrected_audit,
                "raw_geometry": geometry_metrics(biomedical_raw),
                "corrected_geometry": None,
            },
        }
        if variant == PCA384:
            model_payload["biomedical"]["post_pca_geometry"] = geometry_metrics(
                biomedical_pre_isotropy
            )
        persist_validation_failure(
            error,
            source_sha,
            patient_ids,
            mode,
            output,
            report,
            model_payload,
            artifact_width,
            artifact_mode,
            extra_payload={"pca": pca_audit} if variant == PCA384 else None,
        )
        raise error
    np.testing.assert_allclose(
        minilm_targets,
        canonical_targets,
        atol=2e-5,
        rtol=0,
        err_msg="MiniLM regeneration gate failed",
    )
    minilm_raw_geometry = geometry_metrics(minilm_raw)
    reference = geometry_metrics(canonical_targets)
    biomedical_raw_geometry = geometry_metrics(biomedical_raw)
    candidate = geometry_metrics(biomedical_targets)
    model_payload = {
        "minilm": {
            **provenance["minilm"],
            "raw_geometry": minilm_raw_geometry,
            "corrected_geometry": reference,
        },
        "biomedical": {
            **provenance["biomedical"],
            "raw_geometry": biomedical_raw_geometry,
            "corrected_geometry": candidate,
        },
    }
    if variant == PCA384:
        model_payload["biomedical"]["post_pca_geometry"] = geometry_metrics(
            biomedical_pre_isotropy
        )
    try:
        validation = validate_candidate(
            reference,
            candidate,
            patient_ids,
            fino_patient_ids(fino_path),
            expected_fino_count=FINO_PATIENT_COUNT,
            expected_width=artifact_width,
        )
    except ValidationGateError as error:
        persist_validation_failure(
            error,
            source_sha,
            patient_ids,
            mode,
            output,
            report,
            model_payload,
            artifact_width,
            artifact_mode,
            extra_payload={"pca": pca_audit} if variant == PCA384 else None,
        )
        raise

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(output.name + ".tmp")
    output_check = output.with_name(output.name + ".check")
    report_tmp = report.with_name(report.name + ".tmp")
    output_backup = output.with_name(output.name + ".bak")
    report_backup = report.with_name(report.name + ".bak")
    transaction_paths = (output_tmp, output_check, report_tmp, output_backup, report_backup)
    assert not any(path.exists() for path in transaction_paths), "staging artifact gate failed"
    try:
        save_target_bank(output_tmp, patient_ids, biomedical_targets, captions, artifact_mode)
        save_target_bank(output_check, patient_ids, biomedical_targets, captions, artifact_mode)
        output_sha = hashlib.sha256(output_tmp.read_bytes()).hexdigest()
        check_sha = hashlib.sha256(output_check.read_bytes()).hexdigest()
        assert output_sha == check_sha, "deterministic NPZ gate failed"
        payload = {
            "status": "passed",
            "source": {"sha256": source_sha, "rows": len(patient_ids), "mode": mode.item()},
            "artifact": {
                "published": True,
                "sha256": output_sha,
                "rows": len(patient_ids),
                "width": artifact_width,
                "mode": artifact_mode,
            },
            "models": model_payload,
            "validation": validation,
        }
        if variant == PCA384:
            payload["pca"] = pca_audit
        report_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

        output_backed_up = report_backed_up = output_promoted = report_promoted = False
        try:
            if output.exists():
                output.replace(output_backup)
                output_backed_up = True
            if report.exists():
                report.replace(report_backup)
                report_backed_up = True
            output_tmp.replace(output)
            output_promoted = True
            report_tmp.replace(report)
            report_promoted = True
        except Exception:
            if output_promoted and output.exists():
                output.unlink()
            if report_promoted and report.exists():
                report.unlink()
            if output_backed_up:
                output_backup.replace(output)
            if report_backed_up:
                report_backup.replace(report)
            raise
        if output_backup.exists():
            output_backup.unlink()
        if report_backup.exists():
            report_backup.unlink()
        return payload
    finally:
        for path in (output_tmp, output_check, report_tmp):
            if path.exists():
                path.unlink()


def build_reembedded_bank(
    source,
    output,
    report,
    fino_path,
    minilm_binding,
    biomedical_binding,
    expected_source_sha,
    device="cpu",
    variant=RAW768,
):
    if variant != PCA384:
        return _build_reembedded_bank(
            source,
            output,
            report,
            fino_path,
            minilm_binding,
            biomedical_binding,
            expected_source_sha,
            device=device,
            variant=variant,
        )

    preexisting_target_detected = Path(output).resolve().exists()
    preexisting_report_sha256 = None
    report_path = Path(report).resolve()
    try:
        if report_path.is_file():
            preexisting_report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    except OSError:
        pass
    try:
        return _build_reembedded_bank(
            source,
            output,
            report,
            fino_path,
            minilm_binding,
            biomedical_binding,
            expected_source_sha,
            device=device,
            variant=variant,
        )
    except Exception as error:
        try:
            persist_pca_build_failure(
                error,
                source,
                output,
                report,
                fino_path,
                VARIANT_SPECS[PCA384]["target_width"],
                VARIANT_SPECS[PCA384]["artifact_mode"],
                preexisting_target_detected,
                preexisting_report_sha256,
            )
        except Exception as failure_boundary_error:
            error.add_note(f"PCA failure boundary failed: {failure_boundary_error}")
            raise error from failure_boundary_error
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pairs = [argument.split("=", 1) for argument in argv]
    assert all(len(pair) == 2 for pair in pairs), "arguments must be key=value"
    args = dict(pairs)
    required = {"source", "output", "report", "fino", "device"}
    assert (
        len(args) == len(pairs) and required <= set(args) <= required | {"variant"}
    ), "required keys: source output report fino device; optional key: variant"
    variant = args.get("variant", RAW768)
    assert variant in VARIANT_SPECS, f"variant gate failed: {variant}"
    from huggingface_hub import snapshot_download

    minilm_snapshot = Path(
        snapshot_download(repo_id=MINILM_MODEL, revision=MINILM_REVISION, local_files_only=True)
    ).resolve()
    biomedical_snapshot = Path(
        snapshot_download(repo_id=BIOMED_MODEL, revision=BIOMED_REVISION, local_files_only=True)
    ).resolve()
    minilm = EncoderBinding(
        MINILM_MODEL,
        MINILM_REVISION,
        minilm_snapshot,
    )
    biomedical = EncoderBinding(
        BIOMED_MODEL,
        BIOMED_REVISION,
        biomedical_snapshot,
    )
    build_kwargs = {"device": args["device"]}
    if "variant" in args:
        build_kwargs["variant"] = variant
    result = build_reembedded_bank(
        args["source"],
        args["output"],
        args["report"],
        args["fino"],
        minilm,
        biomedical,
        CANONICAL_SHA256,
        **build_kwargs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
