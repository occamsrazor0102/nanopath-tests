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
ISOTROPY_FLOOR = 0.05
ISOTROPY_POWER = 0.1
CANONICAL_SHA256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
CANONICAL_ROWS = 11_428
FINO_PATIENT_COUNT = 9_389


@dataclass(frozen=True)
class EncoderBinding:
    encoder: object
    model: str
    revision: str


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


def validate_candidate(reference, candidate, patient_ids, fino_ids, expected_fino_count=FINO_PATIENT_COUNT):
    patient_set = set(np.asarray(patient_ids, dtype=str).tolist())
    numeric = np.asarray(list(candidate.values()), dtype=np.float64)
    assert candidate["rows"] == len(patient_ids), "row count gate failed"
    assert len(patient_set) == len(patient_ids), "unique patient IDs gate failed"
    assert candidate["width"] == BIOMED_DIM, "width gate failed"
    assert np.isfinite(numeric).all(), "finite geometry gate failed"
    assert candidate["max_unit_norm_error"] <= 1e-5, "unit norm gate failed"
    assert abs(candidate["mean_off_diagonal_cosine"]) <= 0.01, "cosine gate failed"
    assert candidate["effective_rank"] >= 32, "effective rank gate failed"
    assert candidate["participation_ratio"] >= 16, "participation ratio gate failed"
    assert candidate["variance_cv"] <= 0.75, "variance CV gate failed"
    effective_ratio = round(candidate["normalized_effective_rank"] / reference["normalized_effective_rank"], 12)
    participation_ratio = round(candidate["normalized_participation_ratio"] / reference["normalized_participation_ratio"], 12)
    assert 0.5 <= effective_ratio <= 2.0, "effective rank ratio gate failed"
    assert 0.5 <= participation_ratio <= 2.0, "participation ratio ratio gate failed"
    assert len(fino_ids) == expected_fino_count, f"FINO count gate failed: {len(fino_ids)} != {expected_fino_count}"
    missing = sorted(set(fino_ids) - patient_set)
    assert not missing, f"FINO coverage gate failed: {len(missing)} missing"
    return {
        "coverage_count": len(fino_ids) - len(missing),
        "coverage_total": len(fino_ids),
        "coverage_fraction": 1.0 if not fino_ids else (len(fino_ids) - len(missing)) / len(fino_ids),
        "missing_patient_count": len(missing),
        "missing_patient_ids": missing,
        "normalized_effective_rank_ratio": effective_ratio,
        "normalized_participation_ratio_ratio": participation_ratio,
        "thresholds": {
            "rows": len(patient_ids),
            "width": BIOMED_DIM,
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


def build_reembedded_bank(
    source,
    output,
    report,
    fino_path,
    minilm_binding,
    biomedical_binding,
    expected_source_sha,
    expected_fino_count=FINO_PATIENT_COUNT,
):
    source, output, report = Path(source), Path(output), Path(report)
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
    assert isinstance(minilm_binding, EncoderBinding), "MiniLM binding provenance gate failed"
    assert isinstance(biomedical_binding, EncoderBinding), "biomedical binding provenance gate failed"
    assert minilm_binding.model == MINILM_MODEL, "MiniLM model provenance gate failed"
    assert minilm_binding.revision == MINILM_REVISION, "MiniLM revision provenance gate failed"
    assert biomedical_binding.model == BIOMED_MODEL, "biomedical model provenance gate failed"
    assert biomedical_binding.revision == BIOMED_REVISION, "biomedical revision provenance gate failed"

    minilm_raw = encode(minilm_binding.encoder, captions, canonical_targets.shape[1])
    biomedical_raw = encode(biomedical_binding.encoder, captions, BIOMED_DIM)
    minilm_targets = isotropize(minilm_raw)
    biomedical_targets = isotropize(biomedical_raw)
    np.testing.assert_allclose(
        minilm_targets,
        canonical_targets,
        atol=2e-5,
        rtol=0,
        err_msg="MiniLM regeneration gate failed",
    )
    reference = geometry_metrics(canonical_targets)
    candidate = geometry_metrics(biomedical_targets)
    validation = validate_candidate(
        reference, candidate, patient_ids, fino_patient_ids(fino_path), expected_fino_count=expected_fino_count
    )

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
        save_target_bank(output_tmp, patient_ids, biomedical_targets, captions, "biomedical")
        save_target_bank(output_check, patient_ids, biomedical_targets, captions, "biomedical")
        output_sha = hashlib.sha256(output_tmp.read_bytes()).hexdigest()
        check_sha = hashlib.sha256(output_check.read_bytes()).hexdigest()
        assert output_sha == check_sha, "deterministic NPZ gate failed"
        payload = {
            "source": {"sha256": source_sha, "rows": len(patient_ids), "mode": mode.item()},
            "artifact": {"sha256": output_sha, "rows": len(patient_ids), "width": BIOMED_DIM, "mode": "biomedical"},
            "models": {
                "minilm": {
                    "model": minilm_binding.model,
                    "revision": minilm_binding.revision,
                    "raw_geometry": geometry_metrics(minilm_raw),
                    "corrected_geometry": geometry_metrics(minilm_targets),
                },
                "biomedical": {
                    "model": biomedical_binding.model,
                    "revision": biomedical_binding.revision,
                    "raw_geometry": geometry_metrics(biomedical_raw),
                    "corrected_geometry": candidate,
                },
            },
            "validation": validation,
        }
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pairs = [argument.split("=", 1) for argument in argv]
    assert all(len(pair) == 2 for pair in pairs), "arguments must be key=value"
    args = dict(pairs)
    assert len(args) == len(pairs) and set(args) == {"source", "output", "report", "fino", "device"}, "required keys: source output report fino device"
    from sentence_transformers import SentenceTransformer

    minilm = EncoderBinding(
        SentenceTransformer(MINILM_MODEL, revision=MINILM_REVISION, device=args["device"], local_files_only=True),
        MINILM_MODEL,
        MINILM_REVISION,
    )
    biomedical = EncoderBinding(
        SentenceTransformer(BIOMED_MODEL, revision=BIOMED_REVISION, device=args["device"], local_files_only=True),
        BIOMED_MODEL,
        BIOMED_REVISION,
    )
    result = build_reembedded_bank(
        args["source"],
        args["output"],
        args["report"],
        args["fino"],
        minilm,
        biomedical,
        CANONICAL_SHA256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
