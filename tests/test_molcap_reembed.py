import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import reembed_molcap_targets as reembed
from build_molcap_targets import isotropize, save_target_bank
from reembed_molcap_targets import (
    BIOMED_DIM,
    BIOMED_MODEL,
    BIOMED_REVISION,
    ISOTROPY_FLOOR,
    ISOTROPY_POWER,
    MINILM_MODEL,
    MINILM_REVISION,
    geometry_metrics,
)


def test_geometry_metrics_for_orthogonal_vectors():
    targets = np.eye(4, dtype=np.float32)
    metrics = geometry_metrics(targets)
    assert metrics["rows"] == 4
    assert metrics["width"] == 4
    assert metrics["mean_off_diagonal_cosine"] == 0.0
    assert metrics["effective_rank"] == 3.0
    assert metrics["participation_ratio"] == 3.0
    assert metrics["max_unit_norm_error"] == 0.0


def test_constants_pin_models_and_shared_isotropy():
    assert MINILM_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert MINILM_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert BIOMED_MODEL == "pritamdeka/S-PubMedBert-MS-MARCO"
    assert BIOMED_REVISION == "96786c7024f95c5aac7f2b9a18086c7b97b23036"
    assert BIOMED_DIM == 768
    assert reembed.FINO_PATIENT_COUNT == 9_389
    assert ISOTROPY_FLOOR == 0.05
    assert ISOTROPY_POWER == 0.1


def test_canonicalize_component_signs_uses_lowest_largest_loading():
    components = np.array([
        [-0.5, 0.5, 0.1],
        [0.1, -0.8, 0.2],
    ], dtype=np.float64)
    fixed = reembed.canonicalize_component_signs(components)
    np.testing.assert_array_equal(fixed[0], -components[0])
    np.testing.assert_array_equal(fixed[1], -components[1])
    pivots = np.argmax(np.abs(fixed), axis=1)
    assert np.all(fixed[np.arange(len(fixed)), pivots] > 0)


def test_pca_project_unit_is_deterministic_unit_norm_and_variance_audited():
    raw = np.random.default_rng(7).normal(size=(64, 8))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    first, first_audit = reembed.pca_project_unit(raw, n_components=4)
    second, second_audit = reembed.pca_project_unit(raw, n_components=4)
    assert first.shape == (64, 4)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(first, second)
    assert first_audit == second_audit
    assert len(first_audit["eigenvalues_descending"]) == 8
    assert 0.0 <= first_audit["retained_variance_fraction"] <= 1.0
    assert first_audit["component_sha256"] == second_audit["component_sha256"]


def test_pca_project_unit_matches_independent_analytic_oracle_and_audits_frozen_gates():
    centered = np.array(
        [
            [-3.0, -2.0, -1.0],
            [-3.0, -2.0, 1.0],
            [-3.0, 2.0, -1.0],
            [-3.0, 2.0, 1.0],
            [3.0, -2.0, -1.0],
            [3.0, -2.0, 1.0],
            [3.0, 2.0, -1.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    raw = centered + np.array([11.0, -7.0, 5.0])

    projected, audit = reembed.pca_project_unit(raw, n_components=2)

    expected_projected = centered[:, :2] / np.sqrt(13.0)
    expected_eigenvalues = np.array([72.0 / 7.0, 32.0 / 7.0, 8.0 / 7.0])
    np.testing.assert_allclose(projected, expected_projected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(audit["eigenvalues_descending"], expected_eigenvalues)
    assert audit["retained_variance_fraction"] == pytest.approx(13.0 / 14.0)
    assert audit["centering"] == "per-dimension mean subtraction"
    assert audit["eigenvalue_ordering"] == "descending"
    assert audit["projected_shape"] == [8, 2]
    assert audit["projected_finite"] is True
    assert audit["min_pre_normalization_row_norm"] == pytest.approx(np.sqrt(13.0))
    assert audit["max_post_normalization_unit_norm_error"] <= 1e-5
    assert audit["discarded_variance"] == pytest.approx(8.0 / 7.0)
    assert audit["discarded_variance_fraction"] == pytest.approx(1.0 / 14.0)
    assert audit["eligibility_thresholds"] == {
        "projected_shape": [8, 2],
        "projected_finite": True,
        "min_pre_normalization_row_norm_exclusive": 0.0,
        "max_post_normalization_unit_norm_error": 1e-5,
    }


def test_pca_project_unit_rejects_wrong_projected_shape(monkeypatch):
    raw = np.random.default_rng(17).normal(size=(16, 6))
    canonicalize = reembed.canonicalize_component_signs

    def drop_component(components):
        return canonicalize(components)[:-1]

    monkeypatch.setattr(reembed, "canonicalize_component_signs", drop_component)
    with pytest.raises(AssertionError, match="PCA projected shape gate failed"):
        reembed.pca_project_unit(raw, n_components=3)


def test_pca_project_unit_rejects_nonfinite_projection(monkeypatch):
    raw = np.random.default_rng(19).normal(size=(16, 6))
    canonicalize = reembed.canonicalize_component_signs

    def inject_nonfinite_component(components):
        fixed = canonicalize(components)
        fixed[0, 0] = np.nan
        return fixed

    monkeypatch.setattr(reembed, "canonicalize_component_signs", inject_nonfinite_component)
    with pytest.raises(AssertionError, match="PCA projected finite gate failed"):
        reembed.pca_project_unit(raw, n_components=3)


def test_pca_project_unit_rejects_post_normalization_unit_norm_error(monkeypatch):
    raw = np.random.default_rng(23).normal(size=(16, 6))
    real_norm = np.linalg.norm
    calls = 0

    def inject_norm_error(values, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_norm(values, *args, **kwargs)
        return result + 1e-3 if calls == 2 else result

    monkeypatch.setattr(reembed.np.linalg, "norm", inject_norm_error)
    with pytest.raises(AssertionError, match="PCA post-normalization unit-norm gate failed"):
        reembed.pca_project_unit(raw, n_components=3)


def test_pca_zero_projection_row_is_rejected():
    raw = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (8, 1))
    with pytest.raises(AssertionError, match="zero-row"):
        reembed.pca_project_unit(raw, n_components=2)


def test_fino_patient_ids_unions_every_mapping(tmp_path):
    path = tmp_path / "fino.json"
    path.write_text(json.dumps({"discrete": {"a": {"P1": 1}}, "continuous": {"b": {"P2": 0.2}}}))
    assert reembed.fino_patient_ids(path) == {"P1", "P2"}


def valid_geometry():
    return {
        "rows": 2,
        "width": 768,
        "mean_off_diagonal_cosine": 0.0,
        "effective_rank": 40.0,
        "participation_ratio": 20.0,
        "normalized_effective_rank": 0.052,
        "normalized_participation_ratio": 0.03,
        "variance_cv": 0.3,
        "max_unit_norm_error": 1e-7,
    }


def test_validate_candidate_accepts_valid_geometry_and_coverage():
    reference = {"normalized_effective_rank": 0.10, "normalized_participation_ratio": 0.06}
    report = reembed.validate_candidate(
        reference, valid_geometry(), np.array(["P1", "P2"]), {"P1", "P2"}, expected_fino_count=2
    )
    assert report["coverage_fraction"] == 1.0
    assert report["thresholds"]["required_fino_count"] == 2
    assert report["normalized_effective_rank_ratio"] == 0.52
    assert report["normalized_participation_ratio_ratio"] == 0.5


def test_validate_candidate_failure_carries_complete_audit_values():
    reference = {"normalized_effective_rank": 0.10, "normalized_participation_ratio": 0.06}
    candidate = valid_geometry() | {"normalized_effective_rank": 0.049}

    with pytest.raises(reembed.ValidationGateError, match="effective rank ratio") as raised:
        reembed.validate_candidate(
            reference,
            candidate,
            np.array(["P1", "P2"]),
            {"P1", "P2"},
            expected_fino_count=2,
        )

    assert raised.value.gate == "effective rank ratio"
    audit = raised.value.validation
    assert audit["coverage_count"] == 2
    assert audit["coverage_total"] == 2
    assert audit["coverage_fraction"] == 1.0
    assert audit["normalized_effective_rank_ratio"] == 0.49
    assert audit["normalized_participation_ratio_ratio"] == 0.5
    assert audit["gate_values"] == {
        "rows": 2,
        "unique_patient_ids": 2,
        "width": 768,
        "finite_geometry": True,
        "max_unit_norm_error": 1e-7,
        "absolute_mean_off_diagonal_cosine": 0.0,
        "effective_rank": 40.0,
        "participation_ratio": 20.0,
        "variance_cv": 0.3,
        "normalized_effective_rank_ratio": 0.49,
        "normalized_participation_ratio_ratio": 0.5,
        "fino_patient_count": 2,
        "coverage_fraction": 1.0,
    }


def test_validate_candidate_rejects_truncated_all_present_fino_set():
    reference = {"normalized_effective_rank": 0.10, "normalized_participation_ratio": 0.06}
    with pytest.raises(AssertionError, match="FINO count gate"):
        reembed.validate_candidate(reference, valid_geometry(), np.array(["P1", "P2"]), {"P1", "P2"})


@pytest.mark.parametrize(
    ("mutation", "patient_ids", "fino_ids", "gate"),
    [
        ({"rows": 1}, np.array(["P1", "P2"]), {"P1", "P2"}, "row count"),
        ({}, np.array(["P1", "P1"]), {"P1"}, "unique patient"),
        ({"width": 767}, np.array(["P1", "P2"]), {"P1", "P2"}, "width"),
        ({"max_unit_norm_error": 2e-5}, np.array(["P1", "P2"]), {"P1", "P2"}, "unit norm"),
        ({"mean_off_diagonal_cosine": 0.02}, np.array(["P1", "P2"]), {"P1", "P2"}, "cosine"),
        ({"effective_rank": 31.0}, np.array(["P1", "P2"]), {"P1", "P2"}, "effective rank"),
        ({"participation_ratio": 15.0}, np.array(["P1", "P2"]), {"P1", "P2"}, "participation"),
        ({"variance_cv": 0.8}, np.array(["P1", "P2"]), {"P1", "P2"}, "variance CV"),
        ({"normalized_effective_rank": 0.049}, np.array(["P1", "P2"]), {"P1", "P2"}, "effective rank ratio"),
        ({"normalized_effective_rank": 0.201}, np.array(["P1", "P2"]), {"P1", "P2"}, "effective rank ratio"),
        ({"normalized_participation_ratio": 0.029}, np.array(["P1", "P2"]), {"P1", "P2"}, "participation ratio ratio"),
        ({"normalized_participation_ratio": 0.121}, np.array(["P1", "P2"]), {"P1", "P2"}, "participation ratio ratio"),
        ({"variance_cv": np.nan}, np.array(["P1", "P2"]), {"P1", "P2"}, "finite"),
        ({}, np.array(["P1", "P2"]), {"P1", "P3"}, "FINO coverage"),
    ],
)
def test_validate_candidate_rejects_each_hard_gate(mutation, patient_ids, fino_ids, gate):
    reference = {"normalized_effective_rank": 0.10, "normalized_participation_ratio": 0.06}
    candidate = valid_geometry() | mutation
    with pytest.raises(AssertionError, match=gate):
        reembed.validate_candidate(reference, candidate, patient_ids, fino_ids, expected_fino_count=len(fino_ids))


class FakeEncoder:
    def __init__(self, raw, revision):
        self.raw = raw
        self.revision = revision
        self.calls = []

    def encode(self, captions, **kwargs):
        self.calls.append({"captions": list(captions), "revision": self.revision, **kwargs})
        return self.raw.copy()


def fake_snapshot_path(tmp_path, model, revision):
    snapshot = tmp_path / "hub" / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    return snapshot


def bind_fake_encoder(case, tmp_path, encoder, model, revision, *, snapshot_revision=None):
    snapshot_revision = revision if snapshot_revision is None else snapshot_revision
    snapshot = fake_snapshot_path(tmp_path, model, snapshot_revision).resolve()
    case.encoder_registry[snapshot] = encoder
    return reembed.EncoderBinding(model, revision, snapshot)


def fake_binding(case, tmp_path, raw, model, revision, *, snapshot_revision=None):
    return bind_fake_encoder(
        case,
        tmp_path,
        FakeEncoder(raw, revision),
        model,
        revision,
        snapshot_revision=snapshot_revision,
    )


def test_encoder_binding_rejects_caller_supplied_encoder(tmp_path):
    snapshot = fake_snapshot_path(tmp_path, BIOMED_MODEL, BIOMED_REVISION)
    encoder = FakeEncoder(np.eye(2, dtype=np.float32), BIOMED_REVISION)

    with pytest.raises(TypeError):
        reembed.EncoderBinding(encoder, BIOMED_MODEL, BIOMED_REVISION, snapshot)


def test_snapshot_encoder_loader_constructs_only_from_validated_path(tmp_path, monkeypatch):
    snapshot = fake_snapshot_path(tmp_path, BIOMED_MODEL, BIOMED_REVISION).resolve()
    constructed = []

    class FakeSentenceTransformer:
        def __init__(self, model, **kwargs):
            constructed.append((model, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    loaded = reembed.load_snapshot_encoder(snapshot, "cpu")

    assert isinstance(loaded, FakeSentenceTransformer)
    assert constructed == [(str(snapshot), {"device": "cpu", "local_files_only": True})]


def make_reembed_case(tmp_path, monkeypatch):
    rows, dim = 128, 40
    patient_ids = np.array([f"P{i:03d}" for i in range(rows)])
    captions = np.array([f"caption {i}" for i in range(rows)])
    minilm_raw = np.random.default_rng(11).normal(size=(rows, dim)).astype(np.float32)
    biomedical_raw = np.random.default_rng(29).normal(size=(rows, dim)).astype(np.float32)
    minilm_raw /= np.linalg.norm(minilm_raw, axis=1, keepdims=True)
    biomedical_raw /= np.linalg.norm(biomedical_raw, axis=1, keepdims=True)
    source = tmp_path / "canonical.npz"
    save_target_bank(source, patient_ids, isotropize(minilm_raw), captions, "text")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    fino = tmp_path / "fino.json"
    fino.write_text(json.dumps({"discrete": {"all": dict.fromkeys(patient_ids.tolist(), 1)}, "continuous": {}}))
    monkeypatch.setattr(reembed, "CANONICAL_ROWS", rows)
    monkeypatch.setattr(reembed, "BIOMED_DIM", dim)
    monkeypatch.setattr(reembed, "FINO_PATIENT_COUNT", rows)
    encoder_registry, load_calls = {}, []

    def load_snapshot_encoder(snapshot_path, device):
        snapshot_path = Path(snapshot_path).resolve()
        load_calls.append((snapshot_path, device))
        return encoder_registry[snapshot_path]

    monkeypatch.setattr(reembed, "load_snapshot_encoder", load_snapshot_encoder)
    return types.SimpleNamespace(
        rows=rows,
        dim=dim,
        patient_ids=patient_ids,
        captions=captions,
        minilm_raw=minilm_raw,
        biomedical_raw=biomedical_raw,
        source=source,
        source_sha=source_sha,
        fino=fino,
        encoder_registry=encoder_registry,
        load_calls=load_calls,
    )


def test_default_and_explicit_raw768_builds_are_byte_identical(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    default_output, default_report = tmp_path / "default.npz", tmp_path / "default.json"
    explicit_output, explicit_report = tmp_path / "explicit.npz", tmp_path / "explicit.json"
    default_payload = reembed.build_reembedded_bank(
        case.source, default_output, default_report, case.fino,
        minilm, biomedical, case.source_sha,
    )
    explicit_payload = reembed.build_reembedded_bank(
        case.source, explicit_output, explicit_report, case.fino,
        minilm, biomedical, case.source_sha, variant=reembed.RAW768,
    )
    assert default_output.read_bytes() == explicit_output.read_bytes()
    assert default_report.read_bytes() == explicit_report.read_bytes()
    assert default_payload == explicit_payload


def build_pca_fixture(case, tmp_path, monkeypatch, output, report):
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    monkeypatch.setattr(reembed, "PCA_MIN_VARIANCE", 0.0)
    monkeypatch.setattr(
        reembed,
        "validate_candidate",
        lambda *args, **kwargs: {"coverage_count": case.rows, "coverage_total": case.rows},
    )
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    return reembed.build_reembedded_bank(
        case.source, output, report, case.fino,
        minilm, biomedical, case.source_sha, variant=reembed.PCA384,
    )


def test_pca384_build_projects_normalizes_then_isotropizes(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "pca.npz", tmp_path / "pca.json"
    payload = build_pca_fixture(case, tmp_path, monkeypatch, output, report)
    with np.load(output, allow_pickle=False) as bank:
        assert bank["targets"].shape == (case.rows, 4)
        assert bank["mode"].item() == "biomedical-pca384"
    assert payload["artifact"]["width"] == 4
    assert payload["pca"]["output_width"] == 4
    assert payload["models"]["biomedical"]["post_pca_geometry"]["width"] == 4


def test_pca_variance_failure_clears_stale_384_target_and_reports(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "stale-pca.npz", tmp_path / "stale-pca.json"
    output.write_bytes(b"stale")
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    monkeypatch.setattr(reembed, "PCA_MIN_VARIANCE", 1.01)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    with pytest.raises(reembed.ValidationGateError, match="PCA variance retention"):
        reembed.build_reembedded_bank(
            case.source, output, report, case.fino,
            minilm, biomedical, case.source_sha, variant=reembed.PCA384,
        )
    assert not output.exists()
    failure = json.loads(report.read_text())
    assert failure["status"] == "failed"
    assert failure["artifact"]["width"] == 4
    assert failure["gate_error"]["gate"] == "PCA variance retention"


def test_pca_projection_failure_is_reported_as_pca_gate(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "failed-pca.npz", tmp_path / "failed-pca.json"
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    constant_raw = np.tile(case.biomedical_raw[0], (case.rows, 1))
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, constant_raw, BIOMED_MODEL, BIOMED_REVISION)
    with pytest.raises(reembed.ValidationGateError, match="PCA projection gate failed"):
        reembed.build_reembedded_bank(
            case.source, output, report, case.fino,
            minilm, biomedical, case.source_sha, variant=reembed.PCA384,
        )
    failure = json.loads(report.read_text())
    assert failure["gate_error"]["gate"] == "PCA projection"
    assert failure["artifact"]["width"] == 4
    assert failure["pca"] is None
    assert not output.exists()


def test_pca_variant_orders_projection_before_biomedical_isotropy(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    events = []
    original_isotropize = reembed.isotropize
    original_pca = reembed.pca_project_unit

    def record_isotropize(values):
        events.append(("isotropize", values.shape[1]))
        return original_isotropize(values)

    def record_pca(values, n_components):
        events.append(("pca", values.shape[1]))
        return original_pca(values, n_components)

    monkeypatch.setattr(reembed, "isotropize", record_isotropize)
    monkeypatch.setattr(reembed, "pca_project_unit", record_pca)
    build_pca_fixture(case, tmp_path, monkeypatch, tmp_path / "order.npz", tmp_path / "order.json")
    assert events == [("isotropize", case.dim), ("pca", case.dim), ("isotropize", 4)]


def test_pca_success_artifacts_are_byte_deterministic(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    first, first_report = tmp_path / "first-pca.npz", tmp_path / "first-pca.json"
    second, second_report = tmp_path / "second-pca.npz", tmp_path / "second-pca.json"
    build_pca_fixture(case, tmp_path, monkeypatch, first, first_report)
    build_pca_fixture(case, tmp_path, monkeypatch, second, second_report)
    assert first.read_bytes() == second.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()


def pca_staging_paths(output, report):
    return (
        output.with_name(output.name + ".tmp"),
        output.with_name(output.name + ".check"),
        output.with_name(output.name + ".bak"),
        report.with_name(report.name + ".tmp"),
        report.with_name(report.name + ".bak"),
    )


def seed_stale_pca_artifacts(output, report, *, staging=False):
    output.write_bytes(b"stale PCA target")
    report.write_text('{"status":"passed","artifact":{"published":true}}\n')
    if staging:
        for path in pca_staging_paths(output, report):
            path.write_bytes(b"stale PCA staging artifact")


def assert_pca_failure_boundary(output, report, *, exception_type, message_fragment):
    assert not output.exists()
    assert not any(path.exists() for path in pca_staging_paths(output, report))

    def reject_nonstandard_constant(value):
        raise AssertionError(f"non-standard JSON constant: {value}")

    payload = json.loads(report.read_text(), parse_constant=reject_nonstandard_constant)
    assert payload["status"] == "failed"
    assert payload["artifact"]["published"] is False
    assert payload["artifact"]["preexisting_target_detected"] is True
    assert payload["artifact"]["target_path_cleared"] is True
    assert message_fragment in payload["gate_error"]["message"]
    assert payload["failure_boundary"] == {
        "variant": reembed.PCA384,
        "exception_type": exception_type,
        "exception_message": payload["failure_boundary"]["exception_message"],
        "target_path_cleared": True,
        "staging_paths_cleared": True,
    }
    assert message_fragment in payload["failure_boundary"]["exception_message"]


def test_pca_replay_failure_replaces_stale_report_and_clears_all_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    monkeypatch.setattr(reembed, "PCA_MIN_VARIANCE", 0.0)
    output, report = tmp_path / "replay.npz", tmp_path / "replay.json"
    seed_stale_pca_artifacts(output, report, staging=True)
    mismatched_minilm = np.roll(case.minilm_raw, 1, axis=0)
    minilm = fake_binding(case, tmp_path, mismatched_minilm, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)

    with pytest.raises(AssertionError, match="MiniLM regeneration"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
            variant=reembed.PCA384,
        )

    assert_pca_failure_boundary(
        output,
        report,
        exception_type="AssertionError",
        message_fragment="MiniLM regeneration gate failed",
    )


def test_pca_staging_gate_failure_replaces_stale_report_and_clears_all_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "staging.npz", tmp_path / "staging.json"
    seed_stale_pca_artifacts(output, report, staging=True)

    with pytest.raises(AssertionError, match="staging artifact gate failed"):
        build_pca_fixture(case, tmp_path, monkeypatch, output, report)

    assert_pca_failure_boundary(
        output,
        report,
        exception_type="AssertionError",
        message_fragment="staging artifact gate failed",
    )


def test_pca_serialization_failure_replaces_stale_report_and_clears_all_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "serialization.npz", tmp_path / "serialization.json"
    seed_stale_pca_artifacts(output, report)

    def fail_serialization(*args, **kwargs):
        raise OSError("injected PCA serialization failure")

    monkeypatch.setattr(reembed, "save_target_bank", fail_serialization)
    with pytest.raises(OSError, match="injected PCA serialization failure"):
        build_pca_fixture(case, tmp_path, monkeypatch, output, report)

    assert_pca_failure_boundary(
        output,
        report,
        exception_type="OSError",
        message_fragment="injected PCA serialization failure",
    )


def test_pca_deterministic_npz_failure_replaces_stale_report_and_clears_all_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "determinism.npz", tmp_path / "determinism.json"
    seed_stale_pca_artifacts(output, report)
    deterministic_save = reembed.save_target_bank

    def mismatched_save(path, *args):
        deterministic_save(path, *args)
        if Path(path).name.endswith(".check"):
            Path(path).write_bytes(Path(path).read_bytes() + b"mismatch")

    monkeypatch.setattr(reembed, "save_target_bank", mismatched_save)
    with pytest.raises(AssertionError, match="deterministic NPZ gate failed"):
        build_pca_fixture(case, tmp_path, monkeypatch, output, report)

    assert_pca_failure_boundary(
        output,
        report,
        exception_type="AssertionError",
        message_fragment="deterministic NPZ gate failed",
    )


def test_pca_source_failure_replaces_stale_report_and_clears_all_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    output, report = tmp_path / "source-failure.npz", tmp_path / "source-failure.json"
    seed_stale_pca_artifacts(output, report, staging=True)

    with pytest.raises(AssertionError, match="source SHA-256 gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            None,
            None,
            "0" * 64,
            variant=reembed.PCA384,
        )

    assert_pca_failure_boundary(
        output,
        report,
        exception_type="AssertionError",
        message_fragment="source SHA-256 gate failed",
    )


def test_pca_source_failure_does_not_reuse_a_stale_failed_report(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    output, report = tmp_path / "stale-failed-source.npz", tmp_path / "stale-failed-source.json"
    seed_stale_pca_artifacts(output, report, staging=True)
    report.write_text(
        json.dumps(
            {
                "status": "failed",
                "artifact": {"published": False},
                "gate_error": {"gate": "stale gate", "message": "stale failure evidence"},
            }
        )
        + "\n"
    )

    with pytest.raises(AssertionError, match="source SHA-256 gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            None,
            None,
            "0" * 64,
            variant=reembed.PCA384,
        )

    assert_pca_failure_boundary(
        output,
        report,
        exception_type="AssertionError",
        message_fragment="source SHA-256 gate failed",
    )
    payload = json.loads(report.read_text())
    assert payload["gate_error"]["gate"] != "stale gate"


def test_pca_unknown_entry_report_fingerprint_never_reuses_stale_failed_json(
    tmp_path,
    monkeypatch,
):
    case = make_reembed_case(tmp_path, monkeypatch)
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    output, report = tmp_path / "unknown-fingerprint.npz", tmp_path / "unknown-fingerprint.json"
    seed_stale_pca_artifacts(output, report, staging=True)
    report.write_text(
        json.dumps(
            {
                "status": "failed",
                "artifact": {"published": False},
                "gate_error": {"gate": "stale gate", "message": "stale failure evidence"},
            }
        )
        + "\n"
    )
    original_read_bytes = Path.read_bytes
    entry_read_failed = False

    def fail_entry_report_read_once(self):
        nonlocal entry_read_failed
        if self == report.resolve() and not entry_read_failed:
            entry_read_failed = True
            raise OSError("injected transient entry report read failure")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fail_entry_report_read_once)
    with pytest.raises(AssertionError, match="source SHA-256 gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            None,
            None,
            "0" * 64,
            variant=reembed.PCA384,
        )

    assert entry_read_failed is True
    payload = json.loads(report.read_text())
    assert payload["gate_error"]["gate"] != "stale gate"
    assert "source SHA-256 gate failed" in payload["gate_error"]["message"]


def test_pca_provenance_failure_replaces_stale_report_and_clears_all_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    output, report = tmp_path / "provenance.npz", tmp_path / "provenance.json"
    seed_stale_pca_artifacts(output, report, staging=True)
    minilm = reembed.EncoderBinding(
        MINILM_MODEL,
        "wrong-revision",
        fake_snapshot_path(tmp_path, MINILM_MODEL, "wrong-revision"),
    )
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)

    with pytest.raises(AssertionError, match="MiniLM revision provenance gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
            variant=reembed.PCA384,
        )

    assert_pca_failure_boundary(
        output,
        report,
        exception_type="AssertionError",
        message_fragment="MiniLM revision provenance gate failed",
    )


def test_pca_publication_failure_replaces_stale_report_and_clears_all_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "publication.npz", tmp_path / "publication.json"
    seed_stale_pca_artifacts(output, report)
    original_replace = Path.replace
    injected = False

    def fail_first_report_promotion(self, target):
        nonlocal injected
        if not injected and self == report.with_name(report.name + ".tmp") and Path(target) == report:
            injected = True
            raise OSError("injected PCA report publication failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_first_report_promotion)
    with pytest.raises(OSError, match="injected PCA report publication failure"):
        build_pca_fixture(case, tmp_path, monkeypatch, output, report)

    assert injected is True
    assert_pca_failure_boundary(
        output,
        report,
        exception_type="OSError",
        message_fragment="injected PCA report publication failure",
    )


def test_build_reembedded_bank_copies_canonical_rows_and_writes_deterministically(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = FakeEncoder(case.minilm_raw, MINILM_REVISION)
    biomedical = FakeEncoder(case.biomedical_raw, BIOMED_REVISION)
    minilm_binding = bind_fake_encoder(case, tmp_path, minilm, MINILM_MODEL, MINILM_REVISION)
    biomedical_binding = bind_fake_encoder(
        case, tmp_path, biomedical, BIOMED_MODEL, BIOMED_REVISION
    )
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    report, second_report = tmp_path / "first.json", tmp_path / "second.json"

    returned = reembed.build_reembedded_bank(
        case.source,
        first,
        report,
        case.fino,
        minilm_binding,
        biomedical_binding,
        case.source_sha,
    )
    reembed.build_reembedded_bank(
        case.source,
        second,
        second_report,
        case.fino,
        minilm_binding,
        biomedical_binding,
        case.source_sha,
    )

    with np.load(first, allow_pickle=False) as bank:
        first_ids = bank["patient_ids"]
        first_captions = bank["captions"]
        assert bank["mode"].item() == "biomedical"
        assert bank["targets"].shape == (case.rows, case.dim)
    assert first_ids.tolist() == case.patient_ids.tolist()
    assert first_captions.tolist() == case.captions.tolist()
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    assert report.read_bytes() == second_report.read_bytes()
    report_payload = json.loads(report.read_text())
    assert report_payload["status"] == "passed"
    assert report_payload["artifact"]["published"] is True
    assert report_payload["models"]["biomedical"]["revision"] == BIOMED_REVISION
    assert returned == json.loads(report.read_text())
    expected_call = {
        "captions": case.captions.tolist(),
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "batch_size": 64,
        "convert_to_numpy": True,
    }
    assert minilm.calls == [{**expected_call, "revision": MINILM_REVISION}] * 2
    assert biomedical.calls == [{**expected_call, "revision": BIOMED_REVISION}] * 2


def test_build_reembedded_bank_rejects_wrong_encoder_revision(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = reembed.EncoderBinding(
        MINILM_MODEL,
        "wrong",
        fake_snapshot_path(tmp_path, MINILM_MODEL, "wrong"),
    )
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    with pytest.raises(AssertionError, match="revision provenance gate"):
        reembed.build_reembedded_bank(
            case.source,
            tmp_path / "wrong.npz",
            tmp_path / "wrong.json",
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )


def test_build_reembedded_bank_rejects_bare_unannotated_encoder(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="binding provenance gate"):
        reembed.build_reembedded_bank(
            case.source,
            tmp_path / "bare.npz",
            tmp_path / "bare.json",
            case.fino,
            FakeEncoder(case.minilm_raw, MINILM_REVISION),
            FakeEncoder(case.biomedical_raw, BIOMED_REVISION),
            case.source_sha,
        )


def test_report_replace_failure_rolls_back_preexisting_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    output, report = tmp_path / "existing.npz", tmp_path / "existing.json"
    output.write_bytes(b"preexisting output")
    report.write_bytes(b"preexisting report")
    original_replace = Path.replace

    def fail_report_replace(self, target):
        if self == report.with_name(report.name + ".tmp") and Path(target) == report:
            raise OSError("injected report replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_report_replace)
    with pytest.raises(OSError, match="injected report replace failure"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )
    assert output.read_bytes() == b"preexisting output"
    assert report.read_bytes() == b"preexisting report"
    assert not [*tmp_path.glob("*.tmp"), *tmp_path.glob("*.check"), *tmp_path.glob("*.bak")]


def test_deterministic_hash_failure_cleans_staging_artifacts(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    output, report = tmp_path / "failed.npz", tmp_path / "failed.json"
    deterministic_save = reembed.save_target_bank

    def mismatched_save(path, *args):
        deterministic_save(path, *args)
        if Path(path).name.endswith(".check"):
            Path(path).write_bytes(Path(path).read_bytes() + b"mismatch")

    monkeypatch.setattr(reembed, "save_target_bank", mismatched_save)
    with pytest.raises(AssertionError, match="deterministic NPZ gate"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )
    assert not output.exists()
    assert not report.exists()
    assert not [*tmp_path.glob("*.tmp"), *tmp_path.glob("*.check"), *tmp_path.glob("*.bak")]


def test_build_reembedded_bank_rejects_wrong_hash_before_loading(tmp_path):
    source = tmp_path / "canonical.npz"
    source.write_bytes(b"not an npz")
    with pytest.raises(AssertionError, match="source SHA-256"):
        reembed.build_reembedded_bank(source, tmp_path / "out.npz", tmp_path / "report.json", tmp_path / "fino.json", None, None, "wrong")


def test_build_reembedded_bank_requires_exact_keys(tmp_path):
    source = tmp_path / "canonical.npz"
    np.savez(source, patient_ids=["P1"], targets=[[1.0]], captions=["caption"], mode="text", extra=[1])
    with pytest.raises(AssertionError, match="canonical keys"):
        reembed.build_reembedded_bank(source, tmp_path / "out.npz", tmp_path / "report.json", tmp_path / "fino.json", None, None, None)


def test_build_reembedded_bank_requires_canonical_row_count(tmp_path):
    source = tmp_path / "canonical.npz"
    save_target_bank(source, ["P1"], [[1.0]], ["caption"], "text")
    with pytest.raises(AssertionError, match="canonical row count"):
        reembed.build_reembedded_bank(source, tmp_path / "out.npz", tmp_path / "report.json", tmp_path / "fino.json", None, None, None)


def test_cli_accepts_only_pca384_variant(tmp_path, monkeypatch):
    build_calls = []

    def fake_snapshot_download(repo_id, revision, local_files_only):
        assert local_files_only is True
        return str(fake_snapshot_path(tmp_path, repo_id, revision))

    def fake_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setattr(reembed, "build_reembedded_bank", fake_build)
    base_argv = [
        f"source={tmp_path / 'source.npz'}",
        f"output={tmp_path / 'output.npz'}",
        f"report={tmp_path / 'report.json'}",
        f"fino={tmp_path / 'fino.json'}",
        "device=cpu",
    ]
    argv = base_argv + ["variant=pca384"]
    assert reembed.main(argv) == {"ok": True}
    assert build_calls[0][1]["variant"] == reembed.PCA384
    with pytest.raises(AssertionError, match="variant"):
        reembed.main(base_argv + ["variant=unknown"])


def test_cli_pins_revisions_and_stays_offline(tmp_path, monkeypatch):
    snapshots, build_calls = [], []

    def fake_snapshot_download(**kwargs):
        snapshots.append(kwargs)
        return str(fake_snapshot_path(tmp_path, kwargs["repo_id"], kwargs["revision"]))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))

    def fake_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(reembed, "build_reembedded_bank", fake_build)
    argv = [
        f"source={tmp_path / 'source.npz'}",
        f"output={tmp_path / 'output.npz'}",
        f"report={tmp_path / 'report.json'}",
        f"fino={tmp_path / 'fino.json'}",
        "device=cpu",
    ]

    assert reembed.main(argv) == {"ok": True}
    assert snapshots == [
        {"repo_id": MINILM_MODEL, "revision": MINILM_REVISION, "local_files_only": True},
        {"repo_id": BIOMED_MODEL, "revision": BIOMED_REVISION, "local_files_only": True},
    ]
    build_args, build_kwargs = build_calls[0]
    assert isinstance(build_args[4], reembed.EncoderBinding)
    assert (build_args[4].model, build_args[4].revision) == (MINILM_MODEL, MINILM_REVISION)
    assert isinstance(build_args[5], reembed.EncoderBinding)
    assert (build_args[5].model, build_args[5].revision) == (BIOMED_MODEL, BIOMED_REVISION)
    assert build_args[-1] == reembed.CANONICAL_SHA256
    assert build_kwargs == {"device": "cpu"}


def test_validation_failure_persists_audit_report_without_publishing_target(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    output, report = tmp_path / "failed.npz", tmp_path / "failed.json"
    audit = {
        "coverage_count": case.rows,
        "coverage_total": case.rows,
        "coverage_fraction": 1.0,
        "missing_patient_count": 0,
        "missing_patient_ids": [],
        "normalized_effective_rank_ratio": 0.4,
        "normalized_participation_ratio_ratio": 0.6,
        "thresholds": {"normalized_rank_ratio_range": [0.5, 2.0]},
        "gate_values": {"normalized_effective_rank_ratio": 0.4},
    }

    def reject_candidate(*args, **kwargs):
        raise reembed.ValidationGateError("effective rank ratio", audit)

    monkeypatch.setattr(reembed, "validate_candidate", reject_candidate)
    with pytest.raises(reembed.ValidationGateError, match="effective rank ratio gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )

    assert not output.exists()
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["gate_error"] == {
        "gate": "effective rank ratio",
        "message": "effective rank ratio gate failed",
    }
    assert payload["source"] == {"sha256": case.source_sha, "rows": case.rows, "mode": "text"}
    assert payload["artifact"] == {
        "published": False,
        "preexisting_target_detected": False,
        "target_path_cleared": True,
        "rows": case.rows,
        "width": case.dim,
        "mode": "biomedical",
    }
    assert payload["validation"] == audit
    assert set(payload["models"]["minilm"]) == {
        "model",
        "revision",
        "snapshot_commit",
        "raw_geometry",
        "corrected_geometry",
    }
    assert set(payload["models"]["biomedical"]) == {
        "model",
        "revision",
        "snapshot_commit",
        "raw_geometry",
        "corrected_geometry",
    }
    assert not [*tmp_path.glob("*.tmp"), *tmp_path.glob("*.check"), *tmp_path.glob("*.bak")]


def test_validation_failure_removes_preexisting_target_before_publishing_failed_report(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    output, report = tmp_path / "stale.npz", tmp_path / "stale.json"
    output.write_bytes(b"stale target that must not remain trainable")
    report.write_text('{"artifact":{"sha256":"stale"}}\n')

    def reject_candidate(*args, **kwargs):
        raise reembed.ValidationGateError(
            "effective rank ratio",
            {
                "gate_values": {"normalized_effective_rank_ratio": 0.4},
                "thresholds": {"normalized_rank_ratio_range": [0.5, 2.0]},
            },
        )

    monkeypatch.setattr(reembed, "validate_candidate", reject_candidate)
    with pytest.raises(reembed.ValidationGateError, match="effective rank ratio gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )

    assert not output.exists()
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["artifact"]["published"] is False
    assert payload["artifact"]["preexisting_target_detected"] is True
    assert payload["artifact"]["target_path_cleared"] is True


def test_finite_geometry_failure_persists_json_safe_nonfinite_audit(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    output, report = tmp_path / "nonfinite.npz", tmp_path / "nonfinite.json"
    persist_failure = reembed.persist_validation_failure

    def persist_with_nonfinite_pca(*args, **kwargs):
        kwargs["extra_payload"] = {"pca": {"retained_variance_fraction": np.nan}}
        return persist_failure(*args, **kwargs)

    def reject_candidate(*args, **kwargs):
        raise reembed.ValidationGateError(
            "finite geometry",
            {
                "gate_values": {"finite_geometry": False, "variance_cv": np.nan},
                "thresholds": {"finite_geometry": True},
            },
        )

    monkeypatch.setattr(reembed, "validate_candidate", reject_candidate)
    monkeypatch.setattr(reembed, "persist_validation_failure", persist_with_nonfinite_pca)
    with pytest.raises(reembed.ValidationGateError, match="finite geometry gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )

    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["gate_error"]["gate"] == "finite geometry"
    assert payload["validation"]["gate_values"]["variance_cv"] is None
    assert payload["non_finite_values"]["validation.gate_values.variance_cv"] == "nan"
    assert payload["pca"]["retained_variance_fraction"] is None
    assert payload["non_finite_values"]["pca.retained_variance_fraction"] == "nan"
    assert not output.exists()


@pytest.mark.parametrize("aliased_output", ["source", "fino", "report"])
def test_build_reembedded_bank_rejects_input_output_path_aliases(tmp_path, aliased_output):
    source = tmp_path / "source.npz"
    fino = tmp_path / "fino.json"
    report = tmp_path / "report.json"
    output = tmp_path / "output.npz"
    source.write_bytes(b"canonical bytes must survive")
    fino.write_text("{}")
    original_source = source.read_bytes()
    paths = {"source": source, "fino": fino, "report": report}
    output = paths[aliased_output]

    with pytest.raises(AssertionError, match="path collision gate"):
        reembed.build_reembedded_bank(source, output, report, fino, None, None, None)

    assert source.read_bytes() == original_source


def test_build_reembedded_bank_rejects_report_alias_with_source(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    fino = tmp_path / "fino.json"
    source.write_bytes(b"canonical bytes must survive")
    fino.write_text("{}")
    original_source = source.read_bytes()

    with pytest.raises(AssertionError, match="path collision gate"):
        reembed.build_reembedded_bank(source, output, source, fino, None, None, None)

    assert source.read_bytes() == original_source


@pytest.mark.parametrize("input_label", ["source", "fino"])
@pytest.mark.parametrize(
    "staging_label",
    ["output_tmp", "output_check", "output_backup", "report_tmp", "report_backup"],
)
def test_pca_path_collision_never_deletes_inputs_aliased_to_staging(
    tmp_path,
    input_label,
    staging_label,
):
    output = tmp_path / "candidate.npz"
    report = tmp_path / "candidate.geometry.json"
    staging = {
        "output_tmp": output.with_name(output.name + ".tmp"),
        "output_check": output.with_name(output.name + ".check"),
        "output_backup": output.with_name(output.name + ".bak"),
        "report_tmp": report.with_name(report.name + ".tmp"),
        "report_backup": report.with_name(report.name + ".bak"),
    }
    source = tmp_path / "canonical.npz"
    fino = tmp_path / "fino.json"
    if input_label == "source":
        source = staging[staging_label]
    else:
        fino = staging[staging_label]

    source.write_bytes(b"canonical input bytes must survive")
    fino.write_bytes(b"FINO input bytes must survive")
    output.write_bytes(b"stale PCA target")
    report.write_bytes(b"stale PCA report")
    source_bytes = source.read_bytes()
    fino_bytes = fino.read_bytes()
    output_bytes = output.read_bytes()
    report_bytes = report.read_bytes()

    with pytest.raises(AssertionError, match="path collision gate failed") as raised:
        reembed.build_reembedded_bank(
            source,
            output,
            report,
            fino,
            None,
            None,
            None,
            variant=reembed.PCA384,
        )

    assert source.read_bytes() == source_bytes
    assert fino.read_bytes() == fino_bytes
    assert output.read_bytes() == output_bytes
    assert report.read_bytes() == report_bytes
    notes = "\n".join(getattr(raised.value, "__notes__", []))
    assert "PCA failure report not written" in notes
    assert input_label in notes
    assert staging_label in notes


@pytest.mark.parametrize("input_label", ["source", "fino"])
def test_pca_hardlinked_report_tmp_unlink_failure_preserves_input_and_report(
    tmp_path,
    monkeypatch,
    input_label,
):
    source = tmp_path / "canonical.npz"
    fino = tmp_path / "fino.json"
    output = tmp_path / "candidate.npz"
    report = tmp_path / "candidate.geometry.json"
    report_tmp = report.with_name(report.name + ".tmp")
    source.write_bytes(b"canonical input bytes must survive exactly")
    fino.write_bytes(b"FINO input bytes must survive exactly")
    output.write_bytes(b"stale PCA target must survive unsafe entry")
    report.write_bytes(b"stale PCA report must survive unsafe entry")
    protected_input = {"source": source, "fino": fino}[input_label]
    os.link(protected_input, report_tmp)
    before = {path: path.read_bytes() for path in (source, fino, output, report, report_tmp)}
    original_unlink = Path.unlink

    def fail_aliased_unlink(self, *args, **kwargs):
        if self == report_tmp:
            raise OSError("injected report-temp unlink failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_aliased_unlink)
    with pytest.raises(AssertionError, match="path collision gate failed") as raised:
        reembed.build_reembedded_bank(
            source,
            output,
            report,
            fino,
            None,
            None,
            None,
            variant=reembed.PCA384,
        )

    assert {path: path.read_bytes() for path in before} == before
    assert "PCA failure report not written" in "\n".join(getattr(raised.value, "__notes__", []))


def _seed_pca_publication_collision(tmp_path, collision_kind):
    source = tmp_path / "canonical.npz"
    fino = tmp_path / "fino.json"
    output = tmp_path / "candidate.npz"
    report = tmp_path / "candidate.geometry.json"
    source.write_bytes(b"canonical source")
    fino.write_bytes(b"canonical FINO")
    output.write_bytes(b"stale target")
    output_staging = {
        "output_tmp": output.with_name(output.name + ".tmp"),
        "output_check": output.with_name(output.name + ".check"),
        "output_backup": output.with_name(output.name + ".bak"),
    }
    if collision_kind == "pathname_output_report":
        report = output
    elif collision_kind.startswith("pathname_report_"):
        report = output_staging[collision_kind.removeprefix("pathname_report_")]
        report.write_bytes(b"stale report at output staging path")
    else:
        report.write_bytes(b"stale report")
        if collision_kind == "hardlink_output_report":
            report.unlink()
            os.link(output, report)
        elif collision_kind.startswith("hardlink_report_tmp_"):
            staging = output_staging[collision_kind.removeprefix("hardlink_report_tmp_")]
            staging.write_bytes(b"stale output staging artifact")
            os.link(staging, report.with_name(report.name + ".tmp"))
        else:
            staging = output_staging[collision_kind.removeprefix("hardlink_report_")]
            os.link(report, staging)
    paths = {
        source,
        fino,
        output,
        report,
        report.with_name(report.name + ".tmp"),
        report.with_name(report.name + ".bak"),
        *output_staging.values(),
    }
    existing = {path: path.read_bytes() for path in paths if path.exists()}
    return source, fino, output, report, existing


@pytest.mark.parametrize(
    "collision_kind",
    [
        "pathname_output_report",
        "pathname_report_output_tmp",
        "pathname_report_output_check",
        "pathname_report_output_backup",
        "hardlink_output_report",
        "hardlink_report_output_tmp",
        "hardlink_report_output_check",
        "hardlink_report_output_backup",
        "hardlink_report_tmp_output_tmp",
        "hardlink_report_tmp_output_check",
        "hardlink_report_tmp_output_backup",
    ],
)
@pytest.mark.parametrize("unlink_would_fail", [False, True])
def test_pca_publication_family_collisions_fail_closed_without_mutation(
    tmp_path,
    monkeypatch,
    collision_kind,
    unlink_would_fail,
):
    source, fino, output, report, before = _seed_pca_publication_collision(
        tmp_path,
        collision_kind,
    )
    unlink_calls = []
    original_unlink = Path.unlink

    def record_or_fail_unlink(self, *args, **kwargs):
        unlink_calls.append(self)
        if unlink_would_fail:
            raise OSError("injected cleanup failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", record_or_fail_unlink)
    with pytest.raises(AssertionError, match="path collision gate failed") as raised:
        reembed.build_reembedded_bank(
            source,
            output,
            report,
            fino,
            None,
            None,
            None,
            variant=reembed.PCA384,
        )

    assert unlink_calls == []
    assert {path: path.read_bytes() for path in before} == before
    assert "PCA failure report not written" in "\n".join(getattr(raised.value, "__notes__", []))


def test_pca_path_identity_uncertainty_fails_closed_without_mutation(tmp_path, monkeypatch):
    source = tmp_path / "canonical.npz"
    fino = tmp_path / "fino.json"
    output = tmp_path / "candidate.npz"
    report = tmp_path / "candidate.geometry.json"
    for path, contents in (
        (source, b"canonical source"),
        (fino, b"canonical FINO"),
        (output, b"stale target"),
        (report, b"stale report"),
    ):
        path.write_bytes(contents)
    before = {path: path.read_bytes() for path in (source, fino, output, report)}
    original_samefile = Path.samefile

    def uncertain_samefile(self, other):
        if {self, Path(other)} == {source.resolve(), output.resolve()}:
            raise OSError("injected identity uncertainty")
        return original_samefile(self, other)

    monkeypatch.setattr(Path, "samefile", uncertain_samefile)
    with pytest.raises(AssertionError, match="path identity gate failed") as raised:
        reembed.build_reembedded_bank(
            source,
            output,
            report,
            fino,
            None,
            None,
            None,
            variant=reembed.PCA384,
        )

    assert {path: path.read_bytes() for path in before} == before
    assert "PCA failure report not written" in "\n".join(getattr(raised.value, "__notes__", []))


def test_build_reembedded_bank_rejects_binding_from_wrong_snapshot_revision(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(
        case,
        tmp_path,
        case.biomedical_raw,
        BIOMED_MODEL,
        BIOMED_REVISION,
        snapshot_revision="wrong-commit",
    )

    with pytest.raises(AssertionError, match="biomedical snapshot revision provenance gate"):
        reembed.build_reembedded_bank(
            case.source,
            tmp_path / "wrong-snapshot.npz",
            tmp_path / "wrong-snapshot.json",
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )


def test_nonfinite_raw_embeddings_persist_failed_audit_before_isotropy(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    biomedical_raw = case.biomedical_raw.copy()
    biomedical_raw[0, 0] = np.nan
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    output, report = tmp_path / "nonfinite-raw.npz", tmp_path / "nonfinite-raw.json"

    with pytest.raises(reembed.ValidationGateError, match="raw embedding finite gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )

    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["gate_error"]["gate"] == "raw embedding finite"
    assert payload["models"]["biomedical"]["raw_matrix_audit"]["finite"] is False
    assert payload["models"]["biomedical"]["raw_matrix_audit"]["non_finite_count"] == 1
    assert payload["models"]["biomedical"]["corrected_geometry"] is None
    assert not output.exists()


def test_nonfinite_isotropy_output_persists_failed_audit(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(
        case,
        tmp_path,
        np.zeros_like(case.biomedical_raw),
        BIOMED_MODEL,
        BIOMED_REVISION,
    )
    output, report = tmp_path / "nonfinite-corrected.npz", tmp_path / "nonfinite-corrected.json"

    with pytest.raises(reembed.ValidationGateError, match="isotropy gate failed"):
        reembed.build_reembedded_bank(
            case.source,
            output,
            report,
            case.fino,
            minilm,
            biomedical,
            case.source_sha,
        )

    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["gate_error"]["gate"] == "isotropy"
    assert payload["validation"]["gate_values"]["exception_type"] == "AssertionError"
    assert "non-finite" in payload["validation"]["gate_values"]["exception_message"]
    assert not output.exists()
