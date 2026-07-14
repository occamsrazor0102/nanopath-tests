# Focused contracts for hierarchical MolCap aggregation and EMA centroid state.
# These tests keep the state proposal pure and every rejected commit atomic.

import math
import hashlib
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import molcap_relative_gate as relative_gate_module
import train as train_module
from train import (
    HierarchicalCentroidBank,
    centroid_audit,
    centroid_geometry,
    commit_matched_centroid_updates,
    crop_major_tile_mean,
    deterministic_grouped_sum,
    hierarchical_means,
    patient_targets_from_tiles,
    propose_matched_centroid_updates,
    require_centroid_gate,
    teacher_value_student_gradient,
)


def test_centroid_spectral_geometry_reports_sample_covariance_trace_and_full_descending_spectrum():
    centroids = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [2.0, 3.0]], dtype=torch.float32
    )

    geometry = train_module.centroid_spectral_geometry(centroids)

    assert geometry["trace"] == pytest.approx(4.0, rel=0, abs=1e-12)
    assert geometry["spectrum"] == pytest.approx([3.0, 1.0], rel=0, abs=1e-12)


def test_centroid_spectral_geometry_matches_archived_rank_and_participation_formulas():
    centroids = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [2.0, 3.0]], dtype=torch.float64
    )
    eigenvalues = np.asarray([3.0, 1.0])
    probabilities = eigenvalues / eigenvalues.sum()

    geometry = train_module.centroid_spectral_geometry(centroids)

    expected_erank = np.exp(-(probabilities * np.log(probabilities)).sum())
    expected_participation = eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
    assert geometry["effective_rank"] == pytest.approx(expected_erank, rel=1e-12)
    assert geometry["participation_ratio"] == pytest.approx(
        expected_participation, rel=1e-12
    )


def test_centroid_spectral_geometry_reports_raw_mean_offdiagonal_cosine():
    centroids = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [2.0, 3.0]], dtype=torch.float64
    )
    unit = centroids.numpy() / np.linalg.norm(centroids.numpy(), axis=1)[:, None]
    expected = (np.square(unit.sum(axis=0)).sum() - len(unit)) / (
        len(unit) * (len(unit) - 1)
    )

    geometry = train_module.centroid_spectral_geometry(centroids)

    assert geometry["mean_offdiag_cosine"] == pytest.approx(expected, rel=1e-12)
    assert geometry["min_norm"] == pytest.approx(1.0, rel=0, abs=1e-12)


def test_centroid_spectral_geometry_uses_cpu_float64_and_keeps_structural_zeros():
    centroids = torch.tensor(
        [[1.0, 0.0, 7.0], [3.0, 0.0, 7.0], [2.0, 3.0, 7.0]],
        dtype=torch.float32,
    )

    geometry = train_module.centroid_spectral_geometry(centroids)

    assert geometry["compute_device"] == "cpu"
    assert geometry["compute_dtype"] == "torch.float64"
    assert geometry["spectrum"] == pytest.approx([3.0, 1.0, 0.0], abs=1e-12)
    assert len(geometry["spectrum"]) == centroids.shape[1]


def test_relative_centroid_geometry_reports_centered_alignment_and_linear_cka():
    ema = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 3.0]])
    latest = torch.tensor([[2.0, 0.0], [6.0, 0.0], [4.0, 1.5]])
    ema0 = ema.double().numpy() - ema.double().numpy().mean(axis=0)
    latest0 = latest.double().numpy() - latest.double().numpy().mean(axis=0)
    expected_alignment = np.vdot(ema0, latest0) / (
        np.linalg.norm(ema0) * np.linalg.norm(latest0)
    )
    cross = ema0.T @ latest0
    expected_cka = np.square(cross).sum() / np.sqrt(
        np.square(ema0.T @ ema0).sum() * np.square(latest0.T @ latest0).sum()
    )

    geometry = train_module.relative_centroid_geometry(ema, latest)

    assert geometry["alignment"] == pytest.approx(expected_alignment, rel=1e-12)
    assert geometry["linear_cka"] == pytest.approx(expected_cka, rel=1e-12)


def test_relative_centroid_geometry_reports_trace_rank_participation_and_raw_cosine_ratios():
    ema = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 3.0]])
    latest = torch.tensor([[2.0, 0.0], [6.0, 0.0], [4.0, 1.5]])
    ema_eigenvalues = np.asarray([3.0, 1.0])
    latest_eigenvalues = np.asarray([4.0, 0.75])

    def ranks(eigenvalues):
        probabilities = eigenvalues / eigenvalues.sum()
        return (
            np.exp(-(probabilities * np.log(probabilities)).sum()),
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum(),
        )

    ema_erank, ema_participation = ranks(ema_eigenvalues)
    latest_erank, latest_participation = ranks(latest_eigenvalues)
    ema_array, latest_array = ema.double().numpy(), latest.double().numpy()
    ema_unit = ema_array / np.linalg.norm(ema_array, axis=1)[:, None]
    latest_unit = latest_array / np.linalg.norm(latest_array, axis=1)[:, None]
    ema_cosine = (np.square(ema_unit.sum(axis=0)).sum() - 3) / 6
    latest_cosine = (np.square(latest_unit.sum(axis=0)).sum() - 3) / 6

    geometry = train_module.relative_centroid_geometry(ema, latest)

    assert geometry["trace_ratio"] == pytest.approx(4.0 / 4.75, rel=1e-12)
    assert geometry["effective_rank_ratio"] == pytest.approx(
        ema_erank / latest_erank, rel=1e-12
    )
    assert geometry["participation_ratio"] == pytest.approx(
        ema_participation / latest_participation, rel=1e-12
    )
    assert geometry["mean_offdiag_cosine_delta"] == pytest.approx(
        ema_cosine - latest_cosine, rel=1e-12
    )
    assert geometry["ema"]["spectrum"] == pytest.approx([3.0, 1.0])
    assert geometry["latest"]["spectrum"] == pytest.approx([4.0, 0.75])


def test_linear_cka_stays_finite_for_extreme_finite_subnormal_centroids():
    pattern = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    ema = pattern * 1.0e-41
    latest = pattern * 2.0e-41

    geometry = train_module.relative_centroid_geometry(ema, latest)

    assert math.isfinite(geometry["linear_cka"])
    assert geometry["linear_cka"] == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_repeated_nonrepresentable_float64_rows_have_exact_zero_centered_geometry():
    value = torch.tensor(
        [0.1, 0.2, 0.4], dtype=torch.float32
    ).double().mean()
    repeated = torch.full((512, 3), value.item(), dtype=torch.float64)

    geometry = train_module.centroid_spectral_geometry(repeated)

    assert geometry["trace"] == 0.0
    assert geometry["spectrum"] == [0.0, 0.0, 0.0]
    assert geometry["effective_rank"] is None
    assert geometry["participation_ratio"] is None
    assert geometry["min_norm"] == pytest.approx(
        math.sqrt(3.0) * value.item(), rel=1e-15, abs=0.0
    )


def test_centered_geometry_is_exactly_translation_stable_for_nonconstant_rows():
    translation = torch.tensor(
        [0.1, 0.2, 0.4], dtype=torch.float32
    ).double().mean()
    base = torch.tensor([[-2.0], [1.0], [3.0]], dtype=torch.float64)

    base_geometry = train_module.centroid_spectral_geometry(base)
    translated_geometry = train_module.centroid_spectral_geometry(base + translation)

    for name in ("trace", "spectrum", "effective_rank", "participation_ratio"):
        assert translated_geometry[name] == base_geometry[name], name


def test_matched_latest_permutation_seed_uses_unsigned_big_endian_digest_prefix():
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"

    provenance = train_module.matched_latest_permutation_seed(
        target_sha256, mapping_digest
    )

    assert provenance == {
        "digest": "c72be56fb4b30bce20dc37fe1314f536ee376705dbdb5401594cc62df21c7361",
        "seed": 14351816905481980878,
        "seed_bytes": 8,
        "byte_order": "big",
        "unsigned": True,
        "domain": "molcap-matched-latest-v1",
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(provenance["seed"])
    assert generator.initial_seed() == 14351816905481980878


def test_matched_latest_permutations_use_exact_sequential_cpu_randperm_draws():
    ema = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    latest = torch.tensor(
        [[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"

    permutation = train_module.matched_latest_permutation_audit(
        ema, latest, target_sha256, mapping_digest, permutation_count=256
    )

    assert permutation["alignments"][:5] == pytest.approx(
        [
            0.0,
            0.9486832980505138,
            -0.4743416490252569,
            0.4743416490252569,
            -0.15811388300841897,
        ],
        rel=1e-15,
        abs=1e-15,
    )
    assert len(permutation["alignments"]) == 256
    assert permutation["identity_draw_count"] == 13
    assert hashlib.sha256(
        np.asarray(permutation["alignments"], dtype="<f8").tobytes()
    ).hexdigest() == "51a2a36b3e63c62e2c7fa029f3b30a820fc3daf56983873efa22dac4c02fde24"


def test_matched_latest_permutation_p_value_is_exact_one_sided_add_one_value():
    ema = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    latest = torch.tensor(
        [[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )

    permutation = train_module.matched_latest_permutation_audit(
        ema,
        latest,
        "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577",
        "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922",
        permutation_count=256,
    )

    assert permutation["observed_alignment"] == pytest.approx(
        3 / math.sqrt(10), rel=1e-15
    )
    assert permutation["exceedance_count"] == 13
    assert permutation["p_value"] == pytest.approx(14 / 257, rel=0, abs=0)
    assert permutation["p_value_formula"] == (
        "(1 + count(permuted_alignment >= observed_alignment)) / 257"
    )


def relative_gate_config():
    return {
        "gate_version": "matched_latest_v1",
        "min_slide_updates": 2,
        "min_sample_weighted_coverage": 0.95,
        "min_geometry_patients": 512,
        "min_effective_rank": 32.0,
        "min_participation_ratio": 0.5,
        "max_mean_offdiag_cosine": 0.95,
        "min_centroid_norm": 1.0e-6,
        "permutation_count": 256,
        "permutation_seed_domain": "molcap-matched-latest-v1",
        "min_trace_ratio": 0.05263157894736842,
        "min_effective_rank_ratio": 0.5,
        "min_alignment": 0.0,
        "max_permutation_p_value": 0.01,
    }


def passing_relative_gate_audit():
    return {
        "provenance": {
            "target_sha256_match": True,
            "mapping_digest_match": True,
            "world_size": 1,
        },
        "state": {
            "min_slide_updates": 2,
            "ema_finite": True,
            "latest_finite": True,
            "reported_scalars_finite": True,
            "matches": {
                "slide_mapping_equal": True,
                "slide_counts_equal": True,
                "tile_presentation_counts_equal": True,
                "state_step_equal": True,
                "observed_slides_equal": True,
                "mature_slides_equal": True,
                "patient_ids_equal": True,
                "matrix_shapes_equal": True,
            },
        },
        "population": {
            "ema_mature_coverage": 0.95,
            "latest_mature_coverage": 0.95,
            "matched_patient_count": 512,
        },
        "ema": {
            "trace": math.nextafter(0.0, math.inf),
            "min_norm": math.nextafter(1.0e-6, math.inf),
            "effective_rank": 1.0,
            "participation_ratio": 1.0,
            "mean_offdiag_cosine": 1.0,
        },
        "latest": {
            "trace": math.nextafter(0.0, math.inf),
            "min_norm": math.nextafter(1.0e-6, math.inf),
            "effective_rank": 1000.0,
            "participation_ratio": 1000.0,
            "mean_offdiag_cosine": -1.0,
        },
        "relative": {
            "trace_ratio": 0.05263157894736842,
            "effective_rank_ratio": 0.5,
            "participation_ratio": 0.5,
            "alignment": math.nextafter(0.0, math.inf),
            "linear_cka": 0.0,
            "mean_offdiag_cosine_delta": 2.0,
        },
        "permutation": {
            "count": 256,
            "exceedance_count": 1,
            "p_value": 2 / 257,
        },
        "unavailable": [],
    }


def test_matched_latest_gate_uses_exact_boundaries_and_names_every_failed_condition():
    passing = train_module.evaluate_matched_latest_gate(
        passing_relative_gate_audit(), relative_gate_config()
    )
    assert passing["passed"] is True
    assert passing["failures"] == []
    assert passing["thresholds"] == {
        "min_slide_updates": 2,
        "min_sample_weighted_coverage": 0.95,
        "min_geometry_patients": 512,
        "min_centroid_norm": 1.0e-6,
        "min_trace_ratio": 0.05263157894736842,
        "min_effective_rank_ratio": 0.5,
        "min_participation_ratio": 0.5,
        "min_alignment_exclusive": 0.0,
        "permutation_count": 256,
        "max_permutation_p_value": 0.01,
    }

    audit = deepcopy(passing_relative_gate_audit())
    audit["provenance"].update(
        target_sha256_match=False, mapping_digest_match=False, world_size=2
    )
    audit["state"].update(
        min_slide_updates=3,
        ema_finite=False,
        latest_finite=False,
        reported_scalars_finite=False,
    )
    audit["state"]["matches"] = {
        name: False for name in audit["state"]["matches"]
    }
    audit["population"].update(
        ema_mature_coverage=math.nextafter(0.95, -math.inf),
        latest_mature_coverage=math.nextafter(0.95, -math.inf),
        matched_patient_count=511,
    )
    audit["ema"].update(trace=0.0, min_norm=1.0e-6)
    audit["latest"].update(trace=0.0, min_norm=1.0e-6)
    audit["relative"].update(
        trace_ratio=math.nextafter(0.05263157894736842, -math.inf),
        effective_rank_ratio=math.nextafter(0.5, -math.inf),
        participation_ratio=math.nextafter(0.5, -math.inf),
        alignment=0.0,
    )
    audit["permutation"].update(exceedance_count=2, p_value=3 / 257)

    failed = train_module.evaluate_matched_latest_gate(audit, relative_gate_config())

    assert failed["passed"] is False
    assert failed["failures"] == [
        "target_sha256_match",
        "mapping_digest_match",
        "world_size_one",
        "min_slide_updates_exact",
        "ema_state_finite",
        "latest_state_finite",
        "slide_mapping_equal",
        "slide_counts_equal",
        "tile_presentation_counts_equal",
        "state_step_equal",
        "observed_slides_equal",
        "mature_slides_equal",
        "patient_ids_equal",
        "matrix_shapes_equal",
        "ema_min_sample_weighted_coverage",
        "latest_min_sample_weighted_coverage",
        "min_geometry_patients",
        "ema_min_centroid_norm_strict",
        "latest_min_centroid_norm_strict",
        "ema_trace_positive",
        "latest_trace_positive",
        "all_reported_scalars_finite",
        "min_trace_ratio",
        "min_effective_rank_ratio",
        "min_participation_ratio",
        "min_alignment_strict",
        "max_permutation_p_value",
    ]


def oracle_matched_banks():
    mapping = torch.arange(4, dtype=torch.int64)
    ema = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.9)
    latest = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.0)
    with torch.no_grad():
        ema.slide_centroids.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        )
        latest.slide_centroids.copy_(
            torch.tensor([[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0]])
        )
        for bank in (ema, latest):
            bank.slide_counts.fill_(2)
            bank.slide_tile_presentations.copy_(torch.tensor([3, 4, 5, 6]))
            bank.centroid_state_step.fill_(2)
    return ema, latest


def test_matched_latest_audit_uses_canonical_population_and_complete_geometry():
    ema, latest = oracle_matched_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    history_metadata = {
        "target_sha256": target_sha256,
        "mapping_digest": mapping_digest,
    }
    shadow_metadata = {
        **history_metadata,
        "gate_version": "matched_latest_v1",
        "arm": "latest_observation_shadow",
    }

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=history_metadata,
        shadow_metadata=shadow_metadata,
        world_size=1,
    )

    assert audit["gate_version"] == "matched_latest_v1"
    assert audit["state"]["matches"] == {
        "slide_mapping_equal": True,
        "slide_counts_equal": True,
        "tile_presentation_counts_equal": True,
        "state_step_equal": True,
        "observed_slides_equal": True,
        "mature_slides_equal": True,
        "patient_ids_equal": True,
        "matrix_shapes_equal": True,
    }
    assert audit["population"]["matched_patient_count"] == 4
    assert audit["population"]["ema_patient_ids"]["count"] == 4
    assert audit["population"]["ema_patient_ids"] == audit["population"][
        "latest_patient_ids"
    ]
    assert audit["population"]["ema_matrix_shape"] == [4, 2]
    assert audit["population"]["latest_matrix_shape"] == [4, 2]
    assert audit["ema"]["spectrum"] == pytest.approx([2 / 3, 2 / 3])
    assert audit["latest"]["spectrum"] == pytest.approx([8 / 3, 2 / 3])
    assert audit["relative"]["trace_ratio"] == pytest.approx(0.4)
    assert audit["relative"]["effective_rank_ratio"] == pytest.approx(
        1.2125732532083184
    )
    assert audit["relative"]["participation_ratio"] == pytest.approx(1.36)
    assert audit["relative"]["alignment"] == pytest.approx(3 / math.sqrt(10))
    assert audit["relative"]["linear_cka"] == pytest.approx(5 / math.sqrt(34))
    assert audit["relative"]["mean_offdiag_cosine_delta"] == pytest.approx(0.0)
    assert len(audit["permutation"]["alignments"]) == 256
    assert audit["permutation"]["p_value"] == pytest.approx(14 / 257)
    assert audit["shadow"] == {
        "audit_time_present": True,
        "checkpoint_payload_present": True,
        "checkpoint_tensor_payload_bytes": 104,
        "state_step": 2,
        "bank_state_digest": train_module.centroid_bank_state_digest(latest),
        "post_pass_action": "discard_after_durable_pass_report",
        "boundary_proposal": {
            "present": False,
            "type_exact": None,
            "transaction_valid": None,
            "committed_match": None,
            "state_step": 2,
            "first_copy_excluded": True,
            "count": 0,
            "mean": None,
            "q10": None,
            "q50": None,
            "q90": None,
        },
    }
    assert audit["unavailable"] == []


def test_matched_latest_audit_validates_and_records_shadow_boundary_proposal():
    ema, latest = oracle_matched_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    proposal = train_module.CentroidProposal(
        base_state_step=1,
        slide_ids=torch.arange(4, dtype=torch.int64),
        next_slide_centroids=latest.slide_centroids.detach().clone(),
        slide_tile_counts=torch.ones(4, dtype=torch.int64),
        patient_ids=torch.arange(4, dtype=torch.int64),
        patient_centroids=latest.slide_centroids.detach().clone(),
        drift_cosines=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        historical_tile_fraction=torch.tensor(1.0),
    )

    valid = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
        boundary_shadow_proposal=proposal,
    )
    invalid = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
        boundary_shadow_proposal=object(),
    )
    invalid_report = train_module.evaluate_matched_latest_gate(
        invalid, relative_gate_config()
    )

    assert valid["shadow"]["boundary_proposal"] == {
        "present": True,
        "type_exact": True,
        "transaction_valid": True,
        "committed_match": True,
        "state_step": 2,
        "first_copy_excluded": True,
        "count": 4,
        "mean": pytest.approx(0.25),
        "q10": pytest.approx(0.13),
        "q50": pytest.approx(0.25),
        "q90": pytest.approx(0.37),
    }
    assert invalid["shadow"]["boundary_proposal"]["present"] is True
    assert invalid["shadow"]["boundary_proposal"]["committed_match"] is False
    assert "boundary_shadow_proposal_committed" in invalid_report["failures"]


def test_matched_latest_audit_keeps_exact_draw_and_maturity_semantics_when_config_is_invalid():
    ema, latest = oracle_matched_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    invalid = {**relative_gate_config(), "min_slide_updates": 3, "permutation_count": 7}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        invalid,
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )
    report = train_module.evaluate_matched_latest_gate(audit, invalid)

    assert audit["state"]["min_slide_updates"] == 2
    assert audit["population_sizes"]["mature_min_slide_updates"] == 2
    assert audit["permutation"]["count"] == 256
    assert len(audit["permutation"]["alignments"]) == 256
    assert report["failures"][:2] == [
        "config_min_slide_updates_exact",
        "config_permutation_count_exact",
    ]


def test_observed_matrix_audit_remains_complete_when_mature_state_mismatch_fails_gate():
    ema, latest = oracle_matched_banks()
    with torch.no_grad():
        latest.slide_counts[0] = 1
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )
    report = train_module.evaluate_matched_latest_gate(
        audit, relative_gate_config()
    )

    assert audit["state"]["matches"]["mature_slides_equal"] is False
    assert audit["state"]["matches"]["patient_ids_equal"] is True
    assert audit["state"]["matches"]["matrix_shapes_equal"] is True
    assert len(audit["ema"]["spectrum"]) == 2
    assert len(audit["latest"]["spectrum"]) == 2
    assert len(audit["permutation"]["alignments"]) == 256
    assert audit["unavailable"] == []
    assert report["passed"] is False
    assert "slide_counts_equal" in report["failures"]
    assert "mature_slides_equal" in report["failures"]


def test_matched_latest_audit_uses_strict_json_nulls_when_shapes_preclude_relative_math():
    ema, _ = oracle_matched_banks()
    mapping = torch.arange(4, dtype=torch.int64)
    latest = HierarchicalCentroidBank(mapping, feature_dim=3, momentum=0.0)
    with torch.no_grad():
        latest.slide_centroids.copy_(
            torch.tensor(
                [
                    [2.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [-2.0, 0.0, 1.0],
                    [0.0, -1.0, 1.0],
                ]
            )
        )
        latest.slide_counts.fill_(2)
        latest.slide_tile_presentations.copy_(torch.tensor([3, 4, 5, 6]))
        latest.centroid_state_step.fill_(2)
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )

    assert audit["state"]["matches"]["matrix_shapes_equal"] is False
    assert len(audit["ema"]["spectrum"]) == 2
    assert len(audit["latest"]["spectrum"]) == 3
    assert all(value is None for value in audit["relative"].values())
    assert audit["permutation"]["alignments"] is None
    assert audit["permutation"]["p_value"] is None
    assert audit["unavailable"] == ["relative_geometry", "permutation"]
    json_text = json.dumps(audit, allow_nan=False)
    assert "NaN" not in json_text and "Infinity" not in json_text


def test_matched_latest_audit_names_nonfinite_geometry_without_nonfinite_json():
    ema, latest = oracle_matched_banks()
    with torch.no_grad():
        latest.slide_centroids.copy_(
            torch.tensor(
                [[float("nan"), 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0]]
            )
        )
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )

    assert audit["latest"] == {
        "compute_device": "cpu",
        "compute_dtype": "torch.float64",
        "trace": None,
        "spectrum": None,
        "effective_rank": None,
        "participation_ratio": None,
        "mean_offdiag_cosine": None,
        "min_norm": None,
    }
    assert all(value is None for value in audit["relative"].values())
    assert audit["permutation"]["alignments"] is None
    assert audit["unavailable"] == [
        "latest_geometry:nonfinite",
        "relative_geometry",
        "permutation",
    ]
    json.dumps(audit, allow_nan=False)


def test_zero_row_preserves_every_centered_metric_and_permutation_evidence():
    ema, latest = oracle_matched_banks()
    with torch.no_grad():
        latest.slide_centroids[0].zero_()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )
    report = train_module.evaluate_matched_latest_gate(audit, relative_gate_config())

    assert audit["latest"]["min_norm"] == 0.0
    assert audit["latest"]["trace"] > 0.0
    assert len(audit["latest"]["spectrum"]) == 2
    assert audit["latest"]["effective_rank"] is not None
    assert audit["latest"]["participation_ratio"] is not None
    assert audit["latest"]["mean_offdiag_cosine"] is None
    for name in (
        "trace_ratio",
        "effective_rank_ratio",
        "participation_ratio",
        "alignment",
        "linear_cka",
    ):
        assert audit["relative"][name] is not None, name
    assert audit["relative"]["mean_offdiag_cosine_delta"] is None
    assert len(audit["permutation"]["alignments"]) == 256
    assert audit["unavailable"] == [
        "diagnostic:latest.mean_offdiag_cosine:zero_norm",
        "diagnostic:relative.mean_offdiag_cosine_delta:input_unavailable",
    ]
    assert "latest_min_centroid_norm_strict" in report["failures"]
    assert not any(
        failure.startswith("audit_available:diagnostic:")
        for failure in report["failures"]
    )
    json.dumps(report, allow_nan=False)


def test_zero_row_on_ema_preserves_partial_legacy_evidence_without_hard_unavailability():
    ema, latest = oracle_matched_banks()
    with torch.no_grad():
        ema.slide_centroids[0].zero_()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )
    report = train_module.evaluate_matched_latest_gate(audit, relative_gate_config())

    assert audit["ema"]["min_norm"] == 0.0
    assert audit["ema"]["trace"] > 0.0
    assert len(audit["ema"]["spectrum"]) == 2
    assert audit["all_observed"] == {
        "patient_count": 4,
        "min_norm": 0.0,
        "effective_rank": audit["ema"]["effective_rank"],
        "participation_ratio": audit["ema"]["participation_ratio"],
        "mean_offdiag_cosine": None,
    }
    assert all(
        audit["relative"][name] is not None
        for name in (
            "trace_ratio",
            "effective_rank_ratio",
            "participation_ratio",
            "alignment",
            "linear_cka",
        )
    )
    assert len(audit["permutation"]["alignments"]) == 256
    assert "legacy_diagnostics" not in audit["unavailable"]
    assert "ema_min_centroid_norm_strict" in report["failures"]
    assert "audit_available:legacy_diagnostics" not in report["failures"]
    json.dumps(report, allow_nan=False)


def test_constant_nonzero_bank_preserves_zero_spectrum_trace_norm_and_other_bank():
    ema, latest = oracle_matched_banks()
    with torch.no_grad():
        latest.slide_centroids.fill_(1.0)
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )
    report = train_module.evaluate_matched_latest_gate(audit, relative_gate_config())

    assert audit["ema"]["trace"] > 0.0
    assert len(audit["ema"]["spectrum"]) == 2
    assert audit["latest"]["trace"] == 0.0
    assert audit["latest"]["spectrum"] == [0.0, 0.0]
    assert audit["latest"]["min_norm"] == pytest.approx(math.sqrt(2.0))
    assert audit["latest"]["effective_rank"] is None
    assert audit["latest"]["participation_ratio"] is None
    assert audit["latest"]["mean_offdiag_cosine"] == pytest.approx(1.0)
    assert audit["relative"]["trace_ratio"] is None
    assert audit["relative"]["effective_rank_ratio"] is None
    assert audit["relative"]["participation_ratio"] is None
    assert audit["relative"]["alignment"] is None
    assert audit["relative"]["linear_cka"] is None
    assert audit["relative"]["mean_offdiag_cosine_delta"] is not None
    assert audit["permutation"]["alignments"] is None
    assert audit["unavailable"] == [
        "diagnostic:latest.effective_rank:zero_trace",
        "diagnostic:latest.participation_ratio:zero_trace",
        "relative.trace_ratio:latest_zero_trace",
        "relative.effective_rank_ratio:input_unavailable",
        "relative.participation_ratio:input_unavailable",
        "relative.alignment:zero_centered_norm",
        "diagnostic:relative.linear_cka:zero_centered_norm",
        "permutation:zero_centered_norm",
    ]
    assert "latest_trace_positive" in report["failures"]
    assert "ema_trace_positive" not in report["failures"]
    json.dumps(report, allow_nan=False)


def test_constant_nonzero_ema_preserves_partial_legacy_and_latest_positive_trace():
    ema, latest = oracle_matched_banks()
    with torch.no_grad():
        ema.slide_centroids.fill_(1.0)
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )
    report = train_module.evaluate_matched_latest_gate(audit, relative_gate_config())

    assert audit["ema"]["trace"] == 0.0
    assert audit["ema"]["spectrum"] == [0.0, 0.0]
    assert audit["ema"]["min_norm"] == pytest.approx(math.sqrt(2.0))
    assert audit["latest"]["trace"] > 0.0
    assert audit["all_observed"] == {
        "patient_count": 4,
        "min_norm": pytest.approx(math.sqrt(2.0)),
        "effective_rank": None,
        "participation_ratio": None,
        "mean_offdiag_cosine": pytest.approx(1.0),
    }
    assert audit["relative"]["trace_ratio"] == 0.0
    assert audit["relative"]["effective_rank_ratio"] is None
    assert audit["relative"]["participation_ratio"] is None
    assert audit["relative"]["alignment"] is None
    assert audit["relative"]["linear_cka"] is None
    assert audit["relative"]["mean_offdiag_cosine_delta"] is not None
    assert audit["permutation"]["alignments"] is None
    assert "legacy_diagnostics" not in audit["unavailable"]
    assert "ema_trace_positive" in report["failures"]
    assert "latest_trace_positive" not in report["failures"]
    assert "audit_available:legacy_diagnostics" not in report["failures"]
    json.dumps(report, allow_nan=False)


def test_matched_latest_gate_names_unavailable_null_metrics_without_short_circuiting():
    audit = passing_relative_gate_audit()
    audit["state"]["matches"]["matrix_shapes_equal"] = False
    audit["relative"] = {name: None for name in audit["relative"]}
    audit["permutation"].update(
        identity_draw_count=None,
        exceedance_count=None,
        p_value=None,
        alignments=None,
    )
    audit["unavailable"] = ["relative_geometry", "permutation"]

    report = train_module.evaluate_matched_latest_gate(
        audit, relative_gate_config()
    )

    assert report["passed"] is False
    assert report["failures"] == [
        "matrix_shapes_equal",
        "min_trace_ratio",
        "min_effective_rank_ratio",
        "min_participation_ratio",
        "min_alignment_strict",
        "max_permutation_p_value",
        "audit_available:relative_geometry",
        "audit_available:permutation",
    ]
    json.dumps(report, allow_nan=False)


def large_matched_gate_banks(*, latest_sign=1.0):
    mapping = torch.arange(512, dtype=torch.int64)
    ema = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.9)
    latest = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.0)
    index = torch.arange(1, 513, dtype=torch.float32)
    values = torch.stack((index, index.remainder(17) + 1.0), dim=1)
    with torch.no_grad():
        ema.slide_centroids.copy_(values)
        latest.slide_centroids.copy_(values * latest_sign)
        for bank in (ema, latest):
            bank.slide_counts.fill_(2)
            bank.slide_tile_presentations.fill_(3)
            bank.centroid_state_step.fill_(2)
    return ema, latest


def committed_boundary_proposal(bank):
    slide_ids = torch.arange(len(bank.slide_centroids), dtype=torch.int64)
    return train_module.CentroidProposal(
        base_state_step=int(bank.centroid_state_step.item()) - 1,
        slide_ids=slide_ids,
        next_slide_centroids=bank.slide_centroids.detach().clone(),
        slide_tile_counts=torch.ones_like(slide_ids),
        patient_ids=slide_ids.clone(),
        patient_centroids=bank.slide_centroids.detach().clone(),
        drift_cosines=torch.ones(len(slide_ids), dtype=bank.slide_centroids.dtype),
        historical_tile_fraction=torch.tensor(1.0),
    )


def duck_boundary_proposal(proposal):
    return types.SimpleNamespace(
        base_state_step=proposal.base_state_step,
        slide_ids=proposal.slide_ids,
        next_slide_centroids=proposal.next_slide_centroids,
        drift_cosines=proposal.drift_cosines,
    )


def mutate_boundary_transaction(proposal, field):
    if field == "base_state_step":
        return proposal._replace(base_state_step=np.int64(proposal.base_state_step))
    if field == "slide_ids":
        return proposal._replace(slide_ids=proposal.slide_ids.to(torch.float32))
    if field == "next_slide_centroids":
        values = proposal.next_slide_centroids.clone()
        values[0, 0] = float("nan")
        return proposal._replace(next_slide_centroids=values)
    if field == "slide_tile_counts":
        counts = proposal.slide_tile_counts.clone()
        counts[0] = 0
        return proposal._replace(slide_tile_counts=counts)
    if field == "patient_ids":
        patient_ids = proposal.patient_ids.clone()
        patient_ids[-1] = patient_ids[-2]
        return proposal._replace(patient_ids=patient_ids)
    if field == "patient_centroids":
        centroids = proposal.patient_centroids.clone()
        centroids[0, 0] = float("nan")
        return proposal._replace(patient_centroids=centroids)
    if field == "drift_cosines":
        return proposal._replace(
            drift_cosines=torch.full_like(proposal.drift_cosines, 1.5)
        )
    if field == "historical_tile_fraction":
        return proposal._replace(historical_tile_fraction=torch.tensor(1.5))
    raise AssertionError(field)


def mismatch_paired_boundary_transaction(proposal, field):
    if field == "base_state_step":
        return proposal._replace(base_state_step=proposal.base_state_step + 1)
    if field == "slide_ids":
        return proposal._replace(
            slide_ids=proposal.slide_ids[:-1],
            next_slide_centroids=proposal.next_slide_centroids[:-1],
            slide_tile_counts=proposal.slide_tile_counts[:-1],
            patient_ids=proposal.patient_ids[:-1],
            patient_centroids=proposal.patient_centroids[:-1],
            drift_cosines=proposal.drift_cosines[:-1],
        )
    if field == "slide_tile_counts":
        counts = proposal.slide_tile_counts.clone()
        counts[0] = 2
        return proposal._replace(slide_tile_counts=counts)
    if field == "patient_ids":
        patient_ids = proposal.patient_ids.clone()
        patient_ids[-1] = patient_ids[-2]
        return proposal._replace(patient_ids=patient_ids)
    if field == "historical_tile_fraction":
        return proposal._replace(historical_tile_fraction=torch.tensor(0.5))
    raise AssertionError(field)


@pytest.mark.parametrize("malformed_side", ["ema", "shadow"])
def test_relative_runner_persists_exact_proposal_type_failures(
    tmp_path, malformed_side
):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    proposals = {
        "ema": committed_boundary_proposal(ema),
        "shadow": committed_boundary_proposal(latest),
    }
    proposals[malformed_side] = duck_boundary_proposal(proposals[malformed_side])
    expected_failure = f"boundary_{malformed_side}_proposal_type_exact"
    path = tmp_path / f"proposal-type-{malformed_side}.json"

    with pytest.raises(AssertionError, match=expected_failure):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
            boundary_proposal=proposals["ema"],
            boundary_shadow_proposal=proposals["shadow"],
        )

    report = json.loads(path.read_text())
    provenance = report["state"]["boundary_proposals"][malformed_side]
    assert provenance["present"] is True
    assert provenance["type_exact"] is False
    assert provenance["transaction_valid"] is False
    assert provenance["committed_match"] is False
    assert expected_failure in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("malformed_side", ["ema", "shadow"])
def test_relative_runner_persists_sparse_exact_proposal_field_failures(
    tmp_path, malformed_side
):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    proposals = {
        "ema": committed_boundary_proposal(ema),
        "shadow": committed_boundary_proposal(latest),
    }
    proposals[malformed_side] = proposals[malformed_side]._replace(
        slide_ids=proposals[malformed_side].slide_ids.to_sparse()
    )
    expected_failure = f"boundary_{malformed_side}_proposal_transaction_valid"
    path = tmp_path / f"proposal-sparse-{malformed_side}.json"

    with pytest.raises(AssertionError, match=expected_failure):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
            boundary_proposal=proposals["ema"],
            boundary_shadow_proposal=proposals["shadow"],
        )

    report = json.loads(path.read_text())
    provenance = report["state"]["boundary_proposals"][malformed_side]
    assert provenance["type_exact"] is True
    assert provenance["transaction_valid"] is False
    assert provenance["committed_match"] is False
    assert expected_failure in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "field",
    [
        "base_state_step",
        "slide_ids",
        "next_slide_centroids",
        "slide_tile_counts",
        "patient_ids",
        "patient_centroids",
        "drift_cosines",
        "historical_tile_fraction",
    ],
)
def test_boundary_proposal_authenticity_validates_every_transaction_field(field):
    ema, latest = oracle_matched_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    ema_proposal = committed_boundary_proposal(ema)
    shadow_proposal = mutate_boundary_transaction(
        committed_boundary_proposal(latest), field
    )

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
        boundary_proposal=ema_proposal,
        boundary_shadow_proposal=shadow_proposal,
    )
    report = train_module.evaluate_matched_latest_gate(audit, relative_gate_config())

    provenance = audit["state"]["boundary_proposals"]["shadow"]
    assert provenance["type_exact"] is True
    assert provenance["transaction_valid"] is False
    assert provenance["committed_match"] is False
    assert "boundary_shadow_proposal_transaction_valid" in report["failures"]


def test_boundary_proposal_authenticity_allows_float32_cosine_roundoff_at_one():
    ema, latest = oracle_matched_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    def with_cosine_roundoff(proposal):
        return proposal._replace(
            drift_cosines=torch.full_like(proposal.drift_cosines, 1.0 + 3.0e-7)
        )

    audit = train_module.matched_latest_centroid_audit(
        ema,
        latest,
        relative_gate_config(),
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
        boundary_proposal=with_cosine_roundoff(committed_boundary_proposal(ema)),
        boundary_shadow_proposal=with_cosine_roundoff(
            committed_boundary_proposal(latest)
        ),
    )

    proposals = audit["state"]["boundary_proposals"]
    assert proposals["ema"]["transaction_valid"] is True
    assert proposals["shadow"]["transaction_valid"] is True
    assert proposals["ema"]["committed_match"] is True
    assert proposals["shadow"]["committed_match"] is True
    assert proposals["paired"]["applicable"] is True
    assert all(proposals["paired"]["matches"].values())


@pytest.mark.parametrize(
    ("field", "match_name"),
    [
        ("base_state_step", "base_state_step_equal"),
        ("slide_ids", "slide_ids_equal"),
        ("slide_tile_counts", "slide_tile_counts_equal"),
        ("patient_ids", "patient_ids_equal"),
        (
            "historical_tile_fraction",
            "historical_tile_fraction_equal",
        ),
    ],
)
def test_relative_runner_persists_paired_boundary_transaction_mismatches(
    tmp_path, field, match_name
):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    ema_proposal = committed_boundary_proposal(ema)
    shadow_proposal = mismatch_paired_boundary_transaction(
        committed_boundary_proposal(latest), field
    )
    expected_failure = f"boundary_proposal_{match_name}"
    path = tmp_path / f"proposal-pair-{field}.json"

    with pytest.raises(AssertionError):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
            boundary_proposal=ema_proposal,
            boundary_shadow_proposal=shadow_proposal,
        )

    report = json.loads(path.read_text())
    paired = report["state"]["boundary_proposals"]["paired"]
    assert paired["applicable"] is True
    assert paired["matches"][match_name] is False
    assert expected_failure in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("ema_case", "shadow_case", "expected_failure"),
    [
        ("invalid", "valid", "boundary_ema_proposal_committed"),
        ("valid", "invalid", "boundary_shadow_proposal_committed"),
        ("valid", "absent", "boundary_proposal_presence_parity"),
        ("absent", "valid", "boundary_proposal_presence_parity"),
    ],
)
def test_relative_runner_persists_symmetric_boundary_proposal_failures(
    tmp_path, ema_case, shadow_case, expected_failure
):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    choices = {
        "valid": None,
        "invalid": object(),
        "absent": None,
    }
    choices["valid_ema"] = committed_boundary_proposal(ema)
    choices["valid_shadow"] = committed_boundary_proposal(latest)
    ema_proposal = choices["valid_ema"] if ema_case == "valid" else choices[ema_case]
    shadow_proposal = (
        choices["valid_shadow"] if shadow_case == "valid" else choices[shadow_case]
    )
    path = tmp_path / f"proposal-{ema_case}-{shadow_case}.json"

    with pytest.raises(AssertionError, match=expected_failure):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
            boundary_proposal=ema_proposal,
            boundary_shadow_proposal=shadow_proposal,
        )

    report = json.loads(path.read_text())
    proposals = report["state"]["boundary_proposals"]
    assert proposals["ema"]["present"] is (ema_case != "absent")
    assert proposals["shadow"]["present"] is (shadow_case != "absent")
    assert proposals["presence_equal"] is (
        (ema_case == "absent") == (shadow_case == "absent")
    )
    if ema_case == "invalid":
        assert proposals["ema"]["committed_match"] is False
    if shadow_case == "invalid":
        assert proposals["shadow"]["committed_match"] is False
    assert expected_failure in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


def test_relative_runner_rejects_negative_alignment_even_when_cka_is_one_with_complete_json(
    tmp_path,
):
    ema, latest = large_matched_gate_banks(latest_sign=-1.0)
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "relative-failure.json"

    with pytest.raises(AssertionError, match="min_alignment_strict"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
        )

    report = json.loads(path.read_text())
    assert report["gate_version"] == "matched_latest_v1"
    assert report["passed"] is False
    assert report["failures"] == [
        "min_alignment_strict",
        "max_permutation_p_value",
    ]
    assert report["relative"]["alignment"] == pytest.approx(-1.0)
    assert report["relative"]["linear_cka"] == pytest.approx(1.0)
    assert len(report["ema"]["spectrum"]) == 2
    assert len(report["latest"]["spectrum"]) == 2
    assert len(report["permutation"]["alignments"]) == 256
    json.dumps(report, allow_nan=False)


def test_relative_runner_durably_passes_with_absolute_geometry_only_diagnostic(
    tmp_path,
):
    ema, latest = large_matched_gate_banks(latest_sign=1.0)
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "relative-pass.json"
    report = train_module.run_centroid_ramp_gate(
        ema,
        relative_gate_config(),
        path,
        latest_bank=latest,
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=1,
    )

    assert report["passed"] is True and report["failures"] == []
    assert report["ema"]["effective_rank"] < 32.0
    assert report["ema"]["participation_ratio"] < 16.0
    assert report["ema"]["mean_offdiag_cosine"] > 0.95
    assert report["relative"]["alignment"] == pytest.approx(1.0)
    assert report["permutation"]["p_value"] <= 0.01
    expected_strategy = (
        "windows_movefileex_replace_existing_write_through"
        if relative_gate_module.os.name == "nt"
        else "posix_temp_flush_fsync_replace_parent_fsync"
    )
    assert report["persistence"] == {
        "strategy": expected_strategy,
        "durable_before_return": True,
    }
    assert json.loads(path.read_text()) == report


def test_parent_directory_fsync_opens_syncs_and_closes_posix_directory(
    tmp_path, monkeypatch
):
    events = []
    directory_fd = 9173

    monkeypatch.setattr(
        relative_gate_module.os,
        "open",
        lambda path, flags: events.append(("open", path, flags)) or directory_fd,
    )
    monkeypatch.setattr(
        relative_gate_module.os,
        "fsync",
        lambda fd: events.append(("fsync", fd)),
    )
    monkeypatch.setattr(
        relative_gate_module.os,
        "close",
        lambda fd: events.append(("close", fd)),
    )

    assert relative_gate_module._fsync_parent_directory(
        tmp_path, platform_name="posix"
    ) is True
    assert events[0][0:2] == ("open", tmp_path)
    assert events[1:] == [("fsync", directory_fd), ("close", directory_fd)]


def minimal_relative_report():
    return {
        "gate_version": "matched_latest_v1",
        "passed": False,
        "failures": ["synthetic_failure"],
    }


def test_windows_write_through_publish_uses_replace_and_write_through_flags(
    tmp_path,
):
    source = tmp_path / "source.tmp"
    destination = tmp_path / "report.json"
    source.write_text("strict report\n")
    calls = []

    def move_file_ex(source_name, destination_name, flags):
        calls.append((source_name, destination_name, flags))
        relative_gate_module.os.replace(source_name, destination_name)
        return 1

    relative_gate_module._windows_write_through_replace(
        source,
        destination,
        move_file_ex=move_file_ex,
    )

    assert calls == [
        (
            str(source.resolve()),
            str(destination.resolve()),
            0x00000001 | 0x00000008,
        )
    ]
    assert destination.read_text() == "strict report\n"
    assert not source.exists()


def test_windows_writer_persists_truthful_strategy_and_cleans_native_failure_temp(
    tmp_path, monkeypatch
):
    success_path = tmp_path / "windows-success.json"
    failure_path = tmp_path / "windows-failure.json"
    calls = []

    def successful_move(source_name, destination_name, flags):
        calls.append((Path(source_name), Path(destination_name), flags))
        relative_gate_module.os.replace(source_name, destination_name)
        return 1

    monkeypatch.setattr(
        relative_gate_module,
        "_fsync_parent_directory",
        lambda *args, **kwargs: pytest.fail(
            "Windows publication must not claim or call parent-directory fsync"
        ),
    )

    persisted = relative_gate_module._write_matched_latest_gate_report(
        minimal_relative_report(),
        success_path,
        platform_name="nt",
        move_file_ex=successful_move,
    )

    assert persisted["persistence"] == {
        "strategy": "windows_movefileex_replace_existing_write_through",
        "durable_before_return": True,
    }
    assert json.loads(success_path.read_text()) == persisted
    assert calls[0][2] == 0x00000001 | 0x00000008

    failed_sources = []

    def failed_move(source_name, destination_name, flags):
        failed_sources.append(Path(source_name))
        return 0

    with pytest.raises(OSError, match="Access is denied"):
        relative_gate_module._write_matched_latest_gate_report(
            minimal_relative_report(),
            failure_path,
            platform_name="nt",
            move_file_ex=failed_move,
            get_last_error=lambda: 5,
            format_error=lambda code: "Access is denied",
        )

    assert len(failed_sources) == 1
    assert not failed_sources[0].exists()
    assert not failure_path.exists()


def test_report_writer_sanitizes_non_deepcopyable_tensor_before_publication(tmp_path):
    path = tmp_path / "non-deepcopyable.json"
    nonleaf = torch.tensor(float("nan"), requires_grad=True) * 1.0
    source = minimal_relative_report()
    source["nested"] = {"nonleaf_tensor": nonleaf}

    def successful_move(source_name, destination_name, flags):
        relative_gate_module.os.replace(source_name, destination_name)
        return 1

    persisted = relative_gate_module._write_matched_latest_gate_report(
        source,
        path,
        platform_name="nt",
        move_file_ex=successful_move,
    )

    assert persisted["nested"]["nonleaf_tensor"] is None
    assert "nested.nonleaf_tensor" in persisted["nonfinite_paths"]
    assert "report_nonfinite:nested.nonleaf_tensor" in persisted["failures"]
    assert persisted["passed"] is False
    assert json.loads(path.read_text()) == persisted
    assert "persistence" not in source
    assert torch.isnan(source["nested"]["nonleaf_tensor"])


def test_report_writer_uses_unique_same_directory_temps_and_cleans_publish_failures(
    tmp_path,
):
    path = tmp_path / "report.json"
    sources = []

    class PublishFailure(RuntimeError):
        pass

    def failed_replace(source, destination):
        sources.append(Path(source))
        raise PublishFailure("replace failed")

    for _ in range(2):
        with pytest.raises(PublishFailure, match="replace failed"):
            relative_gate_module._write_matched_latest_gate_report(
                minimal_relative_report(),
                path,
                platform_name="posix",
                replace_operation=failed_replace,
            )

    assert len(set(sources)) == 2
    assert all(source.parent == tmp_path for source in sources)
    assert all(not source.exists() for source in sources)
    assert not path.exists()


def test_report_writer_cleans_fsync_failure_without_masking_original_cleanup_error(
    tmp_path, monkeypatch
):
    path = tmp_path / "report.json"
    cleanup_attempts = []

    class FsyncFailure(RuntimeError):
        pass

    class CleanupFailure(RuntimeError):
        pass

    original_unlink = relative_gate_module.Path.unlink

    def failed_fsync(fd):
        raise FsyncFailure("fsync failed")

    def failed_cleanup(candidate, *args, **kwargs):
        if candidate.parent == tmp_path and candidate.name.startswith(".report.json."):
            cleanup_attempts.append(candidate)
            raise CleanupFailure("cleanup failed")
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(relative_gate_module.os, "fsync", failed_fsync)
    monkeypatch.setattr(relative_gate_module.Path, "unlink", failed_cleanup)

    with pytest.raises(FsyncFailure, match="fsync failed"):
        relative_gate_module._write_matched_latest_gate_report(
            minimal_relative_report(), path, platform_name="posix"
        )

    assert len(cleanup_attempts) == 1
    original_unlink(cleanup_attempts[0])


def test_report_writer_cleans_unique_temp_after_fdopen_failure(tmp_path, monkeypatch):
    path = tmp_path / "open-failure.json"

    class OpenFailure(RuntimeError):
        pass

    monkeypatch.setattr(
        relative_gate_module.os,
        "fdopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OpenFailure("fdopen failed")),
    )

    with pytest.raises(OpenFailure, match="fdopen failed"):
        relative_gate_module._write_matched_latest_gate_report(
            minimal_relative_report(), path, platform_name="posix"
        )

    assert list(tmp_path.glob(".open-failure.json.*.tmp")) == []


def test_report_writer_cleans_unique_temp_after_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "write-failure.json"
    original_fdopen = relative_gate_module.os.fdopen

    class WriteFailure(RuntimeError):
        pass

    class FailingWriteHandle:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, value):
            raise WriteFailure("write failed")

        def flush(self):
            self.handle.flush()

        def fileno(self):
            return self.handle.fileno()

    monkeypatch.setattr(
        relative_gate_module.os,
        "fdopen",
        lambda *args, **kwargs: FailingWriteHandle(
            original_fdopen(*args, **kwargs)
        ),
    )

    with pytest.raises(WriteFailure, match="write failed"):
        relative_gate_module._write_matched_latest_gate_report(
            minimal_relative_report(), path, platform_name="posix"
        )

    assert list(tmp_path.glob(".write-failure.json.*.tmp")) == []


def test_relative_runner_write_failure_leaves_live_shadow_state_undisposed(
    tmp_path, monkeypatch
):
    ema, latest = large_matched_gate_banks()
    before = train_module.centroid_bank_state_digest(latest)
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    def failed_write(report, report_path):
        raise OSError("durable publish failed")

    monkeypatch.setattr(
        train_module, "_write_matched_latest_gate_report", failed_write
    )

    with pytest.raises(OSError, match="durable publish failed"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            tmp_path / "failed.json",
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
        )

    assert train_module.centroid_bank_state_digest(latest) == before


def test_centroid_gate_unknown_explicit_version_fails_closed_with_strict_report(tmp_path):
    ema, latest = large_matched_gate_banks()
    config = relative_gate_config()
    config["gate_version"] = "matched_latest_v2"
    path = tmp_path / "unknown-version.json"

    with pytest.raises(AssertionError, match="unknown centroid gate_version"):
        train_module.run_centroid_ramp_gate(
            ema, config, path, latest_bank=latest
        )

    expected_strategy = (
        "windows_movefileex_replace_existing_write_through"
        if relative_gate_module.os.name == "nt"
        else "posix_temp_flush_fsync_replace_parent_fsync"
    )
    assert json.loads(path.read_text()) == {
        "gate_version": "matched_latest_v2",
        "passed": False,
        "failures": ["unknown_gate_version"],
        "failure": "unknown centroid gate_version: 'matched_latest_v2'",
        "persistence": {
            "strategy": expected_strategy,
            "durable_before_return": True,
        },
        "nonfinite_paths": [],
    }


@pytest.mark.parametrize(
    ("gate_version", "reported"),
    [("", ""), (float("nan"), "nan")],
)
def test_empty_or_nonfinite_explicit_gate_version_fails_closed_with_strict_report(
    tmp_path, gate_version, reported
):
    ema, latest = large_matched_gate_banks()
    config = {**relative_gate_config(), "gate_version": gate_version}
    path = tmp_path / f"invalid-version-{reported or 'empty'}.json"

    with pytest.raises(AssertionError, match="unknown centroid gate_version"):
        train_module.run_centroid_ramp_gate(ema, config, path, latest_bank=latest)

    report = json.loads(path.read_text())
    assert report["gate_version"] == reported
    assert report["failures"] == ["unknown_gate_version"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


def test_missing_and_null_gate_versions_preserve_identical_legacy_report_bytes(tmp_path):
    ema, _ = large_matched_gate_banks()
    config = gate_config()
    config.update(
        min_slide_updates=2,
        min_effective_rank=1.0,
        min_participation_ratio=1.0,
        max_mean_offdiag_cosine=0.999,
    )
    missing_path, null_path = tmp_path / "missing.json", tmp_path / "null.json"

    missing = train_module.run_centroid_ramp_gate(ema, config, missing_path)
    explicit_null = train_module.run_centroid_ramp_gate(
        ema, {**config, "gate_version": None}, null_path
    )

    assert missing == explicit_null
    assert missing_path.read_bytes() == null_path.read_bytes()
    assert "gate_version" not in missing
    assert "failures" not in missing


def test_relative_runner_names_missing_shadow_and_persists_null_unavailable_report(
    tmp_path,
):
    ema, _ = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "missing-shadow.json"

    with pytest.raises(AssertionError, match="latest_shadow_present"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=None,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
        )

    report = json.loads(path.read_text())
    assert report["passed"] is False
    assert "latest_shadow_present" in report["failures"]
    assert report["shadow"] == {
        "audit_time_present": False,
        "checkpoint_payload_present": False,
        "checkpoint_tensor_payload_bytes": 0,
        "state_step": None,
        "bank_state_digest": None,
        "post_pass_action": "none",
        "boundary_proposal": {
            "present": False,
            "type_exact": None,
            "transaction_valid": None,
            "committed_match": None,
            "state_step": None,
            "first_copy_excluded": True,
            "count": None,
            "mean": None,
            "q10": None,
            "q50": None,
            "q90": None,
        },
    }
    assert report["latest"]["spectrum"] is None
    assert all(value is None for value in report["relative"].values())
    assert report["permutation"]["alignments"] is None
    assert report["unavailable"] == [
        "latest_shadow",
        "latest_geometry:missing_shadow",
        "relative_geometry",
        "permutation",
    ]
    json.dumps(report, allow_nan=False)


def test_relative_runner_persists_primary_nonfinite_failure_as_null_not_nan(tmp_path):
    ema, latest = large_matched_gate_banks()
    with torch.no_grad():
        ema.slide_centroids[0, 0] = float("nan")
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "primary-nonfinite.json"

    with pytest.raises(AssertionError, match="ema_state_finite"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
        )

    report = json.loads(path.read_text())
    assert report["ema"]["spectrum"] is None
    assert report["relative"]["alignment"] is None
    assert report["permutation"]["alignments"] is None
    assert report["unavailable"] == [
        "ema_geometry:nonfinite",
        "relative_geometry",
        "permutation",
        "legacy_diagnostics",
    ]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "world_size",
    [float("nan"), torch.tensor(float("nan"))],
    ids=("python-float", "torch-scalar"),
)
def test_relative_runner_normalizes_nonfinite_world_size_and_persists_named_failure(
    tmp_path, world_size
):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "nonfinite-world-size.json"

    with pytest.raises(AssertionError, match="world_size_one"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=world_size,
        )

    report = json.loads(path.read_text())
    assert report["provenance"]["world_size"] is None
    assert "provenance.world_size" in report["nonfinite_paths"]
    assert "provenance.world_size:nonfinite" in report["unavailable"]
    assert "world_size_one" in report["failures"]
    assert (
        "audit_available:provenance.world_size:nonfinite" in report["failures"]
    )
    assert "report_nonfinite:provenance.world_size" in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("world_size", "expected_unavailable", "expected_nonfinite_paths"),
    [
        (
            torch.tensor([1.0, float("nan")]),
            (
                "provenance.world_size:non_scalar",
                "provenance.world_size:nonfinite",
            ),
            ("provenance.world_size[1]",),
        ),
        (
            np.asarray([1, 2], dtype=np.int64),
            ("provenance.world_size:non_scalar",),
            (),
        ),
        (
            np.float64(1.0),
            ("provenance.world_size:expected_integer",),
            (),
        ),
    ],
    ids=("torch-vector-nonfinite", "numpy-vector", "numpy-float-scalar"),
)
def test_relative_runner_normalizes_invalid_typed_world_size_before_boolean_checks(
    tmp_path, world_size, expected_unavailable, expected_nonfinite_paths
):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "invalid-typed-world-size.json"

    with pytest.raises(AssertionError, match="world_size_one"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=world_size,
        )

    report = json.loads(path.read_text())
    assert report["provenance"]["world_size"] is None
    assert all(reason in report["unavailable"] for reason in expected_unavailable)
    assert report["nonfinite_paths"] == list(expected_nonfinite_paths)
    assert "world_size_one" in report["failures"]
    assert all(
        f"audit_available:{reason}" in report["failures"]
        for reason in expected_unavailable
    )
    assert all(
        f"report_nonfinite:{path_name}" in report["failures"]
        for path_name in expected_nonfinite_paths
    )
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


def test_relative_runner_persists_invalid_world_size_environment(tmp_path, monkeypatch):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "invalid-environment-world-size.json"
    monkeypatch.setenv("WORLD_SIZE", "not-an-int")

    with pytest.raises(AssertionError, match="world_size_one"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
        )

    report = json.loads(path.read_text())
    reason = "provenance.world_size:expected_integer"
    assert report["provenance"]["world_size"] is None
    assert reason in report["unavailable"]
    assert "world_size_one" in report["failures"]
    assert f"audit_available:{reason}" in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


def test_relative_runner_persists_unreadable_scalar_integer_world_size(tmp_path):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "unreadable-world-size.json"

    with pytest.raises(AssertionError, match="world_size_one"):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=torch.empty((), dtype=torch.int64, device="meta"),
        )

    report = json.loads(path.read_text())
    reason = "provenance.world_size:unreadable"
    assert report["provenance"]["world_size"] is None
    assert reason in report["unavailable"]
    assert "world_size_one" in report["failures"]
    assert f"audit_available:{reason}" in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "world_size",
    [
        1,
        torch.tensor(1, dtype=torch.int64),
        np.int64(1),
        np.asarray(1, dtype=np.int64),
    ],
    ids=("python-int", "torch-scalar-int", "numpy-int", "numpy-zero-d-int"),
)
def test_relative_runner_accepts_typed_scalar_integer_world_size(tmp_path, world_size):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}

    report = train_module.run_centroid_ramp_gate(
        ema,
        relative_gate_config(),
        tmp_path / "valid-typed-world-size.json",
        latest_bank=latest,
        target_sha256=target_sha256,
        mapping_digest=mapping_digest,
        history_metadata=metadata,
        shadow_metadata=metadata,
        world_size=world_size,
    )

    assert type(report["provenance"]["world_size"]) is int
    assert report["provenance"]["world_size"] == 1
    assert report["passed"] is True


def test_completed_relative_report_recursively_nulls_nested_nonfinite_value(
    tmp_path, monkeypatch
):
    ema, latest = large_matched_gate_banks()
    target_sha256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    mapping_digest = "8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922"
    metadata = {"target_sha256": target_sha256, "mapping_digest": mapping_digest}
    path = tmp_path / "nested-nonfinite.json"
    original_evaluate = train_module.evaluate_matched_latest_gate

    def inject_nested_nonfinite(audit, history_cfg):
        report = original_evaluate(audit, history_cfg)
        report["shadow"]["synthetic_nested"] = {
            "values": np.asarray([1.0, float("inf")])
        }
        return report

    monkeypatch.setattr(
        train_module, "evaluate_matched_latest_gate", inject_nested_nonfinite
    )

    with pytest.raises(
        AssertionError,
        match=r"report_nonfinite:shadow\.synthetic_nested\.values\[1\]",
    ):
        train_module.run_centroid_ramp_gate(
            ema,
            relative_gate_config(),
            path,
            latest_bank=latest,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=metadata,
            shadow_metadata=metadata,
            world_size=1,
        )

    report = json.loads(path.read_text())
    path_name = "shadow.synthetic_nested.values[1]"
    assert report["shadow"]["synthetic_nested"]["values"] == [1.0, None]
    assert path_name in report["nonfinite_paths"]
    assert f"report_nonfinite:{path_name}" in report["failures"]
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


def snapshot_buffers(module):
    return {name: value.detach().clone() for name, value in module.named_buffers()}


def assert_buffers_unchanged(module, expected):
    actual = dict(module.named_buffers())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_latest_observation_bank_copies_each_teacher_value_without_gradient():
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    bank = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.0)
    first_teacher = hierarchical_means(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True),
        torch.tensor([0, 1]),
        mapping,
    )

    first = bank.propose(first_teacher)
    assert torch.equal(first.next_slide_centroids, first_teacher.slide_means.detach())
    assert not first.next_slide_centroids.requires_grad
    assert not first.patient_centroids.requires_grad
    bank.commit(first, step=1)

    second_teacher = hierarchical_means(
        torch.tensor([[11.0, 13.0], [17.0, 19.0]], requires_grad=True),
        torch.tensor([0, 1]),
        mapping,
    )
    second = bank.propose(second_teacher)
    assert torch.equal(second.next_slide_centroids, second_teacher.slide_means.detach())
    bank.commit(second, step=2)

    assert torch.equal(bank.slide_centroids, second_teacher.slide_means.detach())


def test_latest_observation_patient_pooling_weights_slides_equally_not_tiles():
    mapping = torch.tensor([0, 0], dtype=torch.int64)
    bank = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.0)
    teacher = hierarchical_means(
        torch.tensor([[1.0, 3.0], [9.0, 11.0], [9.0, 11.0], [9.0, 11.0]]),
        torch.tensor([0, 1, 1, 1]),
        mapping,
    )

    proposal = bank.propose(teacher)

    assert torch.equal(proposal.slide_tile_counts, torch.tensor([1, 3]))
    assert torch.equal(proposal.patient_centroids, torch.tensor([[5.0, 7.0]]))


def test_matched_updates_validate_both_proposals_before_either_bank_mutates():
    mapping = torch.tensor([0, 0, 1], dtype=torch.int64)
    ema = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.9)
    latest = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.0)
    teacher = hierarchical_means(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([0, 1, 2]),
        mapping,
    )
    ema_proposal, latest_proposal = propose_matched_centroid_updates(
        ema, latest, teacher
    )

    assert torch.equal(ema_proposal.slide_ids, latest_proposal.slide_ids)
    assert torch.equal(ema_proposal.slide_tile_counts, latest_proposal.slide_tile_counts)
    assert torch.equal(ema_proposal.patient_ids, latest_proposal.patient_ids)
    ema_before = snapshot_buffers(ema)
    latest_before = snapshot_buffers(latest)
    invalid_latest = latest_proposal._replace(
        patient_centroids=latest_proposal.patient_centroids.clone().fill_(float("nan"))
    )

    with pytest.raises(AssertionError):
        commit_matched_centroid_updates(
            ema,
            ema_proposal,
            latest,
            invalid_latest,
            step=1,
        )

    assert_buffers_unchanged(ema, ema_before)
    assert_buffers_unchanged(latest, latest_before)
    commit_matched_centroid_updates(
        ema,
        ema_proposal,
        latest,
        latest_proposal,
        step=1,
    )
    assert torch.equal(ema.slide_counts, latest.slide_counts)
    assert torch.equal(ema.slide_tile_presentations, latest.slide_tile_presentations)
    assert torch.equal(ema.centroid_state_step, latest.centroid_state_step)


def test_matched_updates_reject_unmatched_patient_observation_counts_before_proposal():
    mapping = torch.tensor([0, 0, 1], dtype=torch.int64)
    ema = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.9)
    latest = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.0)
    with torch.no_grad():
        latest.patient_slide_counts[0] = 1
    teacher = hierarchical_means(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([0, 1, 2]),
        mapping,
    )

    with pytest.raises(AssertionError):
        propose_matched_centroid_updates(ema, latest, teacher)


def test_deterministic_grouped_sum_preserves_values_output_order_and_gradients():
    values = torch.tensor(
        [
            [7.0, 11.0],
            [1.0, 2.0],
            [5.0, 8.0],
            [3.0, 4.0],
            [13.0, 17.0],
            [19.0, 23.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    groups = torch.tensor([2, 0, 1, 0, 2, 1], dtype=torch.int64)
    expected = torch.tensor(
        [[4.0, 6.0], [24.0, 31.0], [20.0, 28.0]], dtype=torch.float32
    )

    actual = deterministic_grouped_sum(values, groups, group_count=3)
    permutation = torch.tensor([5, 3, 0, 2, 1, 4])
    permuted = deterministic_grouped_sum(
        values.detach()[permutation], groups[permutation], group_count=3
    )

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(permuted, expected)
    weights = torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]])
    (actual * weights).sum().backward()
    assert torch.equal(values.grad, weights[groups])


def test_deterministic_grouped_sum_rejects_missing_groups():
    with pytest.raises(AssertionError):
        deterministic_grouped_sum(
            torch.ones(2, 3), torch.tensor([0, 2]), group_count=3
        )


def test_trusted_dense_grouped_sum_gathers_unique_groups_in_order_with_autograd(
    monkeypatch,
):
    def unexpected_device_value_operation(*args, **kwargs):
        raise AssertionError("trusted unique groups must not inspect device values")

    monkeypatch.setattr(torch, "isfinite", unexpected_device_value_operation)
    monkeypatch.setattr(torch, "all", unexpected_device_value_operation)
    monkeypatch.setattr(torch, "segment_reduce", unexpected_device_value_operation)
    values = torch.tensor(
        [[7.0, 11.0], [1.0, 2.0], [13.0, 17.0], [5.0, 8.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    groups = torch.tensor([2, 0, 3, 1], dtype=torch.int64)

    actual = deterministic_grouped_sum(
        values, groups, group_count=4, trusted_dense=True
    )

    expected = torch.tensor(
        [[1.0, 2.0], [5.0, 8.0], [7.0, 11.0], [13.0, 17.0]],
        dtype=torch.float32,
    )
    assert torch.equal(actual, expected)
    weights = torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0]])
    (actual * weights).sum().backward()
    assert torch.equal(values.grad, weights[groups])


def test_trusted_dense_grouped_sum_collisions_are_deterministic_with_autograd(
    monkeypatch,
):
    def unexpected_device_value_operation(*args, **kwargs):
        raise AssertionError("trusted dense groups must not inspect device values")

    monkeypatch.setattr(torch, "isfinite", unexpected_device_value_operation)
    monkeypatch.setattr(torch, "all", unexpected_device_value_operation)
    values = torch.tensor(
        [[7.0, 11.0], [1.0, 2.0], [5.0, 8.0], [3.0, 4.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    groups = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual = deterministic_grouped_sum(
            values, groups, group_count=2, trusted_dense=True
        )
        repeated = deterministic_grouped_sum(
            values.detach(), groups, group_count=2, trusted_dense=True
        )
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )

    expected = torch.tensor([[4.0, 6.0], [12.0, 19.0]], dtype=torch.float32)
    assert torch.equal(actual, expected)
    assert torch.equal(repeated, expected)
    weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    (actual * weights).sum().backward()
    assert torch.equal(values.grad, weights[groups])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_production_shaped_cuda_centroid_commits_are_deterministic_and_match_cpu_oracle():
    device = torch.device("cuda")
    batch_size, views, feature_dim = 128, 2, 1536
    slide_to_patient = torch.cat(
        (
            torch.tensor([0, 0, 1, 1], device=device),
            torch.arange(2, 126, device=device),
        )
    ).long()
    slide_ids = torch.arange(batch_size, device=device)
    generator = torch.Generator(device=device).manual_seed(7777)
    crop_major_steps = [
        2.0
        * torch.randn(
            views, batch_size, feature_dim, generator=generator, device=device
        )
        for _ in range(2)
    ]
    for crop_major in crop_major_steps:
        crop_major.requires_grad_()

    def run_two_steps(*, backward=False):
        bank = HierarchicalCentroidBank(
            slide_to_patient, feature_dim=feature_dim, momentum=0.9
        )
        for step, crop_major in enumerate(crop_major_steps, start=1):
            tile_means = crop_major_tile_mean(
                crop_major.flatten(0, 1), views, batch_size
            )
            teacher = hierarchical_means(tile_means, slide_ids, slide_to_patient)
            bank.commit(bank.propose(teacher), step=step)
            if backward and step == 2:
                teacher.patient_means.square().sum().backward()
        return bank

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = run_two_steps(backward=True)
        second = run_two_steps()
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )

    for name in (
        "slide_centroids",
        "slide_counts",
        "slide_tile_presentations",
        "centroid_state_step",
        "patient_sums",
        "patient_slide_counts",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert first.centroid_state_step.item() == 2
    assert crop_major_steps[1].grad is not None
    assert torch.isfinite(crop_major_steps[1].grad).all()
    assert crop_major_steps[1].grad.norm() > 0

    oracle_sums = torch.zeros(126, feature_dim, dtype=torch.float64)
    oracle_counts = torch.zeros(126, dtype=torch.int64)
    centroids = first.slide_centroids.detach().cpu().double()
    mapping = slide_to_patient.cpu()
    for slide_id, patient_id in enumerate(mapping.tolist()):
        oracle_sums[patient_id] += centroids[slide_id]
        oracle_counts[patient_id] += 1
    torch.testing.assert_close(
        first.patient_sums.detach().cpu().double(), oracle_sums, rtol=0, atol=2e-5
    )
    assert torch.equal(first.patient_slide_counts.cpu(), oracle_counts)


def initial_teacher(*, requires_grad=False):
    return hierarchical_means(
        torch.tensor([[2.0], [4.0], [10.0]], requires_grad=requires_grad),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 0]),
    )


STATE_NAMES = (
    "slide_centroids",
    "slide_counts",
    "slide_tile_presentations",
    "centroid_state_step",
)

HISTORY_METADATA = {
    "version": 1,
    "arm": "centroid",
    "target_sha256": "a" * 64,
    "mapping_digest": "b" * 64,
    "feature_blocks": (4, 6, 8, 11),
    "feature_width": 2,
    "momentum": 0.9,
    "hierarchy": "slide_then_patient",
    "ste": "student_identity_ste",
    "weight": 0.03,
    "ramp_start": 0.5,
    "ramp_len": 0.25,
}


def committed_state_bank():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0, 1]), feature_dim=2, momentum=0.9)
    first = hierarchical_means(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [3.0, 3.0]]),
        torch.tensor([0, 1, 2]),
        bank.slide_to_patient,
    )
    bank.commit(bank.propose(first), step=1)
    second = hierarchical_means(
        torch.tensor([[12.0, 0.0], [12.0, 0.0]]),
        torch.tensor([0, 0]),
        bank.slide_to_patient,
    )
    bank.commit(bank.propose(second), step=2)
    return bank


def dirty_restore_target():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0, 1]), feature_dim=2, momentum=0.9)
    teacher = hierarchical_means(
        torch.tensor([[9.0, 1.0], [5.0, 7.0]]),
        torch.tensor([1, 2]),
        bank.slide_to_patient,
    )
    bank.commit(bank.propose(teacher), step=1)
    return bank


def production_shaped_checkpoint_bank_with_natural_cache_drift():
    feature_dim = 1536
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim, momentum=0.9)

    # A valid 7,813-step EMA history: copy slide 0 at step 1 and slide 1 at
    # step 2, then move one slide coordinate upward by one float32 ULP per step.
    # Each EMA teacher value is solved so the registered 0.9/0.1 update reaches
    # exactly that next float32 value. Adding half-ULP deltas to the 1.5 cache
    # rounds away while the authoritative slide value continues to advance.
    value = torch.tensor(0.5, dtype=torch.float32)
    incremental_sum = torch.tensor(1.5, dtype=torch.float32)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
    for _ in range(3, 7_814):
        next_value = torch.nextafter(value, positive_infinity)
        teacher_value = (next_value - 0.9 * value) / 0.1
        assert torch.equal(0.9 * value + 0.1 * teacher_value, next_value)
        incremental_sum += next_value - value
        value = next_value

    slide_zero = torch.zeros(feature_dim, dtype=torch.float32)
    slide_one = torch.zeros(feature_dim, dtype=torch.float32)
    for block_start in range(0, feature_dim, 384):
        slide_zero[block_start] = 1.0
        slide_one[block_start] = value
        slide_one[block_start + 1] = torch.sqrt(1.0 - value.square())
    bank.slide_centroids.copy_(torch.stack((slide_zero, slide_one)))
    bank.slide_counts.copy_(torch.tensor([1, 7_812]))
    bank.slide_tile_presentations.copy_(bank.slide_counts)
    bank.centroid_state_step.fill_(7_813)
    bank.patient_sums.copy_(bank.slide_centroids.sum(dim=0, keepdim=True))
    for block_start in range(0, feature_dim, 384):
        bank.patient_sums[0, block_start] = incremental_sum
    bank.patient_slide_counts.fill_(2)
    return bank


def assert_restore_rejected_atomically(payload, *, expected_metadata=None, expected_step=2):
    bank = dirty_restore_target()
    before = snapshot_buffers(bank)
    with pytest.raises(AssertionError):
        bank.restore_state(
            payload,
            HISTORY_METADATA if expected_metadata is None else expected_metadata,
            expected_step,
        )
    assert_buffers_unchanged(bank, before)


def test_hierarchy_uses_tile_mean_then_equal_slide_mean_and_is_order_invariant():
    features = torch.tensor([[0.0], [0.0], [6.0], [10.0]])
    slides = torch.tensor([0, 0, 1, 2])
    slide_to_patient = torch.tensor([0, 0, 1])

    first = hierarchical_means(features, slides, slide_to_patient)
    second = hierarchical_means(features.flip(0), slides.flip(0), slide_to_patient)

    torch.testing.assert_close(first.slide_ids, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(first.slide_means, torch.tensor([[0.0], [6.0], [10.0]]))
    torch.testing.assert_close(first.patient_ids, torch.tensor([0, 1]))
    torch.testing.assert_close(first.patient_means, torch.tensor([[3.0], [10.0]]))
    torch.testing.assert_close(first.patient_means, second.patient_means)
    torch.testing.assert_close(first.slide_tile_counts, torch.tensor([2, 1, 1]))


def test_hierarchy_rejects_an_empty_batch():
    with pytest.raises(AssertionError):
        hierarchical_means(
            torch.empty((0, 1)),
            torch.empty(0, dtype=torch.int64),
            torch.tensor([0]),
        )


@pytest.mark.parametrize("slide_ids", [torch.tensor([-1]), torch.tensor([2])])
def test_hierarchy_rejects_negative_or_out_of_bounds_slide_ids(slide_ids):
    with pytest.raises(AssertionError):
        hierarchical_means(torch.tensor([[1.0]]), slide_ids, torch.tensor([0, 1]))


def test_crop_major_views_restore_tile_identity():
    crop_major = torch.tensor([[1.0], [2.0], [3.0], [5.0]])

    torch.testing.assert_close(
        crop_major_tile_mean(crop_major, views=2, batch_size=2),
        torch.tensor([[2.0], [3.5]]),
    )


def test_teacher_forward_student_identity_gradient():
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[7.0, 11.0]], requires_grad=True)

    routed = teacher_value_student_gradient(student, teacher)

    torch.testing.assert_close(routed, teacher)
    routed.backward(torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(student.grad, torch.tensor([[3.0, 5.0]]))
    assert teacher.grad is None


def test_teacher_forward_is_exact_under_extreme_student_scale():
    student = torch.tensor([[1.0e20]], requires_grad=True)
    teacher = torch.tensor([[1.0]], requires_grad=True)

    routed = teacher_value_student_gradient(student, teacher)

    assert torch.equal(routed, teacher.detach())
    routed.sum().backward()
    torch.testing.assert_close(student.grad, torch.ones_like(student))
    assert teacher.grad is None


def test_teacher_student_route_rejects_broadcastable_shape_mismatch():
    with pytest.raises(AssertionError):
        teacher_value_student_gradient(torch.ones(2, 3), torch.ones(1, 3))


def test_patient_targets_group_once_and_require_consistency():
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    present = torch.ones(3)
    tile_patients = torch.tensor([0, 0, 1])
    patient_ids = torch.tensor([0, 1])

    grouped, grouped_present = patient_targets_from_tiles(
        targets, present, tile_patients, patient_ids
    )

    torch.testing.assert_close(grouped, torch.eye(2))
    torch.testing.assert_close(grouped_present, torch.ones(2))


def test_patient_targets_select_a_bitwise_representative_instead_of_averaging():
    repeated = torch.tensor([-1.5255959033966064, 0.0], dtype=torch.float32)
    targets = torch.stack([repeated] * 7 + [torch.tensor([0.0, 1.0])])
    grouped, _ = patient_targets_from_tiles(
        targets,
        torch.ones(8),
        torch.tensor([0] * 7 + [1]),
        torch.tensor([0, 1]),
    )

    assert torch.equal(grouped[0], repeated)


def test_patient_targets_reject_even_one_ulp_of_within_patient_disagreement():
    first = torch.tensor([1.0, 0.0])
    second = first.clone()
    second[0] = torch.nextafter(first[0], torch.tensor(2.0))

    with pytest.raises(AssertionError):
        patient_targets_from_tiles(
            torch.stack((first, second)),
            torch.ones(2),
            torch.tensor([0, 0]),
            torch.tensor([0]),
        )


@pytest.mark.parametrize(
    ("targets", "present"),
    [
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.ones(2)),
        (torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([1.0, 0.0])),
    ],
    ids=["targets", "present"],
)
def test_patient_targets_reject_inconsistent_per_patient_rows(targets, present):
    with pytest.raises(AssertionError):
        patient_targets_from_tiles(
            targets,
            present,
            torch.tensor([0, 0]),
            torch.tensor([0]),
        )


def test_patient_targets_reject_nonbinary_presence():
    with pytest.raises(AssertionError):
        patient_targets_from_tiles(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([0, 0]),
            torch.tensor([0]),
        )


def test_patient_targets_require_float32_inputs():
    with pytest.raises(AssertionError):
        patient_targets_from_tiles(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float16),
            torch.ones(2, dtype=torch.float16),
            torch.tensor([0, 0]),
            torch.tensor([0]),
        )


def test_propose_rejects_wrong_patient_id_dtype_without_mutation():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    teacher = initial_teacher()._replace(patient_ids=torch.tensor([0.0]))
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.propose(teacher)

    assert_buffers_unchanged(bank, before)


def test_propose_is_pure_detached_and_finite_without_history():
    teacher = initial_teacher(requires_grad=True)
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    before = snapshot_buffers(bank)

    proposal = bank.propose(teacher)

    assert_buffers_unchanged(bank, before)
    assert teacher.slide_means.requires_grad
    for value in (
        proposal.next_slide_centroids,
        proposal.patient_centroids,
        proposal.drift_cosines,
        proposal.historical_tile_fraction,
    ):
        assert not value.requires_grad
        assert torch.isfinite(value).all()
    assert proposal.drift_cosines.numel() == 0
    assert proposal.historical_tile_fraction.ndim == 0
    assert proposal.historical_tile_fraction.item() == 0.0
    torch.testing.assert_close(proposal.patient_centroids, torch.tensor([[6.5]]))


def test_bank_first_copies_then_uses_point_nine_ema_and_equal_slide_patient_mean():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    first = bank.propose(initial_teacher())

    assert first.base_state_step == 0
    torch.testing.assert_close(first.next_slide_centroids, torch.tensor([[3.0], [10.0]]))
    assert bank.slide_counts.sum() == 0
    bank.commit(first, step=1)
    torch.testing.assert_close(bank.slide_counts, torch.tensor([1, 1]))
    torch.testing.assert_close(bank.slide_tile_presentations, torch.tensor([2, 1]))

    second_teacher = hierarchical_means(
        torch.tensor([[13.0]]), torch.tensor([0]), torch.tensor([0, 0])
    )
    second = bank.propose(second_teacher)

    torch.testing.assert_close(second.next_slide_centroids, torch.tensor([[4.0]]))
    torch.testing.assert_close(second.patient_centroids, torch.tensor([[7.0]]))
    torch.testing.assert_close(second.historical_tile_fraction, torch.tensor(1.0))
    assert second.drift_cosines.shape == (1,)
    bank.commit(second, step=2)
    committed = snapshot_buffers(bank)
    with pytest.raises(AssertionError):
        bank.commit(second, step=2)
    assert_buffers_unchanged(bank, committed)
    assert bank.centroid_state_step.item() == 2


def test_commit_requires_the_next_state_step_before_any_mutation():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    proposal = bank.propose(initial_teacher())
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.commit(proposal, step=2)

    assert_buffers_unchanged(bank, before)


def test_commit_rejects_unrepresentable_state_step_atomically():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    maximum = torch.iinfo(torch.int64).max
    bank.centroid_state_step.fill_(maximum)
    proposal = bank.propose(initial_teacher())
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.commit(proposal, step=maximum + 1)

    assert_buffers_unchanged(bank, before)


def test_commit_rejects_tile_presentation_overflow_atomically():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    proposal = bank.propose(initial_teacher())
    bank.slide_tile_presentations[0] = torch.iinfo(torch.int64).max
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.commit(proposal, step=1)

    assert_buffers_unchanged(bank, before)


def test_commit_rejects_update_count_overflow_atomically():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    bank.slide_centroids.copy_(torch.tensor([[3.0], [10.0]]))
    bank.slide_counts.copy_(torch.tensor([torch.iinfo(torch.int64).max, 1]))
    bank.patient_sums[0] = 13.0
    bank.patient_slide_counts[0] = 2
    proposal = bank.propose(initial_teacher())
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.commit(proposal, step=1)

    assert_buffers_unchanged(bank, before)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_state_step", -1),
        ("slide_ids", torch.tensor([-1, 1])),
        ("slide_ids", torch.tensor([0, 2])),
        ("slide_ids", torch.tensor([0, 0])),
        ("next_slide_centroids", torch.tensor([[3.0], [float("nan")]])),
        ("next_slide_centroids", torch.tensor([[3.0], [10.0]], requires_grad=True)),
        ("slide_tile_counts", torch.tensor([2, 0])),
        ("slide_tile_counts", torch.tensor([2.0, 1.0])),
        ("patient_ids", torch.tensor([1])),
        ("patient_ids", torch.tensor([0], dtype=torch.int32)),
        ("patient_centroids", torch.tensor([[6.0]])),
        ("drift_cosines", torch.tensor([0.0])),
        ("historical_tile_fraction", torch.tensor(0.5)),
    ],
    ids=[
        "stale-base",
        "negative-slide",
        "oob-slide",
        "duplicate-slide",
        "nonfinite-centroid",
        "attached-centroid",
        "zero-tile-count",
        "float-tile-count",
        "wrong-patient",
        "wrong-patient-dtype",
        "wrong-patient-centroid",
        "wrong-drift-shape",
        "wrong-history-fraction",
    ],
)
def test_commit_rejects_malformed_proposals_atomically(field, value):
    bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
    proposal = bank.propose(initial_teacher())._replace(**{field: value})
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.commit(proposal, step=1)

    assert_buffers_unchanged(bank, before)


def test_centroid_geometry_matches_independent_float64_nonunit_oracle_and_row_permutation():
    centroids = torch.tensor(
        [
            [2.0, 0.0, 1.0],
            [0.0, 1.0, -2.0],
            [-3.0, -1.0, 0.5],
            [4.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    before = centroids.clone()

    metrics = centroid_geometry(centroids)
    permuted = centroid_geometry(centroids[torch.tensor([2, 0, 3, 1])])

    raw = centroids.numpy().astype(np.float64)
    centered = raw - raw.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(raw) - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    positive = eigenvalues[eigenvalues > 0]
    probabilities = positive / eigenvalues.sum()
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    expected = {
        "patient_count": len(raw),
        "min_norm": np.linalg.norm(raw, axis=1).min(),
        "effective_rank": np.exp(-(probabilities * np.log(probabilities)).sum()),
        "participation_ratio": eigenvalues.sum() ** 2 / np.square(eigenvalues).sum(),
        "mean_offdiag_cosine": (
            np.square(unit.sum(axis=0)).sum() - len(unit)
        )
        / (len(unit) * (len(unit) - 1)),
    }

    assert expected["mean_offdiag_cosine"] < 0
    assert metrics.keys() == expected.keys()
    for name, value in expected.items():
        assert metrics[name] == pytest.approx(value, rel=1e-12, abs=1e-12), name
        assert permuted[name] == pytest.approx(value, rel=1e-12, abs=1e-12), name
    assert torch.equal(centroids, before)


@pytest.mark.parametrize(
    "centroids",
    [
        torch.tensor([1.0, 2.0]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
    ],
    ids=["not-matrix", "one-patient", "nonfinite", "zero-norm", "zero-variance"],
)
def test_centroid_geometry_rejects_invalid_populations(centroids):
    with pytest.raises(AssertionError):
        centroid_geometry(centroids)


def committed_audit_bank():
    bank = HierarchicalCentroidBank(torch.tensor([0, 0, 1, 2]), feature_dim=3, momentum=0.9)
    first = hierarchical_means(
        torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
        torch.tensor([0, 1, 2, 3]),
        bank.slide_to_patient,
    )
    bank.commit(bank.propose(first), step=1)
    second = hierarchical_means(
        torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ]
        ),
        torch.tensor([0, 0, 2, 2, 2]),
        bank.slide_to_patient,
    )
    bank.commit(bank.propose(second), step=2)
    return bank


def test_patient_centroids_and_audit_separate_all_observed_from_mature_only():
    bank = committed_audit_bank()

    observed_ids, observed = bank.patient_centroids(min_slide_updates=1)
    mature_ids, mature = bank.patient_centroids(min_slide_updates=2)
    audit = centroid_audit(bank, min_slide_updates=2)

    assert observed.device.type == mature.device.type == "cpu"
    assert observed.dtype == mature.dtype == torch.float64
    assert torch.equal(observed_ids, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(
        observed,
        torch.tensor([[1.0, 0.5, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64),
    )
    assert torch.equal(mature_ids, torch.tensor([0, 1]))
    torch.testing.assert_close(
        mature,
        torch.tensor([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float64),
    )
    assert audit["all_observed"] == centroid_geometry(observed)
    assert audit["mature_only"] == centroid_geometry(mature)
    assert audit["all_observed"]["patient_count"] == 3
    assert audit["mature_only"]["patient_count"] == 2
    assert audit["population_sizes"] == {
        "mature_min_slide_updates": 2,
        "observed_slides": 4,
        "mature_slides": 2,
        "observed_patients": 3,
        "mature_patients": 2,
    }
    assert audit["slide_update_count_distribution"] == {
        "population": "observed_slides",
        "count": 4,
        "mean": 1.5,
        "q0": 1.0,
        "q25": 1.0,
        "q50": 1.5,
        "q75": 2.0,
        "q100": 2.0,
    }
    assert audit["observed_slides_per_patient_distribution"] == {
        "population": "observed_patients",
        "count": 3,
        "mean": pytest.approx(4 / 3),
        "q0": 1.0,
        "q25": 1.0,
        "q50": 1.0,
        "q75": 1.5,
        "q100": 2.0,
    }
    assert audit["boundary_teacher_centroid_drift"] == {
        "first_copy_excluded": True,
        "count": 0,
        "mean": None,
        "q10": None,
        "q50": None,
        "q90": None,
    }


def test_sample_weighted_mature_coverage_uses_tile_presentations():
    bank = committed_audit_bank()
    expected = (
        bank.slide_tile_presentations[bank.slide_counts >= 2].sum()
        / bank.slide_tile_presentations.sum()
    )

    assert bank.sample_weighted_mature_coverage(2) == pytest.approx(float(expected))
    assert bank.sample_weighted_mature_coverage(2) == pytest.approx(7 / 9)


def test_sample_weighted_mature_coverage_rejects_empty_or_invalid_requests():
    bank = HierarchicalCentroidBank(torch.tensor([0, 1]), feature_dim=2, momentum=0.9)
    with pytest.raises(AssertionError):
        bank.sample_weighted_mature_coverage(2)
    with pytest.raises(AssertionError):
        committed_audit_bank().sample_weighted_mature_coverage(0)


def passing_gate_audit():
    return {
        "sample_weighted_mature_coverage": 0.95,
        "all_observed": {
            "patient_count": 512,
            "min_norm": math.nextafter(1.0e-6, math.inf),
            "effective_rank": 32.0,
            "participation_ratio": 16.0,
            "mean_offdiag_cosine": math.nextafter(0.95, -math.inf),
        },
        "mature_only": {
            "patient_count": 2,
            "min_norm": 1.0e-12,
            "effective_rank": 1.0,
            "participation_ratio": 1.0,
            "mean_offdiag_cosine": 1.0,
        },
    }


def gate_config():
    return {
        "min_sample_weighted_coverage": 0.95,
        "min_geometry_patients": 512,
        "min_effective_rank": 32.0,
        "min_participation_ratio": 16.0,
        "max_mean_offdiag_cosine": 0.95,
        "min_centroid_norm": 1.0e-6,
    }


def set_audit_metric(audit, name, value):
    if name == "sample_weighted_mature_coverage":
        audit[name] = value
    else:
        audit["all_observed"][name] = value


def test_centroid_gate_uses_all_observed_hard_population_and_inclusive_registered_boundaries():
    require_centroid_gate(passing_gate_audit(), gate_config())


def test_centroid_audit_keeps_an_insufficient_mature_population_diagnostic_only():
    bank = HierarchicalCentroidBank(torch.tensor([0, 1]), feature_dim=2, momentum=0.9)
    first = hierarchical_means(
        torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        torch.tensor([0, 1]),
        bank.slide_to_patient,
    )
    bank.commit(bank.propose(first), step=1)
    second = hierarchical_means(
        torch.tensor([[2.0, 0.0]]).repeat(19, 1),
        torch.zeros(19, dtype=torch.int64),
        bank.slide_to_patient,
    )
    bank.commit(bank.propose(second), step=2)

    audit = centroid_audit(bank, min_slide_updates=2)

    assert audit["sample_weighted_mature_coverage"] == pytest.approx(20 / 21)
    assert audit["all_observed"]["patient_count"] == 2
    assert audit["mature_only"] == {
        "patient_count": 1,
        "min_norm": 2.0,
        "effective_rank": None,
        "participation_ratio": None,
        "mean_offdiag_cosine": None,
    }
    require_centroid_gate(
        audit,
        {
            "min_sample_weighted_coverage": 0.95,
            "min_geometry_patients": 2,
            "min_effective_rank": 1.0,
            "min_participation_ratio": 1.0,
            "max_mean_offdiag_cosine": 0.1,
            "min_centroid_norm": 1.0,
        },
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sample_weighted_mature_coverage", math.nextafter(0.95, -math.inf)),
        ("patient_count", 511),
        ("effective_rank", math.nextafter(32.0, -math.inf)),
        ("participation_ratio", math.nextafter(16.0, -math.inf)),
        ("mean_offdiag_cosine", 0.95),
        ("min_norm", 1.0e-6),
    ],
    ids=["coverage", "patients", "effective-rank", "participation", "cosine", "norm"],
)
def test_centroid_gate_rejects_every_registered_threshold_failure(name, value):
    audit = passing_gate_audit()
    set_audit_metric(audit, name, value)

    with pytest.raises(AssertionError):
        require_centroid_gate(audit, gate_config())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sample_weighted_mature_coverage", math.inf),
        ("effective_rank", math.inf),
        ("participation_ratio", math.inf),
        ("mean_offdiag_cosine", -math.inf),
        ("min_norm", math.inf),
    ],
    ids=["coverage", "effective-rank", "participation", "cosine", "norm"],
)
def test_centroid_gate_rejects_nonfinite_metrics_even_when_they_pass_inequalities(name, value):
    audit = passing_gate_audit()
    set_audit_metric(audit, name, value)

    with pytest.raises(AssertionError):
        require_centroid_gate(audit, gate_config())


def test_export_state_has_exact_authoritative_cpu_clone_payload():
    bank = committed_state_bank()
    metadata = dict(HISTORY_METADATA)

    payload = bank.export_state(metadata)

    assert type(payload) is dict
    assert tuple(payload) == ("metadata", *STATE_NAMES)
    assert payload["metadata"] == metadata
    assert payload["metadata"] is not metadata
    for name in STATE_NAMES:
        exported = payload[name]
        live = getattr(bank, name)
        assert exported.device.type == "cpu"
        assert exported.dtype == live.dtype and exported.shape == live.shape
        assert not exported.requires_grad
        assert torch.equal(exported, live.cpu())
        if live.device.type == "cpu":
            assert exported.data_ptr() != live.data_ptr()
    payload["metadata"]["version"] = 99
    payload["slide_centroids"].zero_()
    assert metadata["version"] == 1
    assert torch.count_nonzero(bank.slide_centroids) > 0


def test_export_state_rejects_a_noncanonical_float_bank():
    bank = committed_state_bank().half()
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.export_state(HISTORY_METADATA)

    assert_buffers_unchanged(bank, before)


def test_checkpoint_export_canonicalizes_derived_cache_for_exact_resume_behavior():
    source = production_shaped_checkpoint_bank_with_natural_cache_drift()
    metadata = {**HISTORY_METADATA, "feature_width": 1536}
    canonical_sum = source.slide_centroids.sum(dim=0, keepdim=True)
    before_drift = float((source.patient_sums - canonical_sum).abs().max())
    authoritative_before = {name: getattr(source, name).clone() for name in STATE_NAMES}
    block_view = source.slide_centroids.reshape(2, 4, 384)
    torch.testing.assert_close(block_view.norm(dim=-1), torch.ones(2, 4))
    teacher = hierarchical_means(
        source.slide_centroids[1:2].clone(),
        torch.tensor([1]),
        source.slide_to_patient,
    )
    drifted_next = source.propose(teacher)

    payload = source.export_state(metadata)
    restored = HierarchicalCentroidBank(torch.tensor([0, 0]), 1536, momentum=0.9)
    restored.restore_state(payload, metadata, expected_step=7_813)
    uninterrupted_next = source.propose(teacher)
    restored_next = restored.propose(teacher)
    pre_export_next_drift = float(
        (drifted_next.patient_centroids - restored_next.patient_centroids).abs().max()
    )

    assert before_drift == pytest.approx(0.00046563148498535156)
    assert pre_export_next_drift > 0.0002
    for name, expected in authoritative_before.items():
        assert torch.equal(getattr(source, name), expected), name
        assert torch.equal(payload[name], expected), name
    assert torch.equal(source.patient_sums, canonical_sum)
    assert torch.equal(source.patient_sums, restored.patient_sums)
    assert torch.equal(source.patient_slide_counts, restored.patient_slide_counts)
    for field in uninterrupted_next._fields:
        assert torch.equal(
            torch.as_tensor(getattr(uninterrupted_next, field)),
            torch.as_tensor(getattr(restored_next, field)),
        ), field


def test_failed_export_metadata_validation_does_not_canonicalize_any_buffer():
    bank = production_shaped_checkpoint_bank_with_natural_cache_drift()
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.export_state(tuple(HISTORY_METADATA.items()))

    assert_buffers_unchanged(bank, before)


def test_restore_round_trip_is_bitwise_and_rebuilds_caches_for_matching_next_proposal():
    source = committed_state_bank()
    payload = source.export_state(HISTORY_METADATA)
    restored = dirty_restore_target()
    restored.patient_sums.fill_(float("nan"))
    restored.patient_slide_counts.fill_(99)

    restored.restore_state(payload, HISTORY_METADATA, expected_step=2)

    for name in STATE_NAMES:
        assert torch.equal(getattr(restored, name), getattr(source, name)), name
    torch.testing.assert_close(restored.patient_sums, torch.tensor([[3.0, 2.0], [3.0, 3.0]]))
    assert torch.equal(restored.patient_slide_counts, torch.tensor([2, 1]))
    next_teacher = hierarchical_means(
        torch.tensor([[8.0, 4.0], [7.0, 9.0]]),
        torch.tensor([1, 2]),
        source.slide_to_patient,
    )
    expected = source.propose(next_teacher)
    actual = restored.propose(next_teacher)
    assert expected.base_state_step == actual.base_state_step
    for field in expected._fields[1:]:
        assert torch.equal(getattr(expected, field), getattr(actual, field)), field


def test_export_then_restore_canonicalizes_realistic_float_cache_drift_exactly():
    generator = torch.Generator().manual_seed(41)
    mapping = torch.tensor([0, 0, 0, 1, 1, 2])
    source = HierarchicalCentroidBank(mapping, feature_dim=4, momentum=0.9)
    for step in range(1, 21):
        slide_ids = torch.randperm(len(mapping), generator=generator)[:3].sort().values
        teacher = hierarchical_means(
            torch.randn(3, 4, generator=generator) * 10,
            slide_ids,
            mapping,
        )
        source.commit(source.propose(teacher), step=step)
    payload = source.export_state(HISTORY_METADATA)
    restored = HierarchicalCentroidBank(mapping, feature_dim=4, momentum=0.9)
    restored.restore_state(payload, HISTORY_METADATA, expected_step=20)
    assert torch.equal(source.patient_sums, restored.patient_sums)
    assert torch.equal(source.patient_slide_counts, restored.patient_slide_counts)
    next_teacher = hierarchical_means(
        torch.randn(3, 4, generator=generator),
        torch.tensor([0, 3, 5]),
        mapping,
    )

    expected = source.propose(next_teacher)
    actual = restored.propose(next_teacher)

    assert expected.base_state_step == actual.base_state_step
    for field in expected._fields[1:]:
        assert torch.equal(getattr(expected, field), getattr(actual, field)), field


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "not-dict"])
def test_restore_rejects_every_payload_schema_mismatch_atomically(mutation):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    if mutation == "missing":
        payload.pop("slide_counts")
    elif mutation == "unexpected":
        payload["patient_sums"] = torch.zeros(2, 2)
    else:
        payload = list(payload.items())

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize("name", ("metadata", *STATE_NAMES))
def test_restore_rejects_every_non_python_string_payload_key_atomically(name):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    value = payload.pop(name)
    wrong_type_key = np.str_(name)
    assert wrong_type_key == name and type(wrong_type_key) is not str
    payload[wrong_type_key] = value

    assert_restore_rejected_atomically(payload)


def test_restore_rejects_a_self_consistent_noncanonical_float_bank_atomically():
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    payload["slide_centroids"] = payload["slide_centroids"].half()
    bank = dirty_restore_target().half()
    before = snapshot_buffers(bank)

    with pytest.raises(AssertionError):
        bank.restore_state(payload, HISTORY_METADATA, expected_step=2)

    assert_buffers_unchanged(bank, before)


@pytest.mark.parametrize("field", tuple(HISTORY_METADATA), ids=tuple(HISTORY_METADATA))
def test_restore_rejects_each_metadata_value_mismatch_atomically(field):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    payload["metadata"][field] = object()

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize(
    ("field", "equal_value_with_wrong_type"),
    [
        ("version", True),
        ("arm", np.str_("centroid")),
        ("target_sha256", np.str_("a" * 64)),
        ("mapping_digest", np.str_("b" * 64)),
        ("feature_blocks", tuple(np.int64(value) for value in (4, 6, 8, 11))),
        ("feature_width", 2.0),
        ("momentum", np.float64(0.9)),
        ("hierarchy", np.str_("slide_then_patient")),
        ("ste", np.str_("student_identity_ste")),
        ("weight", np.float64(0.03)),
        ("ramp_start", np.float64(0.5)),
        ("ramp_len", np.float64(0.25)),
    ],
    ids=tuple(HISTORY_METADATA),
)
def test_restore_rejects_metadata_type_only_drift_atomically(field, equal_value_with_wrong_type):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    assert payload["metadata"][field] == equal_value_with_wrong_type
    if isinstance(equal_value_with_wrong_type, tuple):
        assert any(
            type(actual) is not type(expected)
            for actual, expected in zip(payload["metadata"][field], equal_value_with_wrong_type)
        )
    else:
        assert type(payload["metadata"][field]) is not type(equal_value_with_wrong_type)
    payload["metadata"][field] = equal_value_with_wrong_type

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "not-dict"])
def test_restore_rejects_metadata_schema_mismatch_atomically(mutation):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    if mutation == "missing":
        payload["metadata"].pop("mapping_digest")
    elif mutation == "unexpected":
        payload["metadata"]["extra"] = "forbidden"
    else:
        payload["metadata"] = tuple(payload["metadata"].items())

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize("name", STATE_NAMES)
def test_restore_rejects_each_state_shape_mismatch_atomically(name):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    payload[name] = payload[name].reshape(-1)
    if payload[name].shape == getattr(committed_state_bank(), name).shape:
        payload[name] = payload[name].reshape(1, *payload[name].shape)

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize("name", STATE_NAMES)
def test_restore_rejects_each_state_dtype_mismatch_atomically(name):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    wrong_dtype = torch.float64 if payload[name].dtype == torch.float32 else torch.int32
    payload[name] = payload[name].to(wrong_dtype)

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize("name", STATE_NAMES)
def test_restore_rejects_each_nontensor_state_atomically(name):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    payload[name] = payload[name].tolist()

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["slide_centroids"].fill_(float("nan")),
        lambda p: p["slide_counts"].__setitem__(0, -1),
        lambda p: p["slide_tile_presentations"].__setitem__(0, -1),
        lambda p: p["slide_tile_presentations"].__setitem__(1, 0),
        lambda p: p["slide_counts"].__setitem__(0, 3),
        lambda p: (
            p["slide_counts"].__setitem__(1, 0),
            p["slide_tile_presentations"].__setitem__(1, 0),
            p["slide_centroids"].__setitem__(1, torch.tensor([1.0, 1.0])),
        ),
    ],
    ids=[
        "nonfinite-centroid",
        "negative-count",
        "negative-presentations",
        "presentations-below-count",
        "count-beyond-state-step",
        "nonzero-unobserved-centroid",
    ],
)
def test_restore_rejects_each_invalid_authoritative_state_atomically(mutate):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    mutate(payload)

    assert_restore_rejected_atomically(payload)


@pytest.mark.parametrize("expected_step", [1, 3])
def test_restore_rejects_checkpoint_step_disagreement_atomically(expected_step):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    assert_restore_rejected_atomically(payload, expected_step=expected_step)


@pytest.mark.parametrize("expected_step", [True, 2.0, torch.tensor(2)])
def test_restore_requires_an_exact_integer_expected_step_atomically(expected_step):
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    assert_restore_rejected_atomically(payload, expected_step=expected_step)


def test_restore_rejects_negative_matching_state_step_atomically():
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    payload["centroid_state_step"].fill_(-1)
    assert_restore_rejected_atomically(payload, expected_step=-1)


def test_restore_rejects_a_matching_but_unreachable_state_step_atomically():
    payload = committed_state_bank().export_state(HISTORY_METADATA)
    assert int(payload["slide_counts"].sum()) == 4
    payload["centroid_state_step"].fill_(5)

    assert_restore_rejected_atomically(payload, expected_step=5)
