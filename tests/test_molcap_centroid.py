# Focused contracts for hierarchical MolCap aggregation and EMA centroid state.
# These tests keep the state proposal pure and every rejected commit atomic.

import math
import sys
import types

import numpy as np
import pytest
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from train import (
    HierarchicalCentroidBank,
    centroid_audit,
    centroid_geometry,
    crop_major_tile_mean,
    hierarchical_means,
    patient_targets_from_tiles,
    require_centroid_gate,
    teacher_value_student_gradient,
)


def snapshot_buffers(module):
    return {name: value.detach().clone() for name, value in module.named_buffers()}


def assert_buffers_unchanged(module, expected):
    actual = dict(module.named_buffers())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


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
