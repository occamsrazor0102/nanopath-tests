# Focused contracts for hierarchical MolCap aggregation and EMA centroid state.
# These tests keep the state proposal pure and every rejected commit atomic.

import sys
import types

import pytest
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from train import (
    HierarchicalCentroidBank,
    crop_major_tile_mean,
    hierarchical_means,
    patient_targets_from_tiles,
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
