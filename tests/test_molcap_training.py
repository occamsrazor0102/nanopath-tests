import numpy as np
import torch
import torch.nn.functional as F

from build_molcap_targets import save_target_bank
from dataloader import load_molcap_bank
from model import MolCapHead, linear_ramp, molcap_loss, seed_neutral_molcap_head


def test_target_bank_lookup_and_missing_patient(tmp_path):
    path = tmp_path / "targets.npz"
    save_target_bank(
        path,
        ["TCGA-AA-0001", "TCGA-BB-0002"],
        np.eye(2, 4, dtype=np.float32),
        ["first", "second"],
        "structured",
    )

    bank = load_molcap_bank(path, target_dim=4)

    np.testing.assert_array_equal(bank["TCGA-AA-0001"], [1, 0, 0, 0])
    assert bank.get("TCGA-CC-0003") is None


def test_target_dimension_mismatch_fails(tmp_path):
    path = tmp_path / "targets.npz"
    save_target_bank(path, ["TCGA-AA-0001"], np.ones((1, 4), np.float32), ["first"], "structured")

    try:
        load_molcap_bank(path, target_dim=6)
    except AssertionError as exc:
        assert "target_dim" in str(exc)
    else:
        raise AssertionError("dimension mismatch was accepted")


def test_crop_major_alignment_has_zero_loss():
    targets = F.normalize(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), dim=-1)
    features = targets.repeat(2, 1).requires_grad_()

    loss = molcap_loss(torch.nn.Identity(), features, targets, torch.ones(2), views=2)

    assert float(loss.detach()) < 1e-6


def test_patch_route_reaches_head_and_patches_not_unused_cls():
    torch.manual_seed(7)
    patches = torch.randn(4, 6, 8, requires_grad=True)
    unused_cls = torch.randn(4, 8, requires_grad=True)
    head = MolCapHead(8, 5)
    targets = F.normalize(torch.randn(2, 5), dim=-1)

    loss = molcap_loss(head, patches.mean(1), targets, torch.ones(2), views=2)
    loss.backward()

    assert patches.grad is not None and float(patches.grad.norm()) > 0
    assert unused_cls.grad is None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_all_missing_targets_have_zero_loss():
    head = MolCapHead(8, 5)
    loss = molcap_loss(head, torch.randn(4, 8), torch.randn(2, 5), torch.zeros(2), views=2)
    assert float(loss.detach()) == 0.0


def test_linear_ramp_endpoints():
    assert linear_ramp(0.49, 0.50, 0.25) == 0.0
    assert linear_ramp(0.625, 0.50, 0.25) == 0.5
    assert linear_ramp(0.75, 0.50, 0.25) == 1.0
    assert linear_ramp(0.90, 0.50, 0.25) == 1.0


def test_optional_head_does_not_advance_baseline_rng():
    torch.manual_seed(7777)
    expected = torch.rand(4)
    torch.manual_seed(7777)
    seed_neutral_molcap_head(8, 6, "cpu")
    actual = torch.rand(4)
    torch.testing.assert_close(actual, expected)
