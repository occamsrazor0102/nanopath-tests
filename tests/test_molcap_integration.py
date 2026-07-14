import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from build_molcap_targets import save_target_bank
from dataloader import TCGATileDataset
from model import DinoV2ViT, MolCapHead, gradient_alignment, molcap_loss, seed_neutral_molcap_head
from train import (
    HierarchicalCentroidBank,
    build_molcap_history_metadata,
    build_molcap_summary,
    centroid_bank_state_digest,
    checkpoint_molcap_state,
    fold_peak_gpu_memory,
    isolated_torch_rng,
    maybe_arm_labless_autosubmit,
    molcap_head_input_dim,
    molcap_route_enabled,
    molcap_step_diagnostics,
    maybe_paired_routed_molcap,
    paired_routed_molcap,
    restore_molcap_history,
    restore_sample_order_prefix,
    run_centroid_ramp_gate,
    sample_order_prefix_digest,
    training_preflight,
    transactional_optimizer_step,
)


def tiny_config(tmp_path, enabled=True, target_dim=6):
    data = tmp_path / "tiles"; data.mkdir(parents=True)
    image = io.BytesIO(); Image.new("RGB", (224, 224), "pink").save(image, format="JPEG")
    pq.write_table(pa.table({"path": ["TCGA-AA-0001-01Z-00-DX1/tile.jpg"], "jpeg": [image.getvalue()]}), data / "shard-00000.parquet", row_group_size=1)
    bank = tmp_path / "targets.npz"
    save_target_bank(bank, ["TCGA-AA-0001"], np.eye(1, target_dim, dtype=np.float32), ["breast carcinoma"], "structured")
    return {
        "config_path": str(tmp_path / "config.yaml"),
        "data": {
            "dataset_dir": str(data), "split_seed": 7777, "val_fraction": 0.0,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "global_crop_scale": [1.0, 1.0], "local_crop_scale": [1.0, 1.0],
            "color_jitter": 0.0, "color_jitter_saturation": 0.0, "hed_jitter": 0.0, "tissue_thresh": 0.0,
        },
        "train": {"global_views": 2, "local_views": 1, "global_size": 28, "local_size": 28},
        "molcap": {"enabled": enabled, "targets": str(bank), "target_dim": target_dim},
    }


@pytest.fixture
def dense_identity_config(tmp_path):
    data = tmp_path / "tiles"
    data.mkdir(parents=True)
    low_tissue = io.BytesIO()
    Image.new("RGB", (224, 224), "gray").save(low_tissue, format="JPEG")
    tissue = io.BytesIO()
    Image.new("RGB", (224, 224), "pink").save(tissue, format="JPEG")
    paths = [
        "TCGA-BB-0002-02Z-00-DX1/low-tissue.jpg",
        "TCGA-AA-0001-01Z-00-DX1/tissue.jpg",
        "TCGA-BB-0002-01Z-00-DX1/tissue.jpg",
    ]
    pq.write_table(
        pa.table(
            {
                "path": paths,
                "jpeg": [low_tissue.getvalue(), tissue.getvalue(), tissue.getvalue()],
            }
        ),
        data / "shard-00000.parquet",
        row_group_size=1,
    )
    bank = tmp_path / "targets.npz"
    patient_ids = ["TCGA-BB-0002", "TCGA-AA-0001"] + [
        f"TCGA-ZZ-{index:05d}" for index in range(11_426)
    ]
    targets = np.zeros((len(patient_ids), 384), dtype=np.float32)
    targets[:, 0] = 1.0
    save_target_bank(bank, patient_ids, targets, ["caption"] * len(patient_ids), "text")
    target_sha256 = hashlib.sha256(bank.read_bytes()).hexdigest()
    return {
        "config_path": str(tmp_path / "config.yaml"),
        "data": {
            "dataset_dir": str(data),
            "split_seed": 7777,
            "val_fraction": 0.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "global_crop_scale": [1.0, 1.0],
            "local_crop_scale": [1.0, 1.0],
            "color_jitter": 0.0,
            "color_jitter_saturation": 0.0,
            "hed_jitter": 0.0,
            "tissue_thresh": 0.0,
        },
        "train": {"global_views": 2, "local_views": 1, "global_size": 28, "local_size": 28},
        "molcap": {
            "enabled": True,
            "targets": str(bank),
            "target_sha256": target_sha256,
            "target_dim": 384,
            "route": "probe_cls_hierarchical",
        },
    }


def test_dataset_emits_target_only_when_enabled(tmp_path):
    enabled = TCGATileDataset(tiny_config(tmp_path / "on"), is_train=True)[0]
    disabled = TCGATileDataset(tiny_config(tmp_path / "off", enabled=False), is_train=True)[0]

    assert enabled["molcap_present"].item() == 1.0
    assert enabled["molcap_target"].shape == (6,)
    assert "molcap_target" not in disabled


def test_dataset_emits_deterministic_dense_centroid_indices(dense_identity_config, monkeypatch):
    dataset = TCGATileDataset(dense_identity_config, is_train=True)

    assert dataset.molcap_patient_ids == ("TCGA-BB-0002", "TCGA-AA-0001")
    assert dataset.molcap_slide_ids == (
        "TCGA-AA-0001-01Z-00-DX1",
        "TCGA-BB-0002-01Z-00-DX1",
        "TCGA-BB-0002-02Z-00-DX1",
    )
    np.testing.assert_array_equal(dataset.molcap_slide_to_patient, [1, 0, 0])
    assert len(dataset.molcap_mapping_digest) == 64
    int(dataset.molcap_mapping_digest, 16)
    assert dataset.molcap_target_sha256 == dense_identity_config["molcap"]["target_sha256"]

    items = [dataset[index] for index in range(len(dataset))]
    assert [
        (item["molcap_slide_idx"].item(), item["molcap_patient_idx"].item())
        for item in items
    ] == [(2, 0), (0, 1), (1, 0)]
    assert all(item["molcap_slide_idx"].dtype == torch.int64 for item in items)
    assert all(item["molcap_patient_idx"].dtype == torch.int64 for item in items)

    dataset.tissue_thresh = 0.5
    monkeypatch.setattr("dataloader.random.randint", lambda low, high: 1)
    resampled = dataset[0]
    assert resampled["sample_idx"].item() == 1
    assert resampled["molcap_slide_idx"].item() == 0
    assert resampled["molcap_patient_idx"].item() == 1


def test_routed_dataset_rejects_corrupt_target_sha(dense_identity_config):
    dense_identity_config["molcap"]["target_sha256"] = "0" * 64

    with pytest.raises(AssertionError, match="target_sha256"):
        TCGATileDataset(dense_identity_config, is_train=True)


def test_routed_dataset_rejects_noncanonical_target_width(dense_identity_config):
    target_path = Path(dense_identity_config["molcap"]["targets"])
    with np.load(target_path, allow_pickle=False) as artifact:
        patient_ids = artifact["patient_ids"]
        captions = artifact["captions"]
    narrow_targets = np.zeros((len(patient_ids), 2), dtype=np.float32)
    narrow_targets[:, 0] = 1.0
    save_target_bank(target_path, patient_ids, narrow_targets, captions, "text")
    dense_identity_config["molcap"]["target_dim"] = 2
    dense_identity_config["molcap"]["target_sha256"] = hashlib.sha256(target_path.read_bytes()).hexdigest()

    with pytest.raises(AssertionError, match="384"):
        TCGATileDataset(dense_identity_config, is_train=True)


def test_shared_probe_readout_matches_probe_and_independent_oracle():
    torch.manual_seed(13)
    model = DinoV2ViT(variant_cfg=(8, 12, 2, 2, "mlp", True, "unused", 0)).eval()
    x = torch.randn(3, 3, 28, 28)
    with torch.no_grad():
        xt, expected = model._prepare_tokens(x), []
        for i, block in enumerate(model.blocks):
            xt = block(xt)
            if i in (4, 6, 8, 11):
                expected.append(model.norm(xt)[:, 0])
        expected = torch.cat(expected, dim=-1)
        default = model(x)
        routed = model(x, feature_blocks=(4, 6, 8, 11))
    assert set(default) == {"x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens"}
    assert routed["x_norm_probe_features"].shape == (3, 32)
    torch.testing.assert_close(routed["x_norm_probe_features"], expected, atol=2e-5, rtol=0)
    torch.testing.assert_close(model.probe_features(x), expected, atol=2e-5, rtol=0)


def test_tiny_patch_route_checkpoint_and_gradient_diagnostics(tmp_path):
    torch.manual_seed(11)
    model = DinoV2ViT(variant_cfg=(8, 1, 2, 2, "mlp", True, "unused", 0))
    head = MolCapHead(8, 6)
    out = model(torch.randn(2, 3, 28, 28))
    target = torch.nn.functional.normalize(torch.randn(2, 6), dim=-1)
    base = out["x_norm_clstoken"].square().mean()
    aux = molcap_loss(head, out["x_norm_patchtokens"].mean(1), target, torch.ones(2), views=1)
    cosine, ratio = gradient_alignment(base, aux, model.blocks[-1].attn.qkv.weight)

    assert torch.isfinite(base + aux + cosine + ratio)
    assert ratio > 0
    (base + aux).backward()
    assert model.blocks[-1].attn.qkv.weight.grad is not None

    path = tmp_path / "step.pt"; torch.save({"molcap_head": head.state_dict()}, path)
    restored = MolCapHead(8, 6)
    restored.load_state_dict(torch.load(path, weights_only=True)["molcap_head"])
    for first, second in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)


def test_768d_training_sample_patch_route_and_checkpoint(tmp_path):
    torch.manual_seed(17)
    sample = TCGATileDataset(tiny_config(tmp_path, target_dim=768), is_train=True)[0]
    model = DinoV2ViT(variant_cfg=(8, 1, 2, 2, "mlp", True, "unused", 0))
    head = MolCapHead(8, 768)

    out = model(sample["global_views"])
    loss = molcap_loss(
        head,
        out["x_norm_patchtokens"].mean(1),
        sample["molcap_target"].unsqueeze(0),
        sample["molcap_present"].unsqueeze(0),
        views=2,
    )
    loss.backward()

    assert sample["molcap_target"].shape == (768,)
    gradients = [model.blocks[-1].attn.qkv.weight.grad, *(parameter.grad for parameter in head.parameters())]
    assert all(gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in gradients)

    path = tmp_path / "biomedical-step.pt"
    torch.save({"molcap_head": head.state_dict()}, path)
    restored = MolCapHead(8, 768)
    restored.load_state_dict(torch.load(path, weights_only=True)["molcap_head"])
    for first, second in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)


def test_pca384_bank_forward_backward_and_checkpoint(tmp_path):
    cfg = tiny_config(tmp_path, target_dim=384)
    save_target_bank(
        Path(cfg["molcap"]["targets"]),
        ["TCGA-AA-0001"],
        np.eye(1, 384, dtype=np.float32),
        ["caption"],
        "biomedical-pca384",
    )
    sample = TCGATileDataset(cfg, is_train=True)[0]
    head = MolCapHead(8, 384)
    features = torch.randn(2, 8, requires_grad=True)
    loss = 1 - (head(features) * sample["molcap_target"]).sum(-1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None and features.grad.norm() > 0
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in head.parameters())
    checkpoint = {"molcap_head": head.state_dict()}
    restored = MolCapHead(8, 384)
    restored.load_state_dict(checkpoint["molcap_head"])


def routed_molcap_config(history_enabled=False):
    return {
        "enabled": True,
        "target_sha256": "a" * 64,
        "target_dim": 6,
        "weight": 0.03,
        "ramp_start": 0.5,
        "ramp_len": 0.25,
        "route": "probe_cls_hierarchical",
        "feature_blocks": [4, 6, 8, 11],
        "input_dim": 32,
        "head_hidden_dim": 512,
        "forward_source": "teacher",
        "gradient_source": "student_identity_ste",
        "history": {
            "enabled": history_enabled,
            "level": "slide_then_patient",
            "momentum": 0.9,
            "min_slide_updates": 2,
            "min_sample_weighted_coverage": 0.95,
            "min_geometry_patients": 2,
            "min_effective_rank": 1.0,
            "min_participation_ratio": 1.0,
            "max_mean_offdiag_cosine": 0.95,
            "min_centroid_norm": 1.0e-6,
        },
    }


def centroid_metadata(history_enabled=True):
    cfg = routed_molcap_config(history_enabled=history_enabled)
    dataset = SimpleNamespace(molcap_target_sha256="a" * 64, molcap_mapping_digest="b" * 64)
    return build_molcap_history_metadata(cfg, dataset)


def test_probe_route_head_initialization_is_seed_neutral_at_1536_input():
    torch.manual_seed(7777)
    expected = torch.rand(4)
    torch.manual_seed(7777)
    seed_neutral_molcap_head(1536, 384, "cpu")
    torch.testing.assert_close(torch.rand(4), expected)


def test_route_predicate_and_head_width_preserve_legacy_patch_molcap():
    legacy = {"enabled": True, "target_dim": 6, "weight": 0.03}
    routed = routed_molcap_config()

    assert molcap_route_enabled(legacy) is False
    assert molcap_head_input_dim(legacy, embed_dim=8) == 8
    assert molcap_route_enabled(routed) is True
    assert molcap_head_input_dim(routed, embed_dim=8) == 32

    routed["input_dim"] = 31
    with pytest.raises(AssertionError):
        molcap_head_input_dim(routed, embed_dim=8)
    routed["input_dim"] = 32
    routed["head_hidden_dim"] = 256
    with pytest.raises(AssertionError):
        molcap_head_input_dim(routed, embed_dim=8)


def test_auxiliary_forward_restores_cpu_rng_on_success_and_exception():
    torch.manual_seed(19)
    model = DinoV2ViT(
        variant_cfg=(8, 12, 2, 2, "mlp", True, "unused", 0),
        drop_path_rate=0.5,
    ).train()
    x = torch.randn(2, 3, 28, 28)
    state = torch.random.get_rng_state()
    with isolated_torch_rng(123, torch.device("cpu")):
        model(x, feature_blocks=(4, 6, 8, 11))
    actual = torch.rand(3)
    torch.random.set_rng_state(state)
    expected = torch.rand(3)
    torch.testing.assert_close(actual, expected)

    state = torch.random.get_rng_state()
    with pytest.raises(RuntimeError, match="auxiliary failure"):
        with isolated_torch_rng(456, torch.device("cpu")):
            torch.rand(5)
            raise RuntimeError("auxiliary failure")
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_auxiliary_forward_restores_all_cuda_rng_states_on_success_and_exception():
    device = torch.device("cuda")
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    model = DinoV2ViT(
        variant_cfg=(8, 12, 2, 2, "mlp", True, "unused", 0),
        drop_path_rate=0.5,
    ).to(device).train()
    x = torch.randn(2, 3, 28, 28, device=device)
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    with isolated_torch_rng(123, device):
        model(x, feature_blocks=(4, 6, 8, 11))
    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_states)
    )

    with pytest.raises(RuntimeError, match="auxiliary failure"):
        with isolated_torch_rng(456, device):
            torch.rand(5, device=device)
            raise RuntimeError("auxiliary failure")
    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_states)
    )


def test_nonzero_paired_route_mechanics_reach_student_and_head_without_teacher_or_bank_mutation():
    torch.manual_seed(41)
    student = DinoV2ViT(
        variant_cfg=(8, 12, 2, 2, "mlp", True, "unused", 0)
    ).train()
    teacher = deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    head = MolCapHead(32, 6)
    global_crops = torch.randn(4, 3, 28, 28)
    local_crops = torch.randn(2, 3, 28, 28)
    masks = torch.zeros(4, 4, dtype=torch.bool)

    with torch.no_grad():
        teacher_out = teacher(global_crops, feature_blocks=(4, 6, 8, 11))
    student(global_crops, masks=masks)
    student(local_crops)
    with isolated_torch_rng(7777 + 1_000_003, torch.device("cpu")):
        student_out = student(global_crops, feature_blocks=(4, 6, 8, 11))

    slide_ids = torch.tensor([0, 1], dtype=torch.int64)
    patient_ids = torch.tensor([0, 1], dtype=torch.int64)
    slide_to_patient = torch.tensor([0, 1], dtype=torch.int64)
    targets = torch.nn.functional.normalize(torch.randn(2, 6), dim=-1)
    present = torch.ones(2)
    bank = HierarchicalCentroidBank(slide_to_patient, feature_dim=32, momentum=0.9)
    before = {name: value.clone() for name, value in bank.named_buffers()}

    route = paired_routed_molcap(
        head,
        student_out["x_norm_probe_features"],
        teacher_out["x_norm_probe_features"],
        slide_ids,
        patient_ids,
        slide_to_patient,
        targets,
        present,
        views=2,
        weight=0.03,
        scale=1.0,
        centroid_bank=None,
    )
    centroid = paired_routed_molcap(
        head,
        student_out["x_norm_probe_features"],
        teacher_out["x_norm_probe_features"],
        slide_ids,
        patient_ids,
        slide_to_patient,
        targets,
        present,
        views=2,
        weight=0.03,
        scale=1.0,
        centroid_bank=bank,
    )

    torch.testing.assert_close(route.patient_features, centroid.patient_features)
    torch.testing.assert_close(route.loss, centroid.loss)
    assert centroid.pending_history is not None
    assert not centroid.pending_history.patient_centroids.requires_grad
    assert not centroid.pending_history.next_slide_centroids.requires_grad
    assert torch.isfinite(centroid.loss) and centroid.loss != 0
    assert torch.equal(centroid.student_hierarchy.patient_ids, patient_ids)
    assert torch.equal(centroid.teacher_hierarchy.patient_ids, patient_ids)
    for name, value in bank.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    centroid.loss.backward()
    gradients = [
        student.blocks[-1].attn.qkv.weight.grad,
        *(parameter.grad for parameter in head.parameters()),
    ]
    assert all(
        gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
        for gradient in gradients
    )
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_validation_style_routed_call_skips_auxiliary_forward_and_preserves_bank():
    class FailIfForwarded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("validation must not run the auxiliary student forward")

    student = FailIfForwarded()
    mapping = torch.tensor([0], dtype=torch.int64)
    bank = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.9)
    before = {name: value.clone() for name, value in bank.named_buffers()}

    result = maybe_paired_routed_molcap(
        student,
        torch.empty(0),
        None,
        MolCapHead(2, 3),
        None,
        None,
        None,
        None,
        mapping,
        feature_blocks=(4, 6, 8, 11),
        seed=7777,
        device=torch.device("cpu"),
        views=2,
        weight=0.03,
        scale=0.0,
        centroid_bank=bank,
    )

    assert result is None
    assert student.calls == 0
    for name, value in bank.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def preflight_config(probe_enabled=False):
    return {
        "train": {
            "max_train_samples": 1_000_000,
            "max_train_flops": 1_000_000_000_000_000_000,
            "batch_size": 128,
        },
        "probe": {
            "enabled": probe_enabled,
            "count": 1 if probe_enabled else 0,
            "datasets": ["dummy"] if probe_enabled else [],
            "segmentation_datasets": [],
            "slide_datasets": [],
            "auc_datasets": [],
            "survival_datasets": [],
            "robustness_datasets": [],
        },
    }


def test_training_preflight_enforces_single_gpu_and_non_scored_short_runner_cap():
    assert training_preflight(preflight_config(), {}) == (1_000_000, False)
    assert training_preflight(
        preflight_config(), {"WORLD_SIZE": "1", "NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "32768"}
    ) == (32_768, True)
    assert training_preflight(
        preflight_config(), {"NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "1000"}
    ) == (1_000, True)
    assert training_preflight(
        preflight_config(probe_enabled=True),
        {"NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "1000000"},
    ) == (1_000_000, False)
    assert training_preflight(
        preflight_config(probe_enabled=True),
        {
            "NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "1000000",
            "LABLESS_AUTOSUBMIT_FILE": "armed.json",
        },
    ) == (1_000_000, False)

    invalid_environments = [
        {"WORLD_SIZE": "2"},
        {"NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "0"},
        {"NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "1000001"},
        {
            "NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "32768",
            "LABLESS_AUTOSUBMIT_FILE": "armed.json",
        },
    ]
    for environment in invalid_environments:
        with pytest.raises(AssertionError):
            training_preflight(preflight_config(), environment)

    with pytest.raises(AssertionError):
        training_preflight(
            preflight_config(probe_enabled=True),
            {"NANOPATH_RUNNER_STOP_AFTER_SAMPLES": "32768"},
        )


def test_full_budget_runner_environment_does_not_disable_labless_autosubmit(monkeypatch):
    monkeypatch.setenv("NANOPATH_RUNNER_STOP_AFTER_SAMPLES", "1000000")
    monkeypatch.setenv("LABLESS_AUTOSUBMIT_FILE", "armed.json")

    assert (
        maybe_arm_labless_autosubmit(
            preflight_config(probe_enabled=True), Path(__file__).resolve().parents[1]
        )
        == "armed.json"
    )


def test_peak_gpu_memory_fold_keeps_prior_peak_and_includes_final_interval():
    gib = 1024**3
    assert fold_peak_gpu_memory(2.0, int(1.5 * gib)) == 2.0
    assert fold_peak_gpu_memory(2.0, int(2.5 * gib)) == 2.5


def test_history_metadata_is_exact_and_uses_dataset_provenance():
    metadata = centroid_metadata()
    assert metadata == {
        "version": 1,
        "arm": "centroid",
        "target_sha256": "a" * 64,
        "mapping_digest": "b" * 64,
        "feature_blocks": (4, 6, 8, 11),
        "feature_width": 32,
        "momentum": 0.9,
        "hierarchy": "slide_then_patient",
        "ste": "student_identity_ste",
        "weight": 0.03,
        "ramp_start": 0.5,
        "ramp_len": 0.25,
    }


def test_checkpoint_probe_returns_before_head_history_export_or_bank_mutation():
    head = MolCapHead(32, 6)
    bank = HierarchicalCentroidBank(torch.tensor([0, 1]), feature_dim=32, momentum=0.9)
    with torch.no_grad():
        bank.patient_sums.fill_(3.0)
    before = {name: value.clone() for name, value in bank.named_buffers()}
    payload = checkpoint_molcap_state(
        {"model": {}},
        full=False,
        checkpoint_step=0,
        molcap_head=head,
        centroid_bank=bank,
        history_metadata=centroid_metadata(),
        sample_order_prefix=[1, 2],
        sample_order_available=True,
    )
    assert set(payload) == {"model"}
    assert "molcap_head" not in payload and "molcap_history" not in payload
    for name, value in bank.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_full_checkpoint_history_is_step_exact_and_resume_rejects_routed_arm_mismatches():
    head = MolCapHead(32, 6)
    bank = HierarchicalCentroidBank(torch.tensor([0, 1]), feature_dim=32, momentum=0.9)
    metadata = centroid_metadata()
    payload = checkpoint_molcap_state(
        {"step": 0},
        full=True,
        checkpoint_step=0,
        molcap_head=head,
        centroid_bank=bank,
        history_metadata=metadata,
        sample_order_prefix=[1, 2],
        sample_order_available=True,
    )
    assert set(payload) >= {
        "molcap_head",
        "molcap_history",
        "molcap_sample_order_prefix",
        "molcap_sample_order_available",
    }
    assert int(payload["molcap_history"]["centroid_state_step"]) == payload["step"] == 0

    restored = HierarchicalCentroidBank(torch.tensor([0, 1]), feature_dim=32, momentum=0.9)
    restore_molcap_history(payload, routed=True, centroid_bank=restored, history_metadata=metadata, checkpoint_step=0)
    with pytest.raises(AssertionError):
        restore_molcap_history({"step": 0}, routed=True, centroid_bank=restored, history_metadata=metadata, checkpoint_step=0)
    with pytest.raises(AssertionError):
        restore_molcap_history(payload, routed=True, centroid_bank=None, history_metadata=None, checkpoint_step=0)
    with pytest.raises(AssertionError):
        restore_molcap_history(payload, routed=True, centroid_bank=restored, history_metadata=metadata, checkpoint_step=1)

    # Legacy patch-route checkpoints are outside the routed arm/history contract.
    restore_molcap_history(payload, routed=False, centroid_bank=None, history_metadata=None, checkpoint_step=0)
    with pytest.raises(AssertionError):
        checkpoint_molcap_state(
            {"step": 1},
            full=True,
            checkpoint_step=1,
            molcap_head=head,
            centroid_bank=bank,
            history_metadata=metadata,
            sample_order_prefix=[],
            sample_order_available=True,
        )


def populated_gate_bank():
    bank = HierarchicalCentroidBank(torch.tensor([0, 1]), feature_dim=2, momentum=0.9)
    with torch.no_grad():
        bank.slide_centroids.copy_(torch.eye(2))
        bank.slide_counts.fill_(2)
        bank.slide_tile_presentations.fill_(2)
        bank.centroid_state_step.fill_(2)
        bank.patient_sums.copy_(torch.eye(2))
        bank.patient_slide_counts.fill_(1)
    return bank


def test_centroid_ramp_gate_persists_strict_pass_and_failure_evidence(tmp_path):
    cfg = routed_molcap_config(history_enabled=True)["history"]
    passed_path = tmp_path / "passed.json"
    report = run_centroid_ramp_gate(populated_gate_bank(), cfg, passed_path)
    assert report["passed"] is True
    assert json.loads(passed_path.read_text())["passed"] is True
    json.dumps(report, allow_nan=False)

    failed_path = tmp_path / "failed.json"
    failing = dict(cfg, min_effective_rank=3.0)
    with pytest.raises(AssertionError):
        run_centroid_ramp_gate(populated_gate_bank(), failing, failed_path)
    failed = json.loads(failed_path.read_text())
    assert failed["passed"] is False
    assert failed["failure"]
    json.dumps(failed, allow_nan=False)

    nonfinite_path = tmp_path / "nonfinite.json"
    bank = populated_gate_bank()
    with torch.no_grad():
        bank.slide_centroids[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        run_centroid_ramp_gate(bank, cfg, nonfinite_path)
    nonfinite = json.loads(nonfinite_path.read_text())
    assert nonfinite["passed"] is False
    json.dumps(nonfinite, allow_nan=False)


def one_slide_proposal():
    bank = HierarchicalCentroidBank(torch.tensor([0]), feature_dim=1, momentum=0.9)
    from train import hierarchical_means

    hierarchy = hierarchical_means(
        torch.tensor([[2.0]]), torch.tensor([0]), torch.tensor([0])
    )
    return bank, bank.propose(hierarchy)


def test_transactional_optimizer_commits_only_after_finite_successful_step():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    bank, proposal = one_slide_proposal()
    grad_norm = transactional_optimizer_step(
        parameter.square(),
        optimizer,
        [parameter],
        clip_grad=1.0,
        centroid_bank=bank,
        pending_history=proposal,
        completed_step=1,
    )
    assert torch.isfinite(grad_norm)
    assert int(bank.centroid_state_step) == 1


@pytest.mark.parametrize("failure", ["loss", "gradient", "optimizer"])
def test_transactional_optimizer_never_commits_on_nonfinite_or_step_exception(failure):
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class RaisingSGD(torch.optim.SGD):
        def step(self, closure=None):
            raise RuntimeError("optimizer failure")

    optimizer = RaisingSGD([parameter], lr=0.1) if failure == "optimizer" else torch.optim.SGD([parameter], lr=0.1)
    bank, proposal = one_slide_proposal()
    if failure == "loss":
        loss = parameter * torch.tensor(float("nan"))
    else:
        loss = parameter.square()
        if failure == "gradient":
            parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    expected = RuntimeError if failure == "optimizer" else AssertionError
    with pytest.raises(expected):
        transactional_optimizer_step(
            loss,
            optimizer,
            [parameter],
            clip_grad=1.0,
            centroid_bank=bank,
            pending_history=proposal,
            completed_step=1,
        )
    assert int(bank.centroid_state_step) == 0
    assert bank.slide_counts.sum() == 0


def test_sample_order_prefix_digest_is_first_8192_signed_little_endian_int64_values():
    values = list(range(-10, 9000))
    digest, count = sample_order_prefix_digest(values)
    expected_bytes = np.asarray(values[:8192], dtype="<i8").tobytes()
    assert digest == hashlib.sha256(expected_bytes).hexdigest()
    assert count == 8192


def test_sample_order_prefix_is_preserved_or_explicitly_unavailable_on_resume():
    prefix = torch.arange(8192, dtype=torch.int64)
    checkpoint = {
        "molcap_sample_order_available": True,
        "molcap_sample_order_prefix": prefix,
    }
    restored, available = restore_sample_order_prefix(checkpoint, routed=True)
    assert available is True
    assert restored == prefix.tolist()
    assert sample_order_prefix_digest(restored) == sample_order_prefix_digest(prefix.tolist())

    partial = dict(checkpoint, molcap_sample_order_prefix=prefix[:128])
    restored, available = restore_sample_order_prefix(partial, routed=True)
    assert restored == prefix[:128].tolist()
    assert available is False
    assert restore_sample_order_prefix({}, routed=True) == ([], False)
    assert restore_sample_order_prefix(checkpoint, routed=False) == ([], False)
    assert restore_sample_order_prefix(None, routed=True) == ([], True)


def test_molcap_diagnostics_are_rng_neutral_nonmutating_and_handle_no_mature_population():
    torch.manual_seed(53)
    head = MolCapHead(2, 3)
    student = torch.tensor([[1.0, 0.0], [3.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    slides = torch.tensor([0], dtype=torch.int64)
    patients = torch.tensor([0], dtype=torch.int64)
    mapping = torch.tensor([0], dtype=torch.int64)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    present = torch.ones(1)
    bank = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.9)
    result = paired_routed_molcap(
        head,
        student,
        teacher,
        slides,
        patients,
        mapping,
        targets,
        present,
        views=2,
        weight=0.03,
        scale=0.0,
        centroid_bank=bank,
    )
    bank.commit(result.pending_history, step=1)
    before = {name: value.clone() for name, value in bank.named_buffers()}
    rng_before = torch.random.get_rng_state()

    diagnostics = molcap_step_diagnostics(
        result,
        head,
        centroid_bank=bank,
        min_slide_updates=2,
        gate_report=None,
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, value in bank.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert diagnostics["molcap_unique_patients"] == 1
    assert diagnostics["molcap_current_slides"] == 1
    assert diagnostics["molcap_history_state_step"] == 1
    assert diagnostics["molcap_observed_slides"] == 1
    assert diagnostics["molcap_mature_slides"] == 0
    assert diagnostics["molcap_mature_patients"] == 0
    assert diagnostics["molcap_sample_weighted_mature_coverage"] == 0.0
    assert diagnostics["molcap_gate_geometry"] is None
    assert diagnostics["molcap_current_all_observed_geometry"]["patient_count"] == 1
    assert diagnostics["molcap_current_all_observed_geometry"]["effective_rank"] is None
    assert diagnostics["molcap_current_mature_geometry"]["patient_count"] == 0
    assert diagnostics["molcap_nonhistorical_tile_fraction"] == 1.0
    assert diagnostics["molcap_observed_slides_per_patient_mean"] == 1.0
    assert diagnostics["molcap_observed_slides_per_patient_q50"] == 1.0
    assert diagnostics["molcap_teacher_drift_mean"] == 1.0
    assert diagnostics["molcap_teacher_drift_q50"] is None
    assert diagnostics["molcap_feature_bank_bytes"] == 2 * 4
    assert diagnostics["molcap_bank_bytes"] == sum(
        buffer.numel() * buffer.element_size() for buffer in bank.buffers()
    )
    assert diagnostics["molcap_bank_state_digest"] == centroid_bank_state_digest(bank)
    assert len(diagnostics["molcap_bank_state_digest"]) == 64
    json.dumps(diagnostics, allow_nan=False)


def test_centroid_bank_state_digest_is_deterministic_and_changes_with_authoritative_state():
    first = populated_gate_bank()
    second = populated_gate_bank()
    initial = centroid_bank_state_digest(first)
    assert initial == centroid_bank_state_digest(first) == centroid_bank_state_digest(second)

    with torch.no_grad():
        second.slide_centroids[0, 0] += 0.125
    assert centroid_bank_state_digest(second) != initial


def test_molcap_summary_contains_pairing_provenance_digest_and_latest_bank_state():
    head = MolCapHead(2, 3)
    slides = torch.tensor([0], dtype=torch.int64)
    patients = torch.tensor([0], dtype=torch.int64)
    mapping = torch.tensor([0], dtype=torch.int64)
    bank = HierarchicalCentroidBank(mapping, feature_dim=2, momentum=0.9)
    result = paired_routed_molcap(
        head,
        torch.tensor([[1.0, 0.0], [3.0, 0.0]], requires_grad=True),
        torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
        slides,
        patients,
        mapping,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.ones(1),
        views=2,
        weight=0.03,
        scale=0.0,
        centroid_bank=bank,
    )
    bank.commit(result.pending_history, step=1)
    dataset = SimpleNamespace(
        molcap_mapping_digest="b" * 64,
        molcap_target_sha256="a" * 64,
        molcap_patient_ids=("P0",),
        molcap_slide_ids=("S0",),
    )
    summary = build_molcap_summary(
        routed_result=result,
        molcap_head=head,
        centroid_bank=bank,
        molcap_cfg=routed_molcap_config(history_enabled=True),
        train_ds=dataset,
        config_sha256="c" * 64,
        git_commit="d" * 40,
        sample_order_prefix=[7, 9],
        sample_order_available=True,
        centroid_gate_report=None,
        centroid_gate_passed=False,
        molcap_grad_cosine=0.25,
        molcap_grad_norm_ratio=1.5,
    )
    expected_digest = hashlib.sha256(np.asarray([7, 9], dtype="<i8").tobytes()).hexdigest()
    assert summary["molcap_mapping_digest"] == "b" * 64
    assert summary["molcap_target_sha256"] == "a" * 64
    assert summary["molcap_config_sha256"] == "c" * 64
    assert summary["molcap_source_commit"] == "d" * 40
    assert summary["molcap_sample_order_digest"] == expected_digest
    assert summary["molcap_sample_order_count"] == 2
    assert summary["molcap_history_state_step"] == 1
    assert summary["molcap_bank_state_digest"] == centroid_bank_state_digest(bank)
    assert summary["molcap_grad_cosine"] == 0.25
    assert summary["molcap_grad_norm_ratio"] == 1.5
    json.dumps(summary, allow_nan=False)
