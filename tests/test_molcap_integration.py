import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from build_molcap_targets import save_target_bank
from dataloader import TCGATileDataset
from model import DinoV2ViT, MolCapHead, gradient_alignment, molcap_loss


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
