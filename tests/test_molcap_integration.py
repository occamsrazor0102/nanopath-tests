import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


def test_dataset_emits_target_only_when_enabled(tmp_path):
    enabled = TCGATileDataset(tiny_config(tmp_path / "on"), is_train=True)[0]
    disabled = TCGATileDataset(tiny_config(tmp_path / "off", enabled=False), is_train=True)[0]

    assert enabled["molcap_present"].item() == 1.0
    assert enabled["molcap_target"].shape == (6,)
    assert "molcap_target" not in disabled


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
