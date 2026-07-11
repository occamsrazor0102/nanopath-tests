from pathlib import Path

import yaml


def test_molcap_config_is_exact_frontier_plus_auxiliary():
    config = yaml.safe_load(Path("configs/molcap-text-s7777.yaml").read_text())

    assert config["train"]["seed"] == 7777
    assert config["data"]["split_seed"] == 7777
    assert config["dino"]["kde_loss_weight"] == 0.05
    assert config["dino"]["lr"] == 0.000125
    assert config["train"]["local_size"] == 112
    assert config["fino"]["discrete"] == [["subtype", 1]]
    assert config["fino"]["continuous"] == [["expr512", 1], ["fga", 1]]
    assert config["molcap"] == {
        "enabled": True,
        "targets": "/data/$USER/nanopath/molcap_text_384.npz",
        "target_dim": 384,
        "weight": 0.03,
        "ramp_start": 0.5,
        "ramp_len": 0.25,
        "diagnose": False,
    }


def test_training_source_wires_optional_molcap_without_probe_changes():
    source = Path("train.py").read_text()

    for token in (
        "seed_neutral_molcap_head", "molcap_loss", "linear_ramp", "gradient_alignment",
        'batch["molcap_target"]', 'batch["molcap_present"]', '"molcap_head"',
        '"molcap_grad_cosine"', '"molcap_grad_norm_ratio"',
        'molcap_cfg.get("diagnose", False)',
    ):
        assert token in source


def test_development_helpers_are_excluded_from_labless_snapshot():
    patterns = Path(".gitignore").read_text().splitlines()
    assert "build_molcap_targets.py" in patterns
    assert "tests/" in patterns
