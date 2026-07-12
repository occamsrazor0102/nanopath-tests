from pathlib import Path

import yaml


MISSING = object()


def changed_leaves(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return set().union(*(changed_leaves(left.get(key, MISSING), right.get(key, MISSING), f"{prefix}.{key}".strip(".")) for key in left.keys() | right.keys()))
    return {prefix} if left != right else set()


def test_changed_leaves_reports_missing_null_value():
    with_resume = {"train": {"resume": None}}
    without_resume = {"train": {}}

    assert changed_leaves(with_resume, without_resume) == {"train.resume"}
    assert changed_leaves(without_resume, with_resume) == {"train.resume"}


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


def test_biomedical_config_is_encoder_only_ab():
    generic = yaml.safe_load(Path("configs/molcap-text-s7777.yaml").read_text())
    biomedical = yaml.safe_load(Path("configs/molcap-biomed-s7777.yaml").read_text())

    assert changed_leaves(generic, biomedical) == {
        "project.name", "project.output_dir", "molcap.targets", "molcap.target_dim"
    }
    assert biomedical["project"]["name"] == "molcap-biomed-s7777"
    assert biomedical["molcap"]["targets"] == "/data/$USER/nanopath/molcap_biomed_768.npz"
    assert biomedical["molcap"]["target_dim"] == 768


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
    assert "reembed_molcap_targets.py" in patterns
    assert "tests/" in patterns
    assert ".superpowers/" in patterns
