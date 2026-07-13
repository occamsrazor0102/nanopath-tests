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


def test_route_and_centroid_configs_differ_at_exactly_four_leaves():
    route = yaml.safe_load(Path("configs/molcap-probe-route-s7777.yaml").read_text())
    centroid = yaml.safe_load(Path("configs/molcap-ema-centroid-s7777.yaml").read_text())
    assert changed_leaves(route, centroid) == {
        "project.name",
        "project.recipe_id",
        "project.output_dir",
        "molcap.history.enabled",
    }


def test_route_and_centroid_configs_freeze_registered_contract():
    route = yaml.safe_load(Path("configs/molcap-probe-route-s7777.yaml").read_text())
    centroid = yaml.safe_load(Path("configs/molcap-ema-centroid-s7777.yaml").read_text())
    expected_history = {
        "level": "slide_then_patient",
        "momentum": 0.9,
        "min_slide_updates": 2,
        "min_sample_weighted_coverage": 0.95,
        "min_geometry_patients": 512,
        "min_effective_rank": 32,
        "min_participation_ratio": 16,
        "max_mean_offdiag_cosine": 0.95,
        "min_centroid_norm": 1.0e-6,
    }
    for config in (route, centroid):
        assert config["train"]["seed"] == config["data"]["split_seed"] == 7777
        assert config["train"]["max_train_samples"] == 1_000_000
        assert config["train"]["activation_checkpointing"] is False
        assert config["molcap"]["targets"] == "/data/$USER/nanopath/molcap_text_384.npz"
        assert config["molcap"]["target_sha256"] == "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
        assert config["molcap"]["route"] == "probe_cls_hierarchical"
        assert config["molcap"]["feature_blocks"] == [4, 6, 8, 11]
        assert config["molcap"]["input_dim"] == 1536
        assert config["molcap"]["head_hidden_dim"] == 512
        assert config["molcap"]["forward_source"] == "teacher"
        assert config["molcap"]["gradient_source"] == "student_identity_ste"
        assert config["molcap"]["target_dim"] == 384
        assert config["molcap"]["weight"] == 0.03
        assert config["molcap"]["ramp_start"] == 0.5
        assert config["molcap"]["ramp_len"] == 0.25
        assert config["molcap"]["diagnose"] is True
        assert {k: v for k, v in config["molcap"]["history"].items() if k != "enabled"} == expected_history
        assert config["probe"] == route["probe"]
        assert config["fino"] == route["fino"]
        assert config["dino"] == route["dino"]


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


def test_pca384_config_controls_width_and_head_capacity():
    generic = yaml.safe_load(Path("configs/molcap-text-s7777.yaml").read_text())
    pca384 = yaml.safe_load(Path("configs/molcap-biomed-pca384-s7777.yaml").read_text())
    assert changed_leaves(generic, pca384) == {
        "project.name", "project.output_dir", "molcap.targets"
    }
    assert pca384["project"]["name"] == "molcap-biomed-pca384-s7777"
    assert pca384["molcap"]["targets"] == "/data/$USER/nanopath/molcap_biomed_pca384.npz"
    assert pca384["molcap"]["target_dim"] == generic["molcap"]["target_dim"] == 384
    assert pca384["train"] == generic["train"]
    assert pca384["dino"] == generic["dino"]
    assert pca384["fino"] == generic["fino"]
    assert pca384["probe"] == generic["probe"]


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
