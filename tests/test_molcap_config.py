import subprocess
from pathlib import Path

import yaml


MISSING = object()


def git_bytes(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"])


def test_locked_probe_and_benchmarking_match_preexperiment_commit():
    revision = "01c1cdf8017a0481636a28ab58a0ddc67d6e0a06"
    paths = ["probe.py"] + subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "--", "benchmarking/"],
        text=True,
    ).splitlines()
    for path in paths:
        # Windows autocrlf may smudge LF blobs to CRLF; no other raw-byte drift is allowed,
        # and the clean-filtered object ID below still locks the submitted content exactly.
        if Path(path).read_bytes() != git_bytes(revision, path):
            eol = subprocess.check_output(
                ["git", "ls-files", "--eol", "--", path], text=True
            ).split()
            assert eol[:2] == ["i/lf", "w/crlf"]
        baseline_oid = subprocess.check_output(
            ["git", "rev-parse", f"{revision}:{path}"], text=True
        ).strip()
        worktree_oid = subprocess.check_output(
            ["git", "hash-object", f"--path={path}", "--", path], text=True
        ).strip()
        assert worktree_oid == baseline_oid


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
    assert route["project"]["name"] == "molcap-probe-route-s7777"
    assert route["project"]["recipe_id"] == "dinov2-vits14-reg-jepa-mask10-molcap-probe-route"
    assert route["project"]["output_dir"] == "/data/$USER/nanopath/molcap/molcap-probe-route-s7777"
    assert route["molcap"]["history"]["enabled"] is False
    assert centroid["project"]["name"] == "molcap-ema-centroid-s7777"
    assert centroid["project"]["recipe_id"] == "dinov2-vits14-reg-jepa-mask10-molcap-ema-centroid"
    assert centroid["project"]["output_dir"] == "/data/$USER/nanopath/molcap/molcap-ema-centroid-s7777"
    assert centroid["molcap"]["history"]["enabled"] is True


def test_route_and_centroid_configs_freeze_registered_contract():
    base = yaml.safe_load(Path("configs/molcap-text-s7777.yaml").read_text())
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
    expected_molcap = {
        **base["molcap"],
        "target_sha256": "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577",
        "diagnose": True,
        "route": "probe_cls_hierarchical",
        "feature_blocks": [4, 6, 8, 11],
        "input_dim": 1536,
        "head_hidden_dim": 512,
        "forward_source": "teacher",
        "gradient_source": "student_identity_ste",
    }
    expected_train_controls = {
        "batch_size": 128,
        "global_views": 2,
        "local_views": 8,
        "global_size": 224,
        "local_size": 112,
    }
    base_preserved_sections = ("data", "model", "train", "dino", "probe", "fino")
    registered_project_leaves = {"name", "recipe_id", "output_dir"}
    for config, history_enabled in ((route, False), (centroid, True)):
        assert set(config) == set(base)
        for section in base_preserved_sections:
            assert config[section] == base[section]
        assert {
            key: value for key, value in config["project"].items()
            if key not in registered_project_leaves
        } == {
            key: value for key, value in base["project"].items()
            if key not in registered_project_leaves
        }
        assert config["molcap"] == {
            **expected_molcap,
            "history": {"enabled": history_enabled, **expected_history},
        }
        assert config["molcap"]["enabled"] is True
        assert config["train"]["seed"] == config["data"]["split_seed"] == 7777
        assert config["train"]["max_train_samples"] == 1_000_000
        assert config["train"]["activation_checkpointing"] is False
        assert {
            key: config["train"][key] for key in expected_train_controls
        } == {
            key: base["train"][key] for key in expected_train_controls
        } == expected_train_controls
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
        assert config["probe"] == base["probe"]
        assert config["fino"] == base["fino"]
        assert config["dino"] == base["dino"]


def test_relative_centroid_config_changes_only_registered_identity_and_gate_leaves():
    path = Path("configs/molcap-ema-relative-s7777.yaml")
    assert path.exists(), "the preregistered relative-centroid config is missing"
    centroid = yaml.safe_load(Path("configs/molcap-ema-centroid-s7777.yaml").read_text())
    relative = yaml.safe_load(path.read_text())

    assert changed_leaves(centroid, relative) == {
        "project.name",
        "project.recipe_id",
        "project.output_dir",
        "molcap.history.gate_version",
        "molcap.history.latest_momentum",
        "molcap.history.permutation_count",
        "molcap.history.permutation_seed_domain",
        "molcap.history.min_trace_ratio",
        "molcap.history.min_effective_rank_ratio",
        "molcap.history.min_participation_ratio",
        "molcap.history.min_alignment",
        "molcap.history.max_permutation_p_value",
    }
    assert relative["project"] == {
        **centroid["project"],
        "name": "molcap-ema-rel-s7777",
        "recipe_id": "dinov2-vits14-reg-jepa-mask10-molcap-ema-relative-centroid",
        "output_dir": "/data/$USER/nanopath/molcap/molcap-ema-rel-s7777",
    }


def test_relative_centroid_config_freezes_matched_latest_gate_contract():
    path = Path("configs/molcap-ema-relative-s7777.yaml")
    assert path.exists(), "the preregistered relative-centroid config is missing"
    centroid = yaml.safe_load(Path("configs/molcap-ema-centroid-s7777.yaml").read_text())
    relative = yaml.safe_load(path.read_text())

    for section in ("data", "model", "train", "dino", "probe", "fino"):
        assert relative[section] == centroid[section]
    assert {
        key: value for key, value in relative["molcap"].items() if key != "history"
    } == {
        key: value for key, value in centroid["molcap"].items() if key != "history"
    }

    relative_history = relative["molcap"]["history"]
    assert relative_history == {
        **centroid["molcap"]["history"],
        "gate_version": "matched_latest_v1",
        "latest_momentum": 0.0,
        "permutation_count": 256,
        "permutation_seed_domain": "molcap-matched-latest-v1",
        "min_trace_ratio": 0.05263157894736842,
        "min_effective_rank_ratio": 0.5,
        "min_participation_ratio": 0.5,
        "min_alignment": 0.0,
        "max_permutation_p_value": 0.01,
    }
    assert relative["train"]["seed"] == relative["data"]["split_seed"] == 7777
    assert relative["train"]["resume"] is None
    assert relative["train"]["batch_size"] == 128
    assert relative["train"]["global_views"] == 2
    assert relative["train"]["local_views"] == 8
    assert relative["train"]["global_size"] == 224
    assert relative["train"]["local_size"] == 112
    assert relative["train"]["max_train_samples"] == 1_000_000
    assert relative["molcap"]["targets"] == "/data/$USER/nanopath/molcap_text_384.npz"
    assert relative["molcap"]["target_sha256"] == (
        "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
    )
    assert relative["molcap"]["feature_blocks"] == [4, 6, 8, 11]
    assert relative["molcap"]["weight"] == 0.03
    assert relative["molcap"]["ramp_start"] == 0.5
    assert relative["molcap"]["ramp_len"] == 0.25
    assert relative_history["enabled"] is True
    assert relative_history["momentum"] == 0.9
    assert relative_history["min_slide_updates"] == 2
    assert relative_history["min_sample_weighted_coverage"] == 0.95
    assert relative_history["min_geometry_patients"] == 512
    assert relative_history["min_centroid_norm"] == 1.0e-6


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


def test_training_source_routes_centroid_without_changing_one_million_schedules():
    source = Path("train.py").read_text()

    for token in (
        'assert int(environment.get("WORLD_SIZE", "1")) == 1',
        '"NANOPATH_RUNNER_STOP_AFTER_SAMPLES"',
        "examples_seen + batch_size <= runner_stop_after_samples",
        "sample_steps_remaining = max(0, runner_stop_after_samples - examples_seen) // batch_size",
        'stop_reason = "runner_stop_after_samples"',
        "sfrac = min(1.0, examples_seen / max_train_samples)",
        "warmup_train_samples = math.ceil(max_train_samples * dino_cfg[\"warmup_fraction\"])",
        "probe_targets = [math.ceil(max_train_samples * (i + 1) / probe_count)",
        'feature_blocks=tuple(molcap_cfg["feature_blocks"])',
        'batch["molcap_slide_idx"]',
        'batch["molcap_patient_idx"]',
        '"molcap_history"',
        '"molcap_centroid_ramp_gate.json"',
        'json.dumps(report, allow_nan=False, indent=2)',
        "centroid_gate_boundary_proposal",
        "if pending_history is not None and molcap_scale == 0.0:",
        "boundary_proposal=centroid_gate_boundary_proposal",
    ):
        assert token in source


def test_training_source_passes_matched_shadow_provenance_before_post_report_discard():
    source = Path("train.py").read_text()
    start = source.index("            if centroid_bank is not None and molcap_scale > 0")
    end = source.index("            # Wrap forward + backward", start)
    boundary = source[start:end]

    for token in (
        "latest_bank=centroid_shadow_bank",
        "target_sha256=train_ds.molcap_target_sha256",
        "mapping_digest=train_ds.molcap_mapping_digest",
        "history_metadata=history_metadata",
        "shadow_metadata=shadow_metadata",
        "boundary_shadow_proposal=centroid_gate_boundary_shadow_proposal",
    ):
        assert token in boundary
    assert boundary.index("run_centroid_ramp_gate(") < boundary.index(
        "discard_latest_observation_shadow("
    )


def test_training_source_keeps_patch_route_and_probe_payload_early_return_explicit():
    source = Path("train.py").read_text()

    assert 'molcap_cfg.get("route") == "probe_cls_hierarchical"' in source
    assert 'sg["x_norm_patchtokens"].mean(1)' in source
    checkpoint_source = source[
        source.index("    def checkpoint_payload") : source.index(
            "    def save_latest_checkpoint"
        )
    ]
    assert "if not full:\n            return payload" in checkpoint_source
    assert checkpoint_source.index("if not full:\n            return payload") < checkpoint_source.index(
        "checkpoint_molcap_state("
    )


def test_development_helpers_are_excluded_from_labless_snapshot():
    patterns = Path(".gitignore").read_text().splitlines()
    assert "build_molcap_targets.py" in patterns
    assert "reembed_molcap_targets.py" in patterns
    assert "tests/" in patterns
    assert ".superpowers/" in patterns
