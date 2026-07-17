from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from labless.submit_to_labless import public_config_path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "labless" / "audit_source_policy.py"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def materialize_snapshot(destination: Path) -> tuple[Path, str]:
    commit = git("rev-parse", "HEAD")
    archive = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", commit],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        members = [member for member in contents.getmembers() if not member.issym() and not member.islnk()]
        contents.extractall(destination, members=members, filter="data")
    return destination, commit


def run_audit(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    assert AUDIT.exists(), "the source-policy audit CLI has not been implemented"
    return subprocess.run(
        [sys.executable, str(AUDIT), *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def test_prelaunch_snapshot_rejects_locked_change_and_external_helper(tmp_path: Path) -> None:
    source_dir, commit = materialize_snapshot(tmp_path / "source")
    (source_dir / "probe.py").write_text((source_dir / "probe.py").read_text() + "\n# test-only locked change\n")
    (source_dir / "external_helper.py").write_text("def helper():\n    return 1\n")

    result = run_audit(
        "--source-dir",
        str(source_dir),
        "--source-commit",
        commit,
        "--review-config",
        "configs/main.yaml",
        "--main-ref",
        "HEAD",
    )

    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["locked_path_changes"] == ["probe.py"]
    assert "helper file outside allowed surface changed: external_helper.py" in payload["policy_errors"]


def test_prelaunch_snapshot_allows_reviewed_train_change_with_locked_config(tmp_path: Path) -> None:
    source_dir, commit = materialize_snapshot(tmp_path / "source")
    train_path = source_dir / "train.py"
    train_path.write_bytes(train_path.read_bytes() + b"\n# test-only reviewed change\n")

    result = run_audit(
        "--source-dir",
        str(source_dir),
        "--source-commit",
        commit,
        "--review-config",
        "configs/main.yaml",
        "--main-ref",
        "HEAD",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["eligible"] is True
    assert payload["changed_files"] == ["train.py"]
    assert payload["locked_path_changes"] == []
    assert payload["policy_errors"] == []


def test_completed_output_derives_metadata_without_mutating_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "completed"
    output_dir.mkdir()
    source_dir, commit = materialize_snapshot(output_dir / "labless_source")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config_path": "configs/main.yaml",
                "wandb": {
                    "id": "test-run",
                    "source_artifact": "nanopath-source-test-run",
                    "git": {"commit": commit, "remote": "https://example.invalid/nanopath.git"},
                },
            }
        )
    )
    before = {path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}

    result = run_audit("--output-dir", str(output_dir), "--main-ref", "HEAD")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "output_dir"
    assert payload["commit"] == commit
    assert payload["locked_path_changes"] == []
    assert payload["policy_errors"] == []
    assert before == {path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}

    report = tmp_path / "audit.json"
    reported = run_audit("--output-dir", str(output_dir), "--main-ref", "HEAD", "--report", str(report))

    assert reported.returncode == 0, reported.stderr
    assert json.loads(report.read_text()) == json.loads(reported.stdout)
    assert before == {path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}


def test_completed_output_requires_a_recorded_config_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "completed"
    output_dir.mkdir()
    _, commit = materialize_snapshot(output_dir / "labless_source")
    (output_dir / "summary.json").write_text(json.dumps({"wandb": {"git": {"commit": commit}}}))

    result = run_audit("--output-dir", str(output_dir), "--main-ref", "HEAD")

    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "eligible": False,
        "error": "summary.config_path is required for --output-dir",
    }


def test_public_config_path_normalizes_windows_absolute_paths() -> None:
    assert public_config_path(r"C:\runs\completed\configs\main.yaml") == "configs/main.yaml"


@pytest.mark.parametrize("value", [r"C:\runs\completed\configs\main.txt", r"C:\runs\completed\config\main.yaml"])
def test_public_config_path_rejects_invalid_windows_forms(value: str) -> None:
    with pytest.raises(ValueError, match="summary.config_path"):
        public_config_path(value)


def test_completed_output_accepts_windows_absolute_recorded_config_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "completed"
    output_dir.mkdir()
    _, commit = materialize_snapshot(output_dir / "labless_source")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config_path": r"C:\runs\completed\configs\main.yaml",
                "wandb": {"git": {"commit": commit}},
            }
        )
    )

    result = run_audit("--output-dir", str(output_dir), "--main-ref", "HEAD")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["review_config"] == "configs/main.yaml"


def test_completed_output_uses_recorded_config_when_an_override_is_supplied(tmp_path: Path) -> None:
    output_dir = tmp_path / "completed"
    output_dir.mkdir()
    source_dir, commit = materialize_snapshot(output_dir / "labless_source")
    main_config = (source_dir / "configs" / "main.yaml").read_bytes()
    (source_dir / "configs" / "alternate.yaml").write_bytes(main_config)
    (source_dir / "configs" / "main.yaml").write_bytes(main_config.replace(b"  count: 1", b"  count: 2", 1))
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config_path": "configs/main.yaml",
                "wandb": {"git": {"commit": commit}},
            }
        )
    )

    result = run_audit(
        "--output-dir",
        str(output_dir),
        "--review-config",
        "configs/alternate.yaml",
        "--main-ref",
        "HEAD",
    )

    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["review_config"] == "configs/main.yaml"
    assert "locked probe config changed: configs/main.yaml" in payload["policy_errors"]


@pytest.mark.parametrize("target", ["summary.json", "labless_source/configs/main.yaml"])
def test_existing_hardlinked_report_preserves_completed_output_and_snapshot(tmp_path: Path, target: str) -> None:
    output_dir = tmp_path / "completed"
    output_dir.mkdir()
    source_dir, commit = materialize_snapshot(output_dir / "labless_source")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config_path": "configs/main.yaml",
                "wandb": {"git": {"commit": commit}},
            }
        )
    )
    snapshot_config = source_dir / "configs" / "main.yaml"
    before_summary, before_snapshot = summary_path.read_bytes(), snapshot_config.read_bytes()
    report = tmp_path / "external-report.json"
    os.link(output_dir / target, report)

    result = run_audit("--output-dir", str(output_dir), "--main-ref", "HEAD", "--report", str(report))

    assert result.returncode == 1, result.stderr
    assert "alias" in json.loads(result.stdout)["error"]
    assert summary_path.read_bytes() == before_summary
    assert snapshot_config.read_bytes() == before_snapshot


def test_malformed_arguments_return_json_and_exit_one(tmp_path: Path) -> None:
    result = run_audit("--source-dir", str(tmp_path), "--source-commit", "a" * 40)

    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout)["eligible"] is False


def test_prelaunch_relative_paths_are_resolved_from_the_caller(tmp_path: Path) -> None:
    source_dir, commit = materialize_snapshot(tmp_path / "source")

    result = run_audit(
        "--source-dir",
        source_dir.name,
        "--source-commit",
        commit,
        "--review-config",
        "configs/main.yaml",
        "--main-ref",
        "HEAD",
        "--report",
        "audit.json",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "audit.json").read_text())["eligible"] is True
