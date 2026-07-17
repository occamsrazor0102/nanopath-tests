#!/usr/bin/env python3
# Audits immutable source snapshots with the checked-in Labless policy engine.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labless.submit_to_labless import collect_source_context, public_config_path


OPTIONS = {"main_ref", "output_dir", "report", "review_config", "source_commit", "source_dir"}


def parse_args(argv: list[str]) -> dict[str, str]:
    opts: dict[str, str] = {}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if not arg.startswith("--"):
            raise ValueError(f"unsupported argument {arg!r}; use --name value")
        key, separator, value = arg[2:].partition("=")
        key = key.replace("-", "_")
        if key not in OPTIONS:
            raise ValueError(f"unsupported argument --{key.replace('_', '-')}")
        if not separator:
            index += 1
            if index == len(argv) or argv[index].startswith("--"):
                raise ValueError(f"missing value for --{key.replace('_', '-')}")
            value = argv[index]
        if not value:
            raise ValueError(f"missing value for --{key.replace('_', '-')}")
        if key in opts:
            raise ValueError(f"duplicate argument --{key.replace('_', '-')}")
        opts[key] = os.path.expandvars(value)
        index += 1
    return opts


def caller_path(value: str, cwd: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else cwd / path).resolve()


def main(argv: list[str] | None = None) -> int:
    old_cwd = os.getcwd()
    caller_cwd = Path(old_cwd).resolve()
    try:
        opts = parse_args(sys.argv[1:] if argv is None else argv)
        output_dir = opts.get("output_dir")
        source_dir_value = opts.get("source_dir")
        if bool(output_dir) == bool(source_dir_value):
            raise ValueError("provide exactly one of --output-dir or --source-dir")
        if output_dir and opts.get("source_commit"):
            raise ValueError("--source-commit is only valid with --source-dir")
        if source_dir_value and (not opts.get("source_commit") or not opts.get("review_config")):
            raise ValueError("--source-dir requires --source-commit and --review-config")

        os.chdir(REPO_ROOT)
        main_name = opts.get("main_ref", "official/main")
        resolved_main = subprocess.run(
            ["git", "rev-parse", "--verify", f"{main_name}^{{commit}}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if resolved_main.returncode:
            raise ValueError(f"main ref does not resolve to a commit: {main_name}")
        main_ref = {"run_id": main_name, "commit": resolved_main.stdout.strip()}

        if output_dir:
            run_dir = caller_path(output_dir, caller_cwd)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("summary.json must contain an object")
            metadata = summary.get("wandb") if isinstance(summary.get("wandb"), dict) else {}
            git_meta = metadata.get("git") if isinstance(metadata.get("git"), dict) else {}
            source_dir = run_dir / "labless_source"
            source = str(metadata.get("source_artifact") or f"nanopath-source-{metadata.get('id', 'local')}")
            commit = git_meta.get("commit")
            recorded_config = summary.get("config_path")
            if not isinstance(recorded_config, str) or not recorded_config.strip():
                raise ValueError("summary.config_path is required for --output-dir")
            review_config = public_config_path(recorded_config)
            mode = "output_dir"
        else:
            source_dir = caller_path(source_dir_value, caller_cwd)
            summary, git_meta = {}, {}
            source = f"prelaunch-{source_dir.name}"
            commit = opts["source_commit"]
            review_config = public_config_path(opts["review_config"])
            mode = "prelaunch"

        context = collect_source_context(main_ref, summary, source_dir, source, commit, git_meta, review_config)
        payload = {
            "changed_files": context["changed_files"],
            "commit": context["commit"],
            "diff_summary": context["diff_summary"],
            "eligible": not context["locked_path_changes"] and not context["policy_errors"],
            "locked_path_changes": context["locked_path_changes"],
            "main_context": context["main_context"],
            "main_ref": main_name,
            "mode": mode,
            "new_source_files": context["new_source_files"],
            "policy_errors": context["policy_errors"],
            "review_config": review_config,
            "source_changed_files": context["source_changed_files"],
        }
        report = opts.get("report")
        if report:
            report_path = caller_path(report, caller_cwd)
            protected = [source_dir.resolve(), *([run_dir] if output_dir else [])]
            if any(report_path == path or path in report_path.parents for path in protected):
                raise ValueError("--report must be outside the audited output and source snapshot")
            if report_path.exists() and any(
                file.is_file() and report_path.samefile(file)
                for root in dict.fromkeys(protected)
                for file in root.rglob("*")
            ):
                raise ValueError("--report must not alias an audited output or source file")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = 0 if payload["eligible"] else 2
    except (AttributeError, KeyError, OSError, subprocess.SubprocessError, TypeError, ValueError, yaml.YAMLError) as exc:
        payload = {"eligible": False, "error": str(exc)}
        result = 1
    finally:
        os.chdir(old_cwd)
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
