import copy
import subprocess
from pathlib import Path

import pytest
import yaml


BASE_REVISION = "06679b7b61e16b402601c694cea5851f2e7bec99"
FROZEN_ACTIVE_SOURCE_REVISION = "701fd77526ddbdf82d80e851ad8a04c35539f525"
MISSING = object()
APPROVED_ADDITIONS = {
    "configs/molcap-probe-route-s7778.yaml",
    "configs/molcap-ema-relative-s7778.yaml",
    "docs/superpowers/plans/2026-07-14-molcap-paired-seed7778-recovery.md",
    "docs/superpowers/specs/2026-07-14-molcap-paired-seed7778-recovery-design.md",
    "tests/test_molcap_seed7778_recovery.py",
}
SEED_LEAVES = {
    "project.name",
    "project.output_dir",
    "data.split_seed",
    "train.seed",
}
PAIR_DELTA = {
    "project.name",
    "project.recipe_id",
    "project.output_dir",
    "molcap.history.enabled",
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


def git_bytes(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"])


def git_yaml(revision: str, path: str) -> dict[str, object]:
    value = yaml.safe_load(git_bytes(revision, path).decode("utf-8"))
    assert type(value) is dict
    return value


def worktree_yaml(path: str) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def changed_leaves(left: object, right: object, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix}
    if isinstance(left, dict) and isinstance(right, dict):
        keys = left.keys() | right.keys()
        return set().union(
            *(
                changed_leaves(
                    left.get(key, MISSING),
                    right.get(key, MISSING),
                    f"{prefix}.{key}".strip("."),
                )
                for key in keys
            )
        )
    if isinstance(left, list) and isinstance(right, list):
        return set().union(
            *(
                changed_leaves(
                    left[index] if index < len(left) else MISSING,
                    right[index] if index < len(right) else MISSING,
                    f"{prefix}[{index}]",
                )
                for index in range(max(len(left), len(right)))
            )
        )
    return {prefix} if left != right else set()


def assert_seed7778_copy(
    parent_path: str,
    child_path: str,
    *,
    expected_name: str,
    expected_output_dir: str,
) -> None:
    parent = git_yaml(BASE_REVISION, parent_path)
    child = worktree_yaml(child_path)
    assert changed_leaves(parent, child) == SEED_LEAVES
    assert child["project"]["name"] == expected_name
    assert child["project"]["output_dir"] == expected_output_dir
    assert child["project"]["recipe_id"] == parent["project"]["recipe_id"]
    split_seed = child["data"]["split_seed"]
    train_seed = child["train"]["seed"]
    assert type(split_seed) is int, "data.split_seed must be an int"
    assert split_seed == 7778
    assert type(train_seed) is int, "train.seed must be an int"
    assert train_seed == 7778
    assert child["train"]["resume"] is None


def git_tree_entries(revision: str) -> dict[str, tuple[str, str]]:
    raw = subprocess.check_output(["git", "ls-tree", "-rz", revision])
    result = {}
    for record in raw.rstrip(b"\0").split(b"\0"):
        metadata, path = record.split(b"\t", 1)
        mode, _kind, oid = metadata.decode("ascii").split()
        result[path.decode("utf-8")] = (mode, oid)
    return result


@pytest.mark.parametrize(
    ("left", "right", "prefix", "expected"),
    [
        pytest.param(128, 128.0, "scalar", {"scalar"}, id="int-versus-float"),
        pytest.param(False, 0, "scalar", {"scalar"}, id="bool-versus-int"),
        pytest.param(
            {"nested": [128, [False]]},
            {"nested": [128.0, [0]]},
            "",
            {"nested[0]", "nested[1][0]"},
            id="nested-list-equivalents",
        ),
    ],
)
def test_changed_leaves_is_recursively_type_sensitive(
    left: object,
    right: object,
    prefix: str,
    expected: set[str],
):
    assert changed_leaves(left, right, prefix) == expected


@pytest.mark.parametrize(
    ("section", "key"),
    [("data", "split_seed"), ("train", "seed")],
)
def test_seed7778_copy_rejects_float_seed_type(monkeypatch, section: str, key: str):
    child = copy.deepcopy(worktree_yaml("configs/molcap-probe-route-s7778.yaml"))
    child[section][key] = 7778.0
    monkeypatch.setitem(globals(), "worktree_yaml", lambda _path: child)

    with pytest.raises(AssertionError, match=rf"{section}\.{key} must be an int"):
        assert_seed7778_copy(
            "configs/molcap-probe-route-s7777.yaml",
            "configs/molcap-probe-route-s7778.yaml",
            expected_name="molcap-probe-route-s7778",
            expected_output_dir="/data/$USER/nanopath/molcap/molcap-probe-route-s7778",
        )


def test_route_seed7778_is_exact_four_leaf_copy_of_locked_parent():
    assert_seed7778_copy(
        "configs/molcap-probe-route-s7777.yaml",
        "configs/molcap-probe-route-s7778.yaml",
        expected_name="molcap-probe-route-s7778",
        expected_output_dir="/data/$USER/nanopath/molcap/molcap-probe-route-s7778",
    )


def test_relative_seed7778_is_exact_four_leaf_copy_of_locked_parent():
    assert_seed7778_copy(
        "configs/molcap-ema-relative-s7777.yaml",
        "configs/molcap-ema-relative-s7778.yaml",
        expected_name="molcap-ema-rel-s7778",
        expected_output_dir="/data/$USER/nanopath/molcap/molcap-ema-rel-s7778",
    )


def test_seed7778_pair_preserves_locked_route_relative_delta():
    locked_route = git_yaml(BASE_REVISION, "configs/molcap-probe-route-s7777.yaml")
    locked_relative = git_yaml(BASE_REVISION, "configs/molcap-ema-relative-s7777.yaml")
    route = worktree_yaml("configs/molcap-probe-route-s7778.yaml")
    relative = worktree_yaml("configs/molcap-ema-relative-s7778.yaml")
    assert changed_leaves(locked_route, locked_relative) == PAIR_DELTA
    assert changed_leaves(route, relative) == PAIR_DELTA


def test_relative_seed7778_locks_inherited_duplicate_key_loader_semantics():
    parent_path = "configs/molcap-ema-relative-s7777.yaml"
    child_path = "configs/molcap-ema-relative-s7778.yaml"
    parent_text = git_bytes(BASE_REVISION, parent_path).decode("utf-8")
    child_text = subprocess.check_output(
        ["git", "show", f":{child_path}"], text=True
    )

    def duplicate_key_occurrences(text: str) -> list[tuple[int, str]]:
        return [
            (line_number, line)
            for line_number, line in enumerate(text.splitlines(), start=1)
            if line.lstrip().startswith("min_participation_ratio:")
        ]

    expected_occurrences = [
        (82, "    min_participation_ratio: 16"),
        (91, "    min_participation_ratio: 0.5"),
    ]
    assert duplicate_key_occurrences(parent_text) == expected_occurrences
    assert duplicate_key_occurrences(child_text) == expected_occurrences

    runtime_source = git_bytes(BASE_REVISION, "train.py").decode("utf-8")
    assert (
        "cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))"
        in runtime_source
    )
    parent = yaml.safe_load(parent_text)
    child = yaml.safe_load(child_text)
    parent_value = parent["molcap"]["history"]["min_participation_ratio"]
    child_value = child["molcap"]["history"]["min_participation_ratio"]
    assert type(parent_value) is float
    assert type(child_value) is float
    assert child_value == parent_value == 0.5


def test_seed7778_frozen_active_source_revision_is_exact_approved_additive_tree():
    baseline = git_tree_entries(BASE_REVISION)
    candidate = git_tree_entries(FROZEN_ACTIVE_SOURCE_REVISION)
    assert set(candidate) == set(baseline) | APPROVED_ADDITIONS
    assert {path: candidate[path] for path in baseline} == baseline
