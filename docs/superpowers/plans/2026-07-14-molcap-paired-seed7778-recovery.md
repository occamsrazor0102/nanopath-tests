# MolCap Paired Seed-7778 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, lock, execute, and publicly submit a fresh seed-7778 MolCap probe-route control and matched-latest EMA patient-centroid arm without reusing or mutating the stopped seed-7777 experiment.

**Architecture:** Add two scored configs as exact four-leaf copies of the locked seed-7777 parents and enforce the entire candidate as one additive Git commit above `06679b7`. Clone the already reviewed ignored Modal harness, replace archived-seed replay with dynamic seed-7778 mapping/sample anchors and same-hardware paired replay, and fix physical containment against `PERSISTENT_ROOT.resolve()`. A new authoritative ledger serializes CPU preflight, paired smokes/calibrations, conditional previews, one sealed launch certificate, two fresh full runs, and per-arm Labless eligibility.

**Tech Stack:** Python 3.12, PyTorch 2.8.0+cu129, PyYAML, NumPy, pytest/unittest, Modal CPU/B200/exact-H100 functions, PowerShell, Git, and the existing Labless submission client.

## Global Constraints

- The binding design is `docs/superpowers/specs/2026-07-14-molcap-paired-seed7778-recovery-design.md`.
- The immutable baseline is `06679b7b61e16b402601c694cea5851f2e7bec99`.
- The final tracked candidate is exactly one commit above that baseline and adds exactly five paths: this plan, the approved spec, two seed-7778 configs, and one forced-tracked config test.
- Every path already tracked at `06679b7` keeps its exact Git mode and blob OID. Do not edit `train.py`, `molcap_relative_gate.py`, `probe.py`, either seed-7777 parent config, existing tests, or benchmark files.
- `tests/` and `.superpowers/` are ignored. Force-add only `tests/test_molcap_seed7778_recovery.py`; never force-add the operational harness, its tests, logs, locks, ledgers, or reports.
- The seed-7777 harness, ledger, reports, outputs, and Labless result remain read-only historical evidence.
- Both scored configs change only `project.name`, `project.output_dir`, `data.split_seed`, and `train.seed` from their own locked parents. Both seeds are exactly `7778`; `resume` remains null.
- Canonical target SHA-256 remains `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`.
- MolCap weight `0.03`, ramp `0.50` through `0.75`, momentum `0.9`, relative thresholds, exact 256-permutation procedure, model, crops, batch, probes, and one-million-sample schedule are unchanged.
- The relative gate requires coverage `>= 0.95`, `min_slide_updates = 2`, at least `512` matched patients, centroid norms `> 1e-6`, positive finite traces, trace ratio `>= 0.05263157894736842`, effective-rank and participation ratios `>= 0.5`, alignment `> 0.0`, and one-sided 256-permutation `p <= 0.01` under domain `molcap-matched-latest-v1` and PyTorch `2.8.0`.
- Raw cosine and absolute effective-rank/participation values remain diagnostic only; every gate report retains complete spectra and all 256 permutation alignments.
- Action caps are `1,024` smoke, `32,768` calibration, conditional `40,960` preview, and `1,000,000` full; full completion is `7,812` steps and `999,936` presentations.
- Hardware order is B200 smokes, paired B200 calibrations, paired exact-H100 calibrations, then fresh B200 route and relative full runs.
- Peak relative-minus-route memory must be at most `0.5 GiB` on each calibration hardware. Timing and the two-hour H100 projection are reported but non-gating.
- Each action allows an initial attempt plus at most two exact fresh infrastructure retries. Source/data/code/path/mechanics/replay failures record a global seed-7779 stop; gate failure stops the relative arm before supervision; OOM/non-finite state leaves that arm incomplete.
- Submit every completed full arm regardless of score. Route Labless response, validation, and score never gate the relative launch.
- Formal BSC promotion is out of scope. The only causal decision is relative-s7778 minus route-s7778 under the frozen mechanism criteria.
- No seed-7778 external action is allowed until tracked tests, complete repo tests, ignored-harness tests, exact-SHA lock generation, and two independent read-only reviews pass.
- All source edits use `apply_patch`. Mechanical copies of the frozen ignored harness/test may use `Copy-Item`; every semantic change after the copy uses `apply_patch`.

## File Structure

- Create `configs/molcap-probe-route-s7778.yaml`: fresh paired route scored config.
- Create `configs/molcap-ema-relative-s7778.yaml`: fresh paired relative-centroid scored config.
- Create `tests/test_molcap_seed7778_recovery.py`: exact parent-delta, pair-delta, and additive-tree lock.
- Create `.superpowers/sdd/modal_molcap_paired_s7778.py`: ignored Modal controller and evidence state machine.
- Create `.superpowers/sdd/test_modal_molcap_paired_s7778.py`: ignored controller contract suite.
- Create `.superpowers/sdd/molcap-paired-s7778-launch-lock.json`: ignored non-self-referential exact source/harness/test lock.
- Create `.superpowers/sdd/molcap-paired-s7778-evidence.json`: ignored local cache of the authoritative volume ledger.
- Create `.superpowers/sdd/seed7778-logs/`: ignored per-action stdout/stderr captures.
- Create `.superpowers/sdd/task-7-seed7778-report.md`: ignored durable execution, metric, and submission report.
- Preserve `.superpowers/sdd/modal_molcap_centroid.py`, `.superpowers/sdd/test_modal_molcap_centroid.py`, `.superpowers/sdd/task-6-report.md`, and the seed-7777 volume namespace byte-for-byte.

---

### Task 1: Add the Exact Additive Seed-7778 Source Candidate

**Files:**
- Create: `configs/molcap-probe-route-s7778.yaml`
- Create: `configs/molcap-ema-relative-s7778.yaml`
- Create: `tests/test_molcap_seed7778_recovery.py`
- Preserve: every tracked path at `06679b7`

**Interfaces:**
- Consumes: Git tree `06679b7`, approved spec, and this plan.
- Produces: one clean additive commit, two scored configs, and a test-enforced source surface used by the harness lock.

- [ ] **Step 1: Write and stage the failing additive-tree/config test**

Create `tests/test_molcap_seed7778_recovery.py` with this complete contract, then force-stage it so the index-based manifest can see the ignored test:

```python
import subprocess
from pathlib import Path

import yaml


BASE_REVISION = "06679b7b61e16b402601c694cea5851f2e7bec99"
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
    assert child["data"]["split_seed"] == 7778
    assert child["train"]["seed"] == 7778
    assert child["train"]["resume"] is None


def git_tree_entries(revision: str) -> dict[str, tuple[str, str]]:
    raw = subprocess.check_output(["git", "ls-tree", "-rz", revision])
    result = {}
    for record in raw.rstrip(b"\0").split(b"\0"):
        metadata, path = record.split(b"\t", 1)
        mode, _kind, oid = metadata.decode("ascii").split()
        result[path.decode("utf-8")] = (mode, oid)
    return result


def git_index_entries() -> dict[str, tuple[str, str]]:
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"])
    result = {}
    for record in raw.rstrip(b"\0").split(b"\0"):
        metadata, path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        assert stage == "0"
        result[path.decode("utf-8")] = (mode, oid)
    return result


def worktree_blob_oid(path: str) -> str:
    return subprocess.check_output(
        ["git", "hash-object", f"--path={path}", "--", path], text=True
    ).strip()


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


def test_seed7778_candidate_index_is_exact_approved_additive_tree():
    baseline = git_tree_entries(BASE_REVISION)
    candidate = git_index_entries()
    assert set(candidate) == set(baseline) | APPROVED_ADDITIONS
    assert {path: candidate[path] for path in baseline} == baseline
    for path, (_mode, indexed_oid) in candidate.items():
        assert worktree_blob_oid(path) == indexed_oid
```

Run:

```powershell
git add -f -- tests/test_molcap_seed7778_recovery.py
python -m pytest -q tests/test_molcap_seed7778_recovery.py
```

Expected: FAIL because both seed-7778 config paths are absent and the indexed candidate lacks two approved additions.

- [ ] **Step 2: Create the two configs by mechanical copy plus the exact four edits**

Run the mechanical copies:

```powershell
Copy-Item -LiteralPath configs/molcap-probe-route-s7777.yaml -Destination configs/molcap-probe-route-s7778.yaml
Copy-Item -LiteralPath configs/molcap-ema-relative-s7777.yaml -Destination configs/molcap-ema-relative-s7778.yaml
```

Use `apply_patch` to make exactly these replacements:

```diff
--- configs/molcap-probe-route-s7777.yaml
+++ configs/molcap-probe-route-s7778.yaml
-  name: molcap-probe-route-s7777
+  name: molcap-probe-route-s7778
-  output_dir: /data/$USER/nanopath/molcap/molcap-probe-route-s7777
+  output_dir: /data/$USER/nanopath/molcap/molcap-probe-route-s7778
-  split_seed: 7777
+  split_seed: 7778
-  seed: 7777
+  seed: 7778
```

```diff
--- configs/molcap-ema-relative-s7777.yaml
+++ configs/molcap-ema-relative-s7778.yaml
-  name: molcap-ema-rel-s7777
+  name: molcap-ema-rel-s7778
-  output_dir: /data/$USER/nanopath/molcap/molcap-ema-rel-s7777
+  output_dir: /data/$USER/nanopath/molcap/molcap-ema-rel-s7778
-  split_seed: 7777
+  split_seed: 7778
-  seed: 7777
+  seed: 7778
```

- [ ] **Step 3: Stage the exact candidate and make the focused tests green**

Run:

```powershell
git add -- configs/molcap-probe-route-s7778.yaml configs/molcap-ema-relative-s7778.yaml docs/superpowers/plans/2026-07-14-molcap-paired-seed7778-recovery.md docs/superpowers/specs/2026-07-14-molcap-paired-seed7778-recovery-design.md
git add -f -- tests/test_molcap_seed7778_recovery.py
python -m pytest -q tests/test_molcap_seed7778_recovery.py
python -m pytest -q tests/test_molcap_config.py tests/test_molcap_seed7778_recovery.py
```

Expected: `4 passed` for the focused file; both files pass together with no failure.

- [ ] **Step 4: Verify the whole tracked suite and exact staged tree**

Run:

```powershell
python -m pytest -q
git diff --cached --check 06679b7b61e16b402601c694cea5851f2e7bec99 --
git diff --cached --name-status --no-renames 06679b7b61e16b402601c694cea5851f2e7bec99 --
```

Expected: `381 passed, 2 skipped`; the diff lists exactly five additions and no modification or deletion.

- [ ] **Step 5: Amend the sole additive commit instead of creating a second commit**

Run:

```powershell
git commit --amend -m "experiment: freeze paired seed-7778 recovery"
git rev-list --count 06679b7b61e16b402601c694cea5851f2e7bec99..HEAD
python -m pytest -q tests/test_molcap_seed7778_recovery.py
git status --short --branch
```

Expected: revision count `1`, `4 passed`, and a clean branch. Record the resulting HEAD; it is the only permitted seed-7778 `SOURCE_COMMIT`.

---

### Task 2: Clone the Frozen Harness and Lock Namespace, Registry, and Runtime Paths

**Files:**
- Create mechanically: `.superpowers/sdd/modal_molcap_paired_s7778.py`
- Create mechanically: `.superpowers/sdd/test_modal_molcap_paired_s7778.py`
- Preserve: `.superpowers/sdd/modal_molcap_centroid.py`
- Preserve: `.superpowers/sdd/test_modal_molcap_centroid.py`

**Interfaces:**
- Consumes: Task 1 `SOURCE_COMMIT`, scored configs, and the frozen seed-7777 harness architecture.
- Produces: seed-7778 namespace constants, exact action registry, alias-safe path validation, and a local lock-material surface.

- [ ] **Step 1: Mechanically copy the reviewed ignored harness and test**

Run:

```powershell
Copy-Item -LiteralPath .superpowers/sdd/modal_molcap_centroid.py -Destination .superpowers/sdd/modal_molcap_paired_s7778.py
Copy-Item -LiteralPath .superpowers/sdd/test_modal_molcap_centroid.py -Destination .superpowers/sdd/test_modal_molcap_paired_s7778.py
```

Change the test import with `apply_patch`:

```python
import modal_molcap_paired_s7778 as harness
```

- [ ] **Step 2: Write failing namespace, registry, and alias-path tests**

Update the copied tests to require these exact constants and registry tuples:

```python
assert harness.SEED == 7_778
assert harness.REQUIRED_NEXT_SEED == 7_779
assert harness.MAX_ATTEMPTS == 3
assert harness.RUN_ROOT.as_posix() == (
    "/data/experiments/readout-local-context/matched-latest-s7778"
)
assert harness.PERSISTENT_EVIDENCE_PATH.as_posix() == (
    "/persistent/experiments/readout-local-context/"
    "matched-latest-s7778/evidence.json"
)
assert harness.CONFIG_BY_ROLE == {
    "route": "molcap-probe-route-s7778.yaml",
    "relative_centroid": "molcap-ema-relative-s7778.yaml",
}
```

The registry expectation is exactly:

```python
EXPECTED_ACTIONS = {
    "preflight-paired-s7778-cpu": ("preflight", "CPU", 0, None, False),
    "smoke-route-s7778-b200": ("route", "B200", 1_024, "smoke/route-s7778-b200", True),
    "smoke-relative-s7778-b200": ("relative_centroid", "B200", 1_024, "smoke/relative-s7778-b200", True),
    "calibrate-route-s7778-b200": ("route", "B200", 32_768, "calibrate/route-s7778-b200", True),
    "calibrate-relative-s7778-b200": ("relative_centroid", "B200", 32_768, "calibrate/relative-s7778-b200", True),
    "preview-relative-s7778-b200": ("relative_centroid", "B200", 40_960, "preview/relative-s7778-b200", True),
    "calibrate-route-s7778-h100": ("route", "H100!", 32_768, "calibrate/route-s7778-h100", True),
    "calibrate-relative-s7778-h100": ("relative_centroid", "H100!", 32_768, "calibrate/relative-s7778-h100", True),
    "preview-relative-s7778-h100": ("relative_centroid", "H100!", 40_960, "preview/relative-s7778-h100", True),
    "full-route-s7778-b200": ("route", "B200", 1_000_000, "full/molcap-probe-route-s7778", True),
    "full-relative-s7778-b200": ("relative_centroid", "B200", 1_000_000, "full/molcap-ema-rel-s7778", True),
}
```

Add direct mocked-`Path.resolve()` tests named:

```text
test_runtime_paths_accept_data_aliases_into_distinct_resolved_volume_root
test_runtime_paths_reject_resolved_escape_from_volume
test_runtime_paths_reject_staged_dataset_alias_into_volume
test_runtime_paths_preserve_exact_logical_config_strings
```

The resolver maps logical `/persistent` to `/__modal/volumes/vo-test`, maps `/data/experiments`, `/data/repo-data`, `/data/torch`, and `/data/huggingface` into that backend, and maps `/data/nanopath_parquet` to `/ephemeral/nanopath_parquet`. The escape case maps one persistent alias to `/escape`; the dataset case maps the staged dataset into the backend.

Run:

```powershell
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "registry or runtime_paths or topology"
```

Expected: FAIL on the old seed-7777 namespace, old single-full registry, and unresolved-root containment.

- [ ] **Step 3: Implement the exact seed-7778 constants and registry**

Use `apply_patch` to set:

```python
SEED = 7_778
REQUIRED_NEXT_SEED = 7_779
MAX_ATTEMPTS = 3
SMOKE_ANCHOR_COUNT = 1_024
CALIBRATION_ANCHOR_COUNT = 8_192

RUN_ROOT = Path("/data/experiments/readout-local-context/matched-latest-s7778")
LOCAL_EVIDENCE_PATH = (
    LOCAL_ROOT / ".superpowers" / "sdd" / "molcap-paired-s7778-evidence.json"
)
PERSISTENT_EVIDENCE_PATH = (
    PERSISTENT_ROOT
    / "experiments/readout-local-context/matched-latest-s7778/evidence.json"
)
CONFIG_BY_ROLE = {
    "route": "molcap-probe-route-s7778.yaml",
    "relative_centroid": "molcap-ema-relative-s7778.yaml",
}
```

Set `SOURCE_COMMIT` to the exact Task-1 HEAD recorded after the amend. Remove `MAPPING_SHA256`, `SAMPLE_SHA256`, every `EXPECTED_SOURCE_*`/`EXPECTED_LOCKED_*`/`EXPECTED_LABLESS_*` constant, every `ARCHIVED_*` constant, and the `archived_reference` registry field. Expected source/config/manifests come from the separately sealed launch-lock payload. Set the app name to `nanopath-molcap-paired-s7778`. Replace singular `FULL_ACTION` with:

```python
ROUTE_FULL_ACTION = registered_action(kind="full", role="route", gpu="B200")
RELATIVE_FULL_ACTION = registered_action(
    kind="full", role="relative_centroid", gpu="B200"
)
FULL_ACTIONS = (ROUTE_FULL_ACTION, RELATIVE_FULL_ACTION)
```

- [ ] **Step 4: Fix physical containment without changing logical paths**

Implement exactly:

```python
def _require_persistent_runtime_paths(config: dict[str, object]) -> None:
    assert config["data"]["dataset_dir"] == DATASET_ROOT.as_posix()
    persistent_root_resolved = PERSISTENT_ROOT.resolve()
    dataset_resolved = Path(config["data"]["dataset_dir"]).resolve()
    assert dataset_resolved == DATASET_ROOT.resolve()
    assert not dataset_resolved.is_relative_to(persistent_root_resolved)
    output_paths = [
        Path(config["project"]["output_dir"]),
        Path(config["project"]["wandb_dir"]),
        Path(config["molcap"]["targets"]),
        *[Path(path) for path in config["probe"]["dataset_roots"].values()],
        Path(os.environ.get("TORCH_HOME", "/data/torch")),
        Path(os.environ.get("HF_HOME", "/data/huggingface")),
    ]
    for path in output_paths:
        resolved = path.resolve()
        assert resolved.is_relative_to(persistent_root_resolved), (path, resolved)
```

Keep every exact logical assertion in `derive_runtime_config()` unchanged except for the new run/config names.

- [ ] **Step 5: Add a non-mutating lock-material interface**

Create these interfaces:

```python
LAUNCH_LOCK_PATH = (
    LOCAL_ROOT / ".superpowers" / "sdd" / "molcap-paired-s7778-launch-lock.json"
)

def candidate_launch_lock_material() -> dict[str, object]:
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=LOCAL_ROOT, text=True
    ).strip()
    runtime_paths = tuple(
        path for path in _git_paths(source_commit) if _included_runtime_path(path)
    )
    locked_paths = tuple(
        path
        for path in _git_paths(LOCKED_COMMIT)
        if path == "probe.py" or path.startswith("benchmarking/")
    )
    source_manifest = _manifest_for_revision(source_commit, runtime_paths)
    locked_manifest = _manifest_for_revision(LOCKED_COMMIT, locked_paths)
    labless_manifest = _labless_snapshot_manifest(
        source_manifest,
        _normalized_source_bytes(_git_bytes(source_commit, ".gitignore")),
    )
    bundle = {
        "schema_version": 1,
        "source_commit": source_commit,
        "locked_commit": LOCKED_COMMIT,
        "source_manifest": source_manifest,
        "locked_manifest": locked_manifest,
        "labless_source_manifest": labless_manifest,
    }
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "source_bundle_sha256": _bundle_digest(bundle),
        "source_manifest_sha256": _manifest_digest(source_manifest),
        "locked_commit": LOCKED_COMMIT,
        "locked_manifest_sha256": _manifest_digest(locked_manifest),
        "labless_manifest_sha256": _manifest_digest(labless_manifest),
        "route_config_sha256": source_manifest[
            "configs/molcap-probe-route-s7778.yaml"
        ],
        "relative_config_sha256": source_manifest[
            "configs/molcap-ema-relative-s7778.yaml"
        ],
        "target_sha256": TARGET_SHA256,
        "action_registry_sha256": hashlib.sha256(
            strict_json(ACTION_REGISTRY).encode("utf-8")
        ).hexdigest(),
    }

def load_launch_lock() -> dict[str, object]:
    lock = json.loads(LAUNCH_LOCK_PATH.read_text(encoding="utf-8"))
    material = candidate_launch_lock_material()
    assert set(lock) == set(material) | {"harness_sha256", "test_sha256"}
    assert {key: lock[key] for key in material} == material
    assert lock["harness_sha256"] == _sha256(Path(__file__))
    test_path = Path(__file__).with_name("test_modal_molcap_paired_s7778.py")
    assert lock["test_sha256"] == _sha256(test_path)
    return copy.deepcopy(lock)
```

`candidate_launch_lock_material()` returns source commit, source bundle digest, source manifest digest, Labless snapshot manifest digest, locked commit/manifest, both scored-config hashes, target SHA, and `SHA256(strict_json(ACTION_REGISTRY))`. It performs no Modal call and writes no file. `load_launch_lock()` additionally verifies the current harness and harness-test SHA values stored in the separate JSON lock, avoiding a self-referential hash constant.

Refactor `_validate_source_bundle(bundle, launch_lock)`, `_client_source_bundle(launch_lock, refresh=False)`, `_require_scored_config_pair(launch_lock)`, CPU/B200/H100 dispatchers, and both remote execution functions to accept the exact launch-lock dict. Local entrypoint loads and validates the file once, passes a deep copy to the remote function, and the remote compares every source/config/manifest value with that payload; remote code never tries to open the local lock file.

The local entrypoint recognizes only three non-registered read-only commands: `lock-material`, `describe`, and `inspect-ledger`. `lock-material` prints candidate material; `describe` requires a valid lock; `inspect-ledger` prints the authoritative ledger without launching compute.

- [ ] **Step 6: Run the focused and complete copied-harness suites**

Run:

```powershell
python -m py_compile .superpowers/sdd/modal_molcap_paired_s7778.py .superpowers/sdd/test_modal_molcap_paired_s7778.py
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "registry or runtime_paths or topology"
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py
```

Expected: exit `0` for all commands; no seed-7777 action, config, run root, or unresolved containment assertion remains in an active contract.

---

### Task 3: Implement Dynamic Preflight Locks, Sample Anchors, and Paired Replay

**Files:**
- Modify with `apply_patch`: `.superpowers/sdd/modal_molcap_paired_s7778.py`
- Modify with `apply_patch`: `.superpowers/sdd/test_modal_molcap_paired_s7778.py`

**Interfaces:**
- Consumes: Task 2 registry, source lock material, existing staging/replay/shadow helpers, and Task 1 configs.
- Produces: immutable action-0 experiment lock, dynamic 1,024/8,192 sample anchors, and same-hardware route-relative checkpoint certificates.

- [ ] **Step 1: Write failing dynamic-lock and sample-anchor tests**

Add tests named:

```text
test_action0_freezes_mapping_and_canonical_source_manifest
test_every_gpu_action_requires_the_exact_action0_lock_before_training
test_action1_anchors_1024_once_and_action2_must_match
test_action3_matches_1024_then_anchors_8192
test_actions4_through8_and_previews_match_8192
test_metric_or_gate_outcome_cannot_replace_an_anchor
test_sample_digest_is_exact_contiguous_little_endian_i8
```

Use a prefix tensor with more than 1,024 elements to prove that the 1,024 digest is computed over `prefix[:1024]`, and compare it with:

```python
expected = hashlib.sha256(
    np.asarray(prefix[:count].cpu().numpy(), dtype="<i8").tobytes(order="C")
).hexdigest()
```

Run:

```powershell
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "action0 or anchor or sample_digest"
```

Expected: FAIL because the copied harness uses pre-observed global mapping/sample constants and requires an exact-length prefix.

- [ ] **Step 2: Implement dynamic mapping and experiment-lock propagation**

Remove every comparison with a global mapping digest. `_independent_training_mapping()` computes seed `7778` and returns its digest. Action 0 records the canonical persistent-source inventory and mapping. Add:

```python
EXPERIMENT_LOCK_KEYS = {
    "schema_version",
    "source_commit",
    "source_bundle_sha256",
    "source_manifest_sha256",
    "labless_manifest_sha256",
    "locked_commit",
    "locked_manifest_sha256",
    "scored_config_sha256",
    "target",
    "dataset",
    "mapping",
    "launch_lock_sha256",
}


def validate_experiment_lock(lock: dict[str, object]) -> bool:
    assert type(lock) is dict and set(lock) == EXPERIMENT_LOCK_KEYS
    assert lock["schema_version"] == 1
    assert lock["source_commit"] == SOURCE_COMMIT
    assert lock["scored_config_sha256"].keys() == CONFIG_BY_ROLE.keys()
    assert lock["target"]["sha256"] == TARGET_SHA256
    assert lock["target"]["rows"] == 11_428
    assert lock["target"]["dim"] == 384
    assert lock["dataset"]["file_count"] == 200
    assert lock["mapping"]["version"] == 1
    assert lock["mapping"]["seed"] == SEED
    assert type(lock["mapping"]["digest"]) is str
    assert len(lock["mapping"]["digest"]) == 64
    return True


def experiment_lock_from_ledger(ledger: dict[str, object]) -> dict[str, object]:
    action = registered_action(kind="preflight", role="preflight", gpu="CPU")
    run = ledger["runs"][action]
    lock = copy.deepcopy(run["preflight"])
    validate_experiment_lock(lock)
    return lock
```

Every GPU dispatcher receives this lock and current anchors. Before `train.py`, `_require_source_locked_preflight()` recomputes source, target, locked files, destination staging, and mapping, then requires equality with the action-0 lock. A changed mapping or source manifest returns `seed7779_stop` before training.

- [ ] **Step 3: Implement exact immutable sample anchors**

Replace the exact-length helper with:

```python
def _sample_prefix_digest(
    checkpoint: dict[str, object], *, expected_count: int
) -> tuple[str, int]:
    import torch

    assert checkpoint["molcap_sample_order_available"] is True
    prefix = checkpoint["molcap_sample_order_prefix"]
    assert isinstance(prefix, torch.Tensor) and prefix.dtype == torch.int64
    assert type(expected_count) is int and 0 < expected_count <= 8_192
    assert prefix.ndim == 1 and prefix.numel() >= expected_count
    array = np.asarray(
        prefix[:expected_count].detach().cpu().contiguous().numpy(), dtype="<i8"
    )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest(), expected_count


def sample_anchor_record(
    action: str, checkpoint: dict[str, object], count: int
) -> dict[str, object]:
    digest, observed = _sample_prefix_digest(checkpoint, expected_count=count)
    return {
        "action": action,
        "sha256": digest,
        "count": observed,
        "dtype": "<i8",
    }
```

Add `sample_anchors = {"smoke_1024": None, "calibration_8192": None}` to the ledger. Action 1 alone may set `smoke_1024`; Action 2 must match it. Action 3 must match the first 1,024 then alone may set `calibration_8192`; Actions 4-6, every required preview, and both fulls must match it. Anchor comparison occurs before a result is appended to `completed_actions`; metrics and gate results are never inputs.

- [ ] **Step 4: Write failing paired replay/history tests**

Retain normalized common-core and shadow-provenance mutation coverage, remove every archived seed-7777 checkpoint expectation, and add:

```text
test_b200_relative_calibration_matches_b200_route_common_core
test_h100_relative_calibration_matches_h100_route_common_core
test_route_calibration_has_no_history_shadow_or_provenance
test_relative_calibration_validates_primary_history_and_latest_shadow
test_each_common_core_mutation_records_global_seed7779
```

Run:

```powershell
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "replay or history or shadow or common_core"
```

Expected: FAIL because `certify_checkpoint_replay()` still loads seed-7777 archives.

- [ ] **Step 5: Replace archived replay with paired same-hardware replay**

Create:

```python
def paired_route_action(action: str) -> str:
    spec = action_spec(action)
    assert spec["kind"] == "calibrate" and spec["role"] == "relative_centroid"
    return registered_action(kind="calibrate", role="route", gpu=str(spec["gpu"]))


def certify_paired_calibration(
    action: str,
    relative_checkpoint_path: Path,
    *,
    experiment_lock: dict[str, object],
    staging: dict[str, object],
) -> dict[str, object]:
    import torch

    route_action = paired_route_action(action)
    route_output = RUN_ROOT / str(action_spec(route_action)["output"])
    route_path = route_output / "latest.pt"
    assert route_path.is_file() and relative_checkpoint_path.is_file()
    route = torch.load(route_path, map_location="cpu", weights_only=False)
    relative = torch.load(
        relative_checkpoint_path, map_location="cpu", weights_only=False
    )
    require_route_checkpoint_state(route)
    require_exact_normalized_replay(
        normalized_checkpoint_state(relative),
        normalized_checkpoint_state(route),
    )
    mapping = experiment_lock["mapping"]
    history = _validate_primary_history(
        relative["molcap_history"],
        mapping=mapping,
        expected_step=256,
        expected_presentations=32_768,
    )
    shadow = validate_shadow_state(
        relative[REPLAY_SHADOW_EXCLUSION],
        target_sha256=TARGET_SHA256,
        mapping_digest=mapping["digest"],
        expected_step=256,
        expected_presentations=32_768,
        primary_state=relative["molcap_history"],
    )
    validate_shadow_provenance(
        relative,
        shadow_state=relative[REPLAY_SHADOW_EXCLUSION],
        gate_report=None,
    )
    sample_sha256, sample_count = _sample_prefix_digest(
        relative, expected_count=CALIBRATION_ANCHOR_COUNT
    )
    return {
        "action": action,
        "route_action": route_action,
        "route_checkpoint_sha256": _sha256(route_path),
        "relative_checkpoint_sha256": _sha256(relative_checkpoint_path),
        "common_core_exact": True,
        "sample_order_digest": sample_sha256,
        "sample_order_count": sample_count,
        "history": history,
        "shadow": shadow,
        "source_commit": experiment_lock["source_commit"],
        "source_manifest_sha256": experiment_lock["source_manifest_sha256"],
        "config_sha256": experiment_lock["scored_config_sha256"][
            "relative_centroid"
        ],
        "target_sha256": experiment_lock["target"]["sha256"],
        "mapping_digest": mapping["digest"],
        "locked_manifest_sha256": experiment_lock["locked_manifest_sha256"],
        "staged_manifest_sha256": staging["source_manifest_sha256"],
    }
```

The implementation derives the route output from `ACTION_REGISTRY`, loads both checkpoints on CPU, requires route absence of history/shadow/provenance, calls `require_exact_normalized_replay()` for every common-core section, validates the relative primary history and latest shadow with the dynamic mapping, verifies the 8,192 anchor, and records both checkpoint SHA values, `common_core_exact=True`, history/shadow digests, gate preview, mapping, staging, source, target, config, and sample identities.

Require each preview/production gate report to contain finite raw diagnostics, both complete covariance spectra, the observed alignment, the deterministic seed, and exactly 256 stored permutation alignments. These fields are retained even though raw cosine and absolute rank are not pass/fail thresholds.

For route calibrations, record checkpoint SHA, route-state absence, and the anchor; no paired comparison is claimed until its relative action. Preserve `_synthetic_shadow_no_effect()` and real B200 common-core mechanics, but seed all synthetic tensors/generators with `7778`.

The route calibration `replay` object uses exact keys `action`, `checkpoint_sha256`, `sample_order_digest`, `sample_order_count`, `route_state_absent`, `source_commit`, `source_manifest_sha256`, `config_sha256`, `target_sha256`, `mapping_digest`, `locked_manifest_sha256`, and `staged_manifest_sha256`. The relative replay object is exactly the return from `certify_paired_calibration()` plus its gate-preview digest.

- [ ] **Step 6: Run focused and complete ignored-harness tests**

Run:

```powershell
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "action0 or anchor or replay or history or shadow or common_core"
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py
```

Expected: exit `0`; no active code reads `MAPPING_SHA256`, `SAMPLE_SHA256`, `ARCHIVED_CHECKPOINT`, or `ARCHIVED_EMA_HISTORY`.

---

### Task 4: Implement the Durable Ledger, Bounded Retry, Certificate, Two Fulls, and Publication Independence

**Files:**
- Modify with `apply_patch`: `.superpowers/sdd/modal_molcap_paired_s7778.py`
- Modify with `apply_patch`: `.superpowers/sdd/test_modal_molcap_paired_s7778.py`

**Interfaces:**
- Consumes: Task 3 experiment lock, anchors, paired calibration evidence, and existing durable JSON/upload helpers.
- Produces: strict authoritative state machine, one immutable certificate, ordered route/relative full eligibility, and publication-independent training state.

- [ ] **Step 1: Write failing retry and stop-classification tests**

Add:

```text
test_only_explicit_pre_scientific_infrastructure_failure_is_retryable
test_retry_requires_identical_artifact_identity_and_fresh_state
test_three_total_attempts_are_allowed_and_fourth_is_blocked
test_resource_exhaustion_or_cuda_oom_is_not_blanket_infrastructure
test_code_mechanics_replay_and_path_failures_record_global_seed7779
test_gate_failure_is_arm_local_and_oom_nonfinite_are_arm_incomplete
```

Run:

```powershell
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "retry or infrastructure or seed7779 or incomplete"
```

Expected: FAIL on the copied unbounded/broad retry behavior and seed-7778 stop schema.

- [ ] **Step 2: Implement the strict seed-7778 ledger schema and bounded attempts**

The exact top-level keys are:

```python
EVIDENCE_LEDGER_KEYS = {
    "schema_version",
    "source_commit",
    "launch_lock_sha256",
    "completed_actions",
    "runs",
    "sample_anchors",
    "infrastructure_failures",
    "infrastructure_stop",
    "seed7779_stop",
    "caption_supervision_stop",
    "incomplete_arms",
    "launch_certificate",
    "launch_certificate_sha256",
    "labless_eligibility",
}
```

Initialize per-arm fields as:

```python
"sample_anchors": {"smoke_1024": None, "calibration_8192": None},
"incomplete_arms": {"route": None, "relative_centroid": None},
"labless_eligibility": {"route": None, "relative_centroid": None},
```

Rename `record_seed7778_stop()` to `record_seed7779_stop()` and require `required_seed == 7779`. Infrastructure retry requires: explicit provider allocation/preemption, service/network transport, container loss, or volume I/O; phase before any identity/scientific assertion; identical launch-lock/action identity; no completed run; and fewer than three recorded attempts. CUDA OOM, generic `ResourceExhaustedError`, code exceptions, non-finite state, manual post-training cancellation, gate, mechanics, replay, and identity failures are never infrastructure retries. The third infrastructure failure records `infrastructure_stop`; an attempt-four launch is rejected.

- [ ] **Step 3: Write failing state-machine and certificate tests**

Add all four B200/H100 preview combinations plus:

```text
test_route_full_is_required_before_relative_full
test_certificate_is_built_once_before_route_full
test_both_full_actions_verify_identical_certificate_bytes_and_sha
test_route_full_completion_cannot_mutate_certificate
test_certificate_or_full_restaging_mutation_records_seed7779
```

Run:

```powershell
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "sequence or preview or certificate or full"
```

Expected: FAIL because the copied controller has one full action, inserts previews after all base actions, and can rebuild its certificate.

- [ ] **Step 4: Implement the exact next-action state machine**

Implement `_required_next_action(ledger) -> str | None` with this order:

```text
preflight-paired-s7778-cpu
smoke-route-s7778-b200
smoke-relative-s7778-b200
calibrate-route-s7778-b200
calibrate-relative-s7778-b200
preview-relative-s7778-b200 only when B200 maturity < 0.95
calibrate-route-s7778-h100
calibrate-relative-s7778-h100
preview-relative-s7778-h100 only when H100 maturity < 0.95
seal certificate
full-route-s7778-b200
full-relative-s7778-b200
complete
```

Preview requirement is evaluated immediately after each relative calibration. `None` is returned only after both full actions complete or a restrictive stop makes further launch illegal. Route local completion and artifact validation are mandatory before the relative full; publication is not.

- [ ] **Step 5: Build and seal one non-self-referential launch certificate**

Replace the seed-7777 builder/validator with:

```python
def required_completed_prefull_actions(
    ledger: dict[str, object]
) -> tuple[str, ...]:
    fixed = (
        "preflight-paired-s7778-cpu",
        "smoke-route-s7778-b200",
        "smoke-relative-s7778-b200",
        "calibrate-route-s7778-b200",
        "calibrate-relative-s7778-b200",
    )
    calibrations, _ = _preview_inputs_from_ledger(ledger)
    b200_preview = tuple(
        action
        for action in required_preview_actions(calibrations)
        if action.endswith("-b200")
    )
    h100_fixed = (
        "calibrate-route-s7778-h100",
        "calibrate-relative-s7778-h100",
    )
    h100_preview = tuple(
        action
        for action in required_preview_actions(calibrations)
        if action.endswith("-h100")
    )
    result = fixed + b200_preview + h100_fixed + h100_preview
    assert all(action in ledger["runs"] for action in result)
    return result


def prefull_action_result_hashes(
    ledger: dict[str, object]
) -> dict[str, str]:
    return {
        action: hashlib.sha256(
            strict_json(ledger["runs"][action]).encode("utf-8")
        ).hexdigest()
        for action in required_completed_prefull_actions(ledger)
    }


def paired_replay_certificate(ledger: dict[str, object]) -> dict[str, object]:
    result = {}
    for gpu in ("B200", "H100!"):
        route_action = registered_action(kind="calibrate", role="route", gpu=gpu)
        relative_action = registered_action(
            kind="calibrate", role="relative_centroid", gpu=gpu
        )
        route = _result_evidence(ledger, route_action)["replay"]
        relative = _result_evidence(ledger, relative_action)["replay"]
        assert relative["route_checkpoint_sha256"] == route["checkpoint_sha256"]
        assert relative["common_core_exact"] is True
        result[gpu] = {"route": copy.deepcopy(route), "relative": copy.deepcopy(relative)}
    return result


def mechanics_certificate(ledger: dict[str, object]) -> dict[str, bool]:
    relative = _result_evidence(ledger, "smoke-relative-s7778-b200")
    result = {
        "synthetic_b200_no_effect": relative["mechanics"][
            "synthetic_b200_no_effect"
        ],
        "real_b200_no_effect": relative["mechanics"]["real_b200_no_effect"],
    }
    assert result == {
        "synthetic_b200_no_effect": True,
        "real_b200_no_effect": True,
    }
    return result


def preview_certificate(ledger: dict[str, object]) -> dict[str, object]:
    calibrations, previews = _preview_inputs_from_ledger(ledger)
    validate_preview_certificate(calibrations, previews)
    return {"calibrations": calibrations, "runs": previews}


def memory_certificate(
    ledger: dict[str, object], *, limit_gib: float
) -> dict[str, float]:
    result = {"limit_gib": limit_gib}
    for gpu in ("B200", "H100!"):
        route = _result_evidence(
            ledger, registered_action(kind="calibrate", role="route", gpu=gpu)
        )
        relative = _result_evidence(
            ledger,
            registered_action(kind="calibrate", role="relative_centroid", gpu=gpu),
        )
        delta = float(relative["peak_memory_gib"]) - float(route["peak_memory_gib"])
        assert math.isfinite(delta) and delta <= limit_gib
        result[gpu] = delta
    return result


def timing_certificate(ledger: dict[str, object]) -> dict[str, float]:
    b200 = _result_evidence(ledger, "calibrate-relative-s7778-b200")
    h100 = _result_evidence(ledger, "calibrate-relative-s7778-h100")
    b200_rate = float(b200["calibration"]["median_flops_per_second"])
    h100_rate = float(h100["calibration"]["median_flops_per_second"])
    projected_b200 = float(b200["train_loop_wall_seconds"]) * (7_812 / 256)
    projected_h100 = projected_b200 * b200_rate / h100_rate
    assert all(
        math.isfinite(value) and value > 0
        for value in (b200_rate, h100_rate, projected_b200, projected_h100)
    )
    return {
        "b200_rate": b200_rate,
        "h100_rate": h100_rate,
        "projected_b200_full_seconds": projected_b200,
        "projected_h100_full_seconds": projected_h100,
    }


def validate_seed7778_launch_certificate_structure(
    certificate: dict[str, object]
) -> bool:
    assert set(certificate) == {
        "schema_version",
        "launch_lock",
        "experiment_lock",
        "sample_anchors",
        "actions",
        "paired_replay",
        "mechanics",
        "previews",
        "memory",
        "timing",
        "full_actions",
    }
    assert certificate["schema_version"] == 1
    assert certificate["launch_lock"] == load_launch_lock()
    assert certificate["sample_anchors"]["smoke_1024"]["count"] == 1_024
    assert certificate["sample_anchors"]["calibration_8192"]["count"] == 8_192
    assert certificate["mechanics"] == {
        "synthetic_b200_no_effect": True,
        "real_b200_no_effect": True,
    }
    assert float(certificate["memory"]["B200"]) <= MEMORY_DELTA_LIMIT_GIB
    assert float(certificate["memory"]["H100!"]) <= MEMORY_DELTA_LIMIT_GIB
    assert tuple(certificate["full_actions"]) == FULL_ACTIONS
    return True


def build_seed7778_launch_certificate(
    ledger: dict[str, object]
) -> tuple[dict[str, object], str]:
    certificate = {
        "schema_version": 1,
        "launch_lock": copy.deepcopy(load_launch_lock()),
        "experiment_lock": experiment_lock_from_ledger(ledger),
        "sample_anchors": copy.deepcopy(ledger["sample_anchors"]),
        "actions": prefull_action_result_hashes(ledger),
        "paired_replay": paired_replay_certificate(ledger),
        "mechanics": mechanics_certificate(ledger),
        "previews": preview_certificate(ledger),
        "memory": memory_certificate(ledger, limit_gib=MEMORY_DELTA_LIMIT_GIB),
        "timing": timing_certificate(ledger),
        "full_actions": list(FULL_ACTIONS),
    }
    validate_seed7778_launch_certificate_structure(certificate)
    digest = hashlib.sha256(strict_json(certificate).encode("utf-8")).hexdigest()
    return certificate, digest


def validate_seed7778_launch_certificate(
    certificate: dict[str, object], certificate_sha256: str
) -> bool:
    encoded = strict_json(certificate).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == certificate_sha256
    validate_seed7778_launch_certificate_structure(certificate)
    assert tuple(certificate["full_actions"]) == FULL_ACTIONS
    return True
```

The certificate includes the separate launch-lock JSON (source, bundle, manifests, configs, target, harness/test SHA, registry digest), canonical dataset/mapping, both sample anchors, all completed pre-full action-result hashes, paired checkpoint/common-core/history/shadow evidence, both hardware gate/preview decisions, memory deltas, timing projection, and `FULL_ACTIONS`. Build only when every required pre-full action passes and `launch_certificate is None`. Later calls validate byte equality; route completion may append run/eligibility evidence but cannot replace or rebuild the certificate. Both full remote calls receive the same certificate and SHA and independently restage/revalidate the canonical dataset and 8,192 anchor.

- [ ] **Step 6: Implement per-arm eligibility and publication independence**

Change `labless_eligibility(full_result)` to require the role explicitly. Route requires local full completion, exact source/caps, all probes, artifact validation, and no centroid gate. Relative requires the same plus a passed production matched-latest gate. Score remains reported-only.

Add tests:

```text
test_route_full_is_eligible_without_centroid_gate_regardless_of_score
test_route_publication_pending_transport_failure_or_rejection_does_not_block_relative_full
test_relative_gate_stop_leaves_route_submission_eligible
test_publication_state_is_not_an_action_and_never_consumes_training_retry
test_publication_mutations_do_not_change_required_next_action
```

Publication records are written only to the ignored report and downloaded Labless payloads after Labless commands. They are absent from the authoritative action ledger, `ACTION_REGISTRY`, certificate inputs, retry counts, and `_required_next_action()`.

- [ ] **Step 7: Run the focused and complete controller suites**

Run:

```powershell
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py -k "retry or infrastructure or seed7779 or sequence or preview or certificate or eligibility or publication"
python -m py_compile .superpowers/sdd/modal_molcap_paired_s7778.py .superpowers/sdd/test_modal_molcap_paired_s7778.py
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py
python .superpowers/sdd/test_modal_molcap_paired_s7778.py
```

Expected: every command exits `0`; direct unittest prints `OK`; strict JSON contains no NaN/Infinity; the frozen seed-7777 harness/test hashes remain unchanged.

---

### Task 5: Freeze Exact Identities, Obtain Independent Reviews, and Publish the Source Commit to Main

**Files:**
- Create with `apply_patch`: `.superpowers/sdd/molcap-paired-s7778-launch-lock.json`
- Create with `apply_patch`: `.superpowers/sdd/task-7-seed7778-report.md`
- Read only: every tracked source path and both operational files after lock creation

**Interfaces:**
- Consumes: final Task 1 source commit and final Task 4 ignored harness/test.
- Produces: exact immutable lock, clean reviews, GitHub main at the source commit, and authorization for Action 0.

- [ ] **Step 1: Run all local verification before computing final hashes**

Run:

```powershell
python -m pytest -q tests/test_molcap_seed7778_recovery.py
python -m pytest -q
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py
python .superpowers/sdd/test_modal_molcap_paired_s7778.py
git diff --check
git status --short --branch
git rev-list --count 06679b7b61e16b402601c694cea5851f2e7bec99..HEAD
```

Expected: tracked tests report `381 passed, 2 skipped`; both harness invocations exit `0`; diff check is empty; tracked tree is clean; revision count is `1`.

- [ ] **Step 2: Compute candidate lock material and exact operational hashes**

Run:

```powershell
$env:PYTHONUTF8 = "1"
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action lock-material
Get-FileHash -Algorithm SHA256 .superpowers/sdd/modal_molcap_paired_s7778.py
Get-FileHash -Algorithm SHA256 .superpowers/sdd/test_modal_molcap_paired_s7778.py
git rev-parse HEAD
```

Expected: `lock-material` performs no remote function call and prints strict JSON for the exact source commit, bundle/manifest/Labless/locked/config/target/registry identities. Use `apply_patch` to create the launch-lock JSON with those exact printed values plus the two exact file hashes. Do not edit the harness or test afterward.

- [ ] **Step 3: Revalidate the completed lock and `describe` surface**

Run:

```powershell
$env:PYTHONUTF8 = "1"
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action describe
python -m pytest -q .superpowers/sdd/test_modal_molcap_paired_s7778.py
Get-FileHash -Algorithm SHA256 .superpowers/sdd/modal_molcap_paired_s7778.py
Get-FileHash -Algorithm SHA256 .superpowers/sdd/test_modal_molcap_paired_s7778.py
git status --short --branch
```

Expected: `describe` exactly matches the launch lock, app `nanopath-molcap-paired-s7778`, eleven registry entries, new ledger path, `fresh_full_route=true`, `full_fallback=false`, and no external action. File hashes equal the launch-lock values and tracked status is clean.

- [ ] **Step 4: Obtain two independent read-only review gates**

Dispatch one reviewer for scientific/action integrity and one for source/path/ledger durability. Give each the exact source commit and harness/test SHA values. Require both to inspect the exact files, run or audit the contract suite, and return no P0-P2 findings. Any semantic fix invalidates both file hashes and repeats Steps 1-4 from the start; do not waive a finding.

- [ ] **Step 5: Verify the public push surface and fast-forward GitHub main**

Run:

```powershell
git diff --name-status --no-renames 06679b7b61e16b402601c694cea5851f2e7bec99..HEAD
git log --oneline origin/main..HEAD
git grep -n -I -E "(BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})" HEAD -- configs docs/superpowers tests
git fetch origin main
git merge-base --is-ancestor origin/main HEAD
git push origin HEAD:main
git ls-remote --heads origin main
```

Expected: the baseline diff is exactly the five approved additions; secret scan prints nothing; ancestry check exits `0`; the remote `refs/heads/main` SHA exactly equals `git rev-parse HEAD`. This direct main push is explicitly user-approved. No tracked file changes after this point.

- [ ] **Step 6: Start the ignored durable report**

Use `apply_patch` to create `.superpowers/sdd/task-7-seed7778-report.md` containing the exact source/harness/test/launch-lock hashes, review verdicts, remote-main verification, zero external actions, and the new authoritative ledger path. This report is ignored and never enters the candidate source.

---

### Task 6: Execute and Verify the Fresh CPU Preflight

**Files:**
- Update with `apply_patch`: `.superpowers/sdd/task-7-seed7778-report.md`
- Runtime: `.superpowers/sdd/molcap-paired-s7778-evidence.json`
- Runtime: `/persistent/experiments/readout-local-context/matched-latest-s7778/evidence.json`
- Runtime logs: `.superpowers/sdd/seed7778-logs/00-preflight-paired-s7778-cpu.*.log`

**Interfaces:**
- Consumes: exact reviewed launch lock and empty new ledger namespace.
- Produces: Action-0 experiment lock with canonical dataset manifest and seed-7778 mapping.

- [ ] **Step 1: Recheck hashes and prove the seed-7777 ledger is untouched**

Run `Get-FileHash` for both operational files, `git rev-parse HEAD`, `git status --porcelain=v1 --untracked-files=all`, `describe`, and `inspect-ledger`. Require exact launch-lock values and an empty seed-7778 ledger. Read the seed-7777 ledger only to record that its completed-action/stop state is unchanged; never write it.

- [ ] **Step 2: Launch exactly Action 0**

Run:

```powershell
$env:PYTHONUTF8 = "1"
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action preflight-paired-s7778-cpu
```

For long execution under Codex, start the same command with `Start-Process -WindowStyle Hidden -PassThru`, redirect stdout/stderr to the two Action-0 log files, and poll without a blocking sleep longer than 60 seconds.

- [ ] **Step 3: Validate the authoritative result before any GPU**

Run:

```powershell
$env:PYTHONUTF8 = "1"
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action inspect-ledger
```

Require completed actions exactly `[preflight-paired-s7778-cpu]`; no retry/stop/incomplete/certificate; exact source/config/target/locked/launch-lock identities; 200 Parquet shards; target `11,428 x 384`, unit norm and 100% coverage; seed-7778 mapping patient/slide counts and newly computed digest; canonical source dataset file count/bytes/manifest; and durable local/volume ledger equality. Record app ID, timings, hashes, and log hashes in the report.

- [ ] **Step 4: Apply the frozen stop rule**

If Action 0 records source/data/code/path mismatch, stop with `required_seed=7779`. If it records an explicit retryable infrastructure failure, archive partial state and retry the identical action only while total attempts are at most three. Do not launch Action 1 until the exact passed ledger exists.

---

### Task 7: Execute Paired Smokes, Calibrations, Conditional Previews, and Seal the Certificate

**Files:**
- Update with `apply_patch`: `.superpowers/sdd/task-7-seed7778-report.md`
- Runtime logs: `.superpowers/sdd/seed7778-logs/01-*` through `08-*`
- Runtime outputs under `/persistent/experiments/readout-local-context/matched-latest-s7778/`

**Interfaces:**
- Consumes: Action-0 experiment lock and exact operational hashes.
- Produces: paired mechanics, anchors, same-hardware replay, relative gate previews, memory/timing evidence, and one immutable launch certificate.

- [ ] **Step 1: Run the two B200 mechanics smokes serially**

Run exactly:

```powershell
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action smoke-route-s7778-b200
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action smoke-relative-s7778-b200
```

After each action, run `inspect-ledger` and recheck operational hashes. Route atomically anchors the exact first-1,024 `<i8` digest; relative must match it. Require identical normalized common core, zero optimized loss/gradient/RNG change from the audit shadow, exact primary proposal, valid relative history/shadow, route absence, and fresh staged-manifest equality.

- [ ] **Step 2: Run the paired B200 calibrations**

Run exactly:

```powershell
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action calibrate-route-s7778-b200
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action calibrate-relative-s7778-b200
```

Route must reproduce the 1,024 smoke prefix and atomically anchor the first 8,192 values. Relative must match 8,192, have an exact normalized common core against the B200 route checkpoint, valid separate history/shadow identities, and a valid gate preview.

- [ ] **Step 3: Run the B200 preview only if preregistered maturity requires it**

Inspect Action 4 maturity. If it is below `0.95`, run exactly:

```powershell
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action preview-relative-s7778-b200
```

If maturity is at least `0.95`, require the state machine to reject this action. A required preview must match the 8,192 anchor and pass the unchanged gate; there is no second extension.

- [ ] **Step 4: Run the paired exact-H100 calibrations**

Run exactly:

```powershell
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action calibrate-route-s7778-h100
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action calibrate-relative-s7778-h100
```

Require exact-H100 allocation, 8,192 anchor equality across hardware, exact same-hardware common core, valid route/relative state separation, gate preview, and staged identity.

- [ ] **Step 5: Run the H100 preview only if preregistered maturity requires it**

If H100 relative maturity is below `0.95`, run exactly:

```powershell
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action preview-relative-s7778-h100
```

Otherwise require rejection. The B200 decision cannot force an H100 preview or vice versa.

- [ ] **Step 6: Verify the sealed certificate before any full run**

Run `inspect-ledger` and require: all exact prerequisite actions; only required previews; both gate previews passed; B200/H100 relative-minus-route memory each `<= 0.5 GiB`; timing values finite; H100-equivalent projection reported; mechanics true; route absence and relative history/shadow valid; both anchors exact; one non-null certificate and SHA; no stop/incomplete field. Recompute `SHA256(strict_json(certificate))` locally and match the ledger. Recheck the two operational file hashes and update the report.

Any identity/path/mechanics/replay/certificate mismatch records seed `7779` and stops. A gate failure stops the relative experiment before fulls. A retry occurs only for a declared infrastructure failure within the three-attempt cap.

---

### Task 8: Execute Both Full Runs, Submit Every Completed Arm, and Report the Paired Result

**Files:**
- Update with `apply_patch`: `.superpowers/sdd/task-7-seed7778-report.md`
- Runtime logs: `.superpowers/sdd/seed7778-logs/09-full-route-s7778-b200.*.log`
- Runtime logs: `.superpowers/sdd/seed7778-logs/10-full-relative-s7778-b200.*.log`
- Local publication staging outside the repository: `$env:TEMP/nanopath-paired-s7778/`

**Interfaces:**
- Consumes: immutable launch certificate, pushed main source, and locked Labless client.
- Produces: completed route and eligible relative endpoint artifacts, public submissions, paired mechanism verdict, and final durable report.

- [ ] **Step 1: Launch and validate the fresh route full**

Run:

```powershell
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action full-route-s7778-b200
```

Require from-scratch `resume: null`, exact certificate/hash, fresh canonical restage, 8,192 anchor, `7,812` steps, `999,936` presentations, max-sample stop, all 12 probes, exact source/config/target/mapping/locked identities, finite metrics, and route absence of centroid gate/history/shadow. Store route eligibility in the ledger. Do not inspect its score as a launch condition.

- [ ] **Step 2: Launch the fresh relative full immediately after local route validation**

Run regardless of route score or publication state:

```powershell
modal run .superpowers/sdd/modal_molcap_paired_s7778.py --action full-relative-s7778-b200
```

Require the identical sealed certificate, fresh restage, 8,192 anchor, and a production matched-latest gate immediately before first nonzero MolCap scale. If the gate fails, preserve the pre-supervision stop, do not submit the incomplete relative arm, and continue with route publication. If it passes, require `7,812` steps, `999,936` presentations, all 12 probes, discarded latest shadow, valid final EMA history, and per-arm eligibility.

- [ ] **Step 3: Download only lightweight completed publication artifacts**

Create route/relative local staging directories outside the repo. For each completed arm, use `modal volume get nanopath-readout-local-context` to retrieve only `summary.json`, `metrics.jsonl`, `modal_result.json`, and `labless_source/`; retrieve `molcap_centroid_ramp_gate.json` for relative. Do not download or submit checkpoints, targets, raw data, host paths, or logs.

The remote roots are:

```text
/experiments/readout-local-context/matched-latest-s7778/full/molcap-probe-route-s7778
/experiments/readout-local-context/matched-latest-s7778/full/molcap-ema-rel-s7778
```

- [ ] **Step 4: Dry-run every completed Labless payload**

Immediately before dry-run and again before each real post, require `git ls-remote --heads origin main` to equal the locked `SOURCE_COMMIT`. If the remote advanced, do not force-push, retrain, or change the candidate; record a publication/source-state issue and resolve it independently of the already completed training actions.

For route:

```powershell
python .\labless\submit_to_labless.py "output_dir=$routeDir" "run_name=molcap-route-s7778" "notes=Fresh paired seed-7778 probe-CLS route control for the matched-latest EMA centroid experiment." "review_config=configs/molcap-probe-route-s7778.yaml" "hardware=NVIDIA B200" "dry_run=true"
```

For relative, only if completed:

```powershell
python .\labless\submit_to_labless.py "output_dir=$relativeDir" "run_name=molcap-rel-s7778" "notes=Fresh paired seed-7778 matched-latest EMA patient-centroid MolCap arm with the frozen pre-supervision relative gate." "review_config=configs/molcap-ema-relative-s7778.yaml" "hardware=NVIDIA B200" "dry_run=true"
```

Require exit `0`, zero validation/locked-path/policy errors, exact source commit on GitHub main, full caps, and the metric exactly matching the completed summary. Run labels are at most 20 characters.

- [ ] **Step 5: Submit every completed arm independently**

Repeat each passing dry-run command without `dry_run=true`, complete GitHub device authentication, and capture stdout/stderr. A transport/service failure retries the immutable completed artifact independently; a deterministic policy rejection is recorded and never changes source, metrics, or the other arm. Require server response JSON and preserve its run/submission IDs. Route publication must proceed even if the relative arm gate-stopped.

- [ ] **Step 6: Verify the public records**

Fetch:

```powershell
$public = Invoke-RestMethod "https://api.labless.dev/api/nano-projects/nanopath/experiment-log?limit=100"
```

Find exact titles `molcap-route-s7778` and, when completed, `molcap-rel-s7778`; require public metric equality, source identity, seed `7778`, `unvalidated` or later validation state, and saved run IDs. Follow each row's `api_url` and record the detail URL and main/review state. Public response state is reporting-only.

- [ ] **Step 7: Compute the preregistered paired mechanism verdict**

Use the completed summaries to compute relative minus route for progression, mutation, survival, linear, kNN, few-shot, and overall score. Support requires all three molecular/slide metrics positive, their unweighted mean at least `0.003`, and each of linear/kNN/few-shot greater than `-0.003`. Report the historical `0.6719107210` threshold and public frontier descriptively only; do not declare formal BSC promotion.

- [ ] **Step 8: Finalize and independently verify the durable report**

The ignored report records exact source/harness/test/lock hashes, every Modal app/action ID and attempt, ledger/certificate hashes, staging/mapping/anchor identities, mechanics/replay/gates, memory/timing and H100 projection, full steps/probes/metrics, endpoint deltas, mechanism verdict, Labless IDs/URLs/states, all stops, and the fact that seed-7777 evidence remained unchanged. Run final exact-SHA and source-tree checks plus a read-only review. Do not create another tracked commit or modify GitHub main after the locked source push.
