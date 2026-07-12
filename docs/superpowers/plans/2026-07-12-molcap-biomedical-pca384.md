# MolCap Biomedical PCA-384 Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, gate, train, probe, time-calibrate, and submit one width- and head-capacity-controlled seed-7777 MolCap experiment using a deterministic 768-to-384 PCA projection of pinned S-PubMedBERT caption embeddings.

**Architecture:** Extend the audited re-embedding helper with an explicit `pca384` variant while preserving `raw768` as the default compatibility path. The PCA variant mean-centers normalized 768-D embeddings, fits deterministic top-384 PCA, normalizes projected rows, applies the unchanged isotropy transform, and publishes only after the new variance-retention gate and every existing geometry/coverage gate pass. Training uses an otherwise byte-matched 384-D config; an ignored Modal harness builds the target on CPU, runs B200/H100 calibration, executes the public full run on B200, and records an H100-equivalent time projection.

**Tech Stack:** Python 3.12, NumPy, sentence-transformers, PyTorch 2.8/CUDA 12.9, pytest, YAML, Modal CPU/B200/H100/H200, Labless submission wrapper.

## Global Constraints

- Approved specification: `docs/superpowers/specs/2026-07-12-molcap-biomedical-pca384-design.md`.
- Branch baseline: `b5869cba494e5b57233afa45fbde4efa0561577d` on `codex/molcap-biomed-pca384-s7777`.
- Canonical target SHA-256: `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`.
- Canonical rows: exactly 11,428; FINO patients: exactly 9,389 with full coverage.
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Biomedical encoder: `pritamdeka/S-PubMedBert-MS-MARCO@96786c7024f95c5aac7f2b9a18086c7b97b23036`.
- PCA input/output: exactly 768 to 384; retained variance must be at least `0.99`.
- PCA projected rows are L2-normalized before the unchanged `0.05` floor / `-0.1` isotropy transform.
- Existing absolute and MiniLM-relative target gates remain unchanged.
- Default/no-variant behavior remains the historical `raw768` path.
- The failed 768-D artifacts, config, report, and result document remain untouched.
- The PCA config differs from `molcap-text-s7777.yaml` only at `project.name`, `project.output_dir`, and `molcap.targets`.
- Seed 7777, split, batch 128, views, objective weights, ramp, FINO, and probes do not change.
- `model.py`, `dataloader.py`, `train.py`, `probe.py`, and `benchmarking/` must remain untouched.
- Any target-gate failure stops before GPU training; thresholds are not revised.
- Any completed full run is submitted to Labless regardless of score.
- Public execution prefers one B200; B300/`B200+` are forbidden under CUDA 12.9. Infrastructure fallback order is H200 then exact H100.
- Preprocessing is excluded from training time. Smoke and calibration runs are never submitted.

---

### Task 1: Deterministic PCA-384 Projection and Variant-Safe Target Builder

**Files:**
- Modify: `reembed_molcap_targets.py`
- Modify: `tests/test_molcap_reembed.py`
- Reuse unchanged: `build_molcap_targets.py:isotropize,save_target_bank`

**Interfaces:**
- Produces: `canonicalize_component_signs(components) -> np.ndarray`.
- Produces: `pca_project_unit(raw, n_components=384) -> tuple[np.ndarray, dict[str, object]]`.
- Extends: `validate_candidate(reference, candidate, patient_ids, fino_ids, expected_fino_count=FINO_PATIENT_COUNT, expected_width=768) -> dict[str, object]`.
- Extends: `persist_validation_failure(error, source_sha, patient_ids, mode, output, report, model_payload, artifact_width, artifact_mode, extra_payload=None)`.
- Extends: `build_reembedded_bank(source, output, report, fino_path, minilm_binding, biomedical_binding, expected_source_sha, device="cpu", variant="raw768")`.
- CLI adds optional `variant=pca384`; omitting it preserves `raw768`.

- [ ] **Step 1: Write failing PCA math and sign-canonicalization tests**

Add focused tests before changing production code:

```python
def test_canonicalize_component_signs_uses_lowest_largest_loading():
    components = np.array([
        [-0.5, 0.5, 0.1],
        [0.1, -0.8, 0.2],
    ], dtype=np.float64)
    fixed = reembed.canonicalize_component_signs(components)
    np.testing.assert_array_equal(fixed[0], -components[0])
    np.testing.assert_array_equal(fixed[1], -components[1])
    pivots = np.argmax(np.abs(fixed), axis=1)
    assert np.all(fixed[np.arange(len(fixed)), pivots] > 0)


def test_pca_project_unit_is_deterministic_unit_norm_and_variance_audited():
    raw = np.random.default_rng(7).normal(size=(64, 8))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    first, first_audit = reembed.pca_project_unit(raw, n_components=4)
    second, second_audit = reembed.pca_project_unit(raw, n_components=4)
    assert first.shape == (64, 4)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(first, second)
    assert first_audit == second_audit
    assert len(first_audit["eigenvalues_descending"]) == 8
    assert 0.0 <= first_audit["retained_variance_fraction"] <= 1.0
    assert first_audit["component_sha256"] == second_audit["component_sha256"]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_molcap_reembed.py::test_canonicalize_component_signs_uses_lowest_largest_loading tests/test_molcap_reembed.py::test_pca_project_unit_is_deterministic_unit_norm_and_variance_audited -q
```

Expected: failures because both PCA functions are absent.

- [ ] **Step 3: Implement deterministic PCA primitives**

Add frozen constants and helpers:

```python
RAW768 = "raw768"
PCA384 = "pca384"
PCA_DIM = 384
PCA_MIN_VARIANCE = 0.99


def canonicalize_component_signs(components):
    fixed = np.asarray(components, dtype=np.float64).copy()
    pivots = np.argmax(np.abs(fixed), axis=1)
    signs = np.where(fixed[np.arange(len(fixed)), pivots] < 0, -1.0, 1.0)
    return fixed * signs[:, None]


def array_sha256(array):
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def pca_project_unit(raw, n_components=PCA_DIM):
    values = np.asarray(raw, dtype=np.float64)
    assert values.ndim == 2 and 0 < n_components < values.shape[1]
    assert np.isfinite(values).all(), "PCA input finite gate failed"
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.lexsort((np.arange(len(eigenvalues)), -eigenvalues))
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = canonicalize_component_signs(eigenvectors[:, order[:n_components]].T)
    scores = centered @ components.T
    norms = np.linalg.norm(scores, axis=1, keepdims=True)
    assert (norms > 0).all(), "PCA projection zero-row gate failed"
    projected = (scores / norms).astype(np.float32)
    total = float(eigenvalues.sum())
    retained = float(eigenvalues[:n_components].sum())
    audit = {
        "fit_rows": len(values),
        "input_width": values.shape[1],
        "output_width": n_components,
        "solver": "numpy.linalg.eigh",
        "covariance_denominator": "n-1",
        "sign_rule": "lowest-index largest-absolute loading positive",
        "eigenvalues_descending": eigenvalues.tolist(),
        "eigenvalues_sha256": array_sha256(eigenvalues),
        "mean_sha256": array_sha256(mean),
        "component_sha256": array_sha256(components),
        "retained_variance": retained,
        "total_variance": total,
        "retained_variance_fraction": retained / total,
        "discarded_energy_fraction": 1.0 - retained / total,
        "eigenvalue_384": float(eigenvalues[n_components - 1]),
        "eigenvalue_385": float(eigenvalues[n_components]),
        "eigengap_384_385": float(eigenvalues[n_components - 1] - eigenvalues[n_components]),
    }
    return projected, audit
```

- [ ] **Step 4: Run the PCA tests and verify GREEN**

Run the Step 2 command. Expected: both pass with no warnings.

- [ ] **Step 5: Write failing variant, order, gate, and compatibility tests**

Add tests proving:

```python
def test_default_and_explicit_raw768_builds_are_byte_identical(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    default_output, default_report = tmp_path / "default.npz", tmp_path / "default.json"
    explicit_output, explicit_report = tmp_path / "explicit.npz", tmp_path / "explicit.json"
    default_payload = reembed.build_reembedded_bank(
        case.source, default_output, default_report, case.fino,
        minilm, biomedical, case.source_sha,
    )
    explicit_payload = reembed.build_reembedded_bank(
        case.source, explicit_output, explicit_report, case.fino,
        minilm, biomedical, case.source_sha, variant=reembed.RAW768,
    )
    assert default_output.read_bytes() == explicit_output.read_bytes()
    assert default_report.read_bytes() == explicit_report.read_bytes()
    assert default_payload == explicit_payload


def build_pca_fixture(case, tmp_path, monkeypatch, output, report):
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    monkeypatch.setattr(reembed, "PCA_MIN_VARIANCE", 0.0)
    monkeypatch.setattr(
        reembed,
        "validate_candidate",
        lambda *args, **kwargs: {"coverage_count": case.rows, "coverage_total": case.rows},
    )
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    return reembed.build_reembedded_bank(
        case.source, output, report, case.fino,
        minilm, biomedical, case.source_sha, variant=reembed.PCA384,
    )


def test_pca384_build_projects_normalizes_then_isotropizes(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "pca.npz", tmp_path / "pca.json"
    payload = build_pca_fixture(case, tmp_path, monkeypatch, output, report)
    with np.load(output, allow_pickle=False) as bank:
        assert bank["targets"].shape == (case.rows, 4)
        assert bank["mode"].item() == "biomedical-pca384"
    assert payload["artifact"]["width"] == 4
    assert payload["pca"]["output_width"] == 4
    assert payload["models"]["biomedical"]["post_pca_geometry"]["width"] == 4


def test_pca_variance_failure_clears_stale_384_target_and_reports(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    output, report = tmp_path / "stale-pca.npz", tmp_path / "stale-pca.json"
    output.write_bytes(b"stale")
    monkeypatch.setitem(reembed.VARIANT_SPECS[reembed.PCA384], "target_width", 4)
    monkeypatch.setattr(reembed, "PCA_MIN_VARIANCE", 1.01)
    minilm = fake_binding(case, tmp_path, case.minilm_raw, MINILM_MODEL, MINILM_REVISION)
    biomedical = fake_binding(case, tmp_path, case.biomedical_raw, BIOMED_MODEL, BIOMED_REVISION)
    with pytest.raises(reembed.ValidationGateError, match="PCA variance retention"):
        reembed.build_reembedded_bank(
            case.source, output, report, case.fino,
            minilm, biomedical, case.source_sha, variant=reembed.PCA384,
        )
    assert not output.exists()
    failure = json.loads(report.read_text())
    assert failure["status"] == "failed"
    assert failure["artifact"]["width"] == 4
    assert failure["gate_error"]["gate"] == "PCA variance retention"
```

Add these explicit edge tests:

```python
def test_cli_accepts_only_pca384_variant(tmp_path, monkeypatch):
    build_calls = []

    def fake_snapshot_download(repo_id, revision, local_files_only):
        assert local_files_only is True
        return str(fake_snapshot_path(tmp_path, repo_id, revision))

    def fake_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setattr(reembed, "build_reembedded_bank", fake_build)
    base_argv = [
        f"source={tmp_path / 'source.npz'}",
        f"output={tmp_path / 'output.npz'}",
        f"report={tmp_path / 'report.json'}",
        f"fino={tmp_path / 'fino.json'}",
        "device=cpu",
    ]
    argv = base_argv + ["variant=pca384"]
    assert reembed.main(argv) == {"ok": True}
    assert build_calls[0][1]["variant"] == reembed.PCA384
    with pytest.raises(AssertionError, match="variant"):
        reembed.main(base_argv + ["variant=unknown"])


def test_pca_zero_projection_row_is_rejected():
    raw = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (8, 1))
    with pytest.raises(AssertionError, match="zero-row"):
        reembed.pca_project_unit(raw, n_components=2)


def test_pca_variant_orders_projection_before_biomedical_isotropy(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    events = []
    original_isotropize = reembed.isotropize
    original_pca = reembed.pca_project_unit

    def record_isotropize(values):
        events.append(("isotropize", values.shape[1]))
        return original_isotropize(values)

    def record_pca(values, n_components):
        events.append(("pca", values.shape[1]))
        return original_pca(values, n_components)

    monkeypatch.setattr(reembed, "isotropize", record_isotropize)
    monkeypatch.setattr(reembed, "pca_project_unit", record_pca)
    build_pca_fixture(case, tmp_path, monkeypatch, tmp_path / "order.npz", tmp_path / "order.json")
    assert events == [("isotropize", case.dim), ("pca", case.dim), ("isotropize", 4)]


def test_pca_success_artifacts_are_byte_deterministic(tmp_path, monkeypatch):
    case = make_reembed_case(tmp_path, monkeypatch)
    first, first_report = tmp_path / "first-pca.npz", tmp_path / "first-pca.json"
    second, second_report = tmp_path / "second-pca.npz", tmp_path / "second-pca.json"
    build_pca_fixture(case, tmp_path, monkeypatch, first, first_report)
    build_pca_fixture(case, tmp_path, monkeypatch, second, second_report)
    assert first.read_bytes() == second.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()
```

Extend the existing strict-JSON failure test so its injected `extra_payload`
contains `{"pca": {"retained_variance_fraction": np.nan}}`; assert the JSON
contains `null` at that field and records
`pca.retained_variance_fraction: nan` in `non_finite_values`.

```python
payload = json.loads(report.read_text())
assert payload["pca"]["retained_variance_fraction"] is None
assert payload["non_finite_values"]["pca.retained_variance_fraction"] == "nan"
```

- [ ] **Step 6: Run expanded tests and verify RED**

Run:

```powershell
python -m pytest tests/test_molcap_reembed.py -q
```

Expected: new variant tests fail because `build_reembedded_bank` and the CLI do
not yet support `pca384` or a dynamic artifact width.

- [ ] **Step 7: Implement the variant without changing raw768 behavior**

Implement these exact rules:

```python
VARIANT_SPECS = {
    RAW768: {"target_width": 768, "artifact_mode": "biomedical"},
    PCA384: {"target_width": 384, "artifact_mode": "biomedical-pca384"},
}
```

- Validate the variant before encoder loading.
- Keep biomedical encoding at raw width 768 for both variants.
- For `pca384`, call `pca_project_unit` after raw-finite validation and before
  `isotropize`.
- Require `retained_variance_fraction >= 0.99` before isotropy.
- Report pre-PCA, normalized post-PCA, and corrected geometry.
- Pass `expected_width` into `validate_candidate`.
- Pass target width/mode and PCA audit into every success/failure report.
- Save the projected/isotropized 384-D target for `pca384`.
- Preserve all raw768 payload keys and values so default/explicit output bytes
  remain identical.
- Accept CLI keys `{source, output, report, fino, device}` plus optional
  `variant`; default to `raw768` and reject every other key/value.

- [ ] **Step 8: Run focused and full Task 1 verification**

Run:

```powershell
python -m pytest tests/test_molcap_reembed.py -q
python -m pytest -q
python -m py_compile reembed_molcap_targets.py build_molcap_targets.py
git diff --check
git diff --name-only b5869cb..HEAD -- model.py dataloader.py train.py probe.py benchmarking/
```

Expected: all tests pass; compile/diff checks exit 0; the locked-path command
prints nothing.

- [ ] **Step 9: Commit Task 1**

```powershell
git add -f reembed_molcap_targets.py tests/test_molcap_reembed.py
git commit -m "feat: add gated MolCap PCA-384 targets"
```

---

### Task 2: Strict PCA-384 Config and Real 384-D Integration

**Files:**
- Create: `configs/molcap-biomed-pca384-s7777.yaml`
- Modify: `tests/test_molcap_config.py`
- Modify: `tests/test_molcap_integration.py`

**Interfaces:**
- Consumes: `configs/molcap-text-s7777.yaml` and the existing target-bank/head/loss/checkpoint APIs.
- Produces: a locked training config with exactly three changed leaves and a tested `biomedical-pca384` bank mode.

- [ ] **Step 1: Write the failing exact-config-diff test**

```python
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
```

- [ ] **Step 2: Run the config test and verify RED**

Run:

```powershell
python -m pytest tests/test_molcap_config.py::test_pca384_config_controls_width_and_head_capacity -q
```

Expected: missing-config failure.

- [ ] **Step 3: Create the config with only permitted changes**

Copy `configs/molcap-text-s7777.yaml`, then change exactly:

```yaml
project:
  name: molcap-biomed-pca384-s7777
  output_dir: /data/$USER/nanopath/molcap/molcap-biomed-pca384-s7777
molcap:
  targets: /data/$USER/nanopath/molcap_biomed_pca384.npz
  target_dim: 384
```

- [ ] **Step 4: Verify the config test passes**

Run all config tests. Expected: pass, including both historical biomedical
config tests and the new three-leaf diff.

- [ ] **Step 5: Add an explicit `biomedical-pca384` integration regression**

Use the real deterministic bank writer and existing training APIs:

```python
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
```

- [ ] **Step 6: Run the integration regression**

```powershell
python -m pytest tests/test_molcap_integration.py::test_pca384_bank_forward_backward_and_checkpoint -q
```

Expected: pass because the production path is already dimension- and
mode-agnostic. If it fails in production code, stop and review the approved
architecture rather than modifying `model.py`, `dataloader.py`, or `train.py`.

- [ ] **Step 7: Run Task 2 and full verification**

```powershell
python -m pytest tests/test_molcap_config.py tests/test_molcap_integration.py tests/test_molcap_training.py -q
python -m pytest -q
python -m py_compile reembed_molcap_targets.py dataloader.py model.py train.py
git diff --check
git diff --name-only b5869cb..HEAD -- model.py dataloader.py train.py probe.py benchmarking/
```

Expected: all pass; locked-path output is empty.

- [ ] **Step 8: Commit Task 2**

```powershell
git add configs/molcap-biomed-pca384-s7777.yaml tests/test_molcap_config.py tests/test_molcap_integration.py
git commit -m "experiment: add paired MolCap PCA-384 arm"
```

---

### Task 3: Target Build, Independent Audit, B200 Smoke, and Hardware Calibration

**Files:**
- Create locally but keep ignored: `.superpowers/sdd/modal_molcap_biomed_pca384.py`
- Create locally but keep ignored: `.superpowers/sdd/pca384-target-audit.json`
- Create locally but keep ignored: `.superpowers/sdd/pca384-hardware-calibration.json`
- Never modify: the historical 768-D artifacts or run directories.

**Interfaces:**
- Modal CPU action `build-target` produces the PCA target/report and an independent rebuild.
- Modal B200 action `smoke-b200` proves compatibility and gradient diagnostics.
- Modal B200/H100 actions `calibrate-b200` and `calibrate-h100` produce comparable 32,768-sample summaries.
- Optional infrastructure actions `smoke-h200`, `full-h200`, and `full-h100` implement only the approved fallback order.

- [ ] **Step 1: Build the ignored execution harness**

Adapt the proven biomedical harness with distinct constants:

```python
TARGET_OUTPUT = DATA_ROOT / "molcap_biomed_pca384.npz"
TARGET_REPORT = DATA_ROOT / "molcap_biomed_pca384.geometry.json"
RUN_NAME = "molcap-biomed-pca384-s7777"
CONFIG_NAME = "molcap-biomed-pca384-s7777.yaml"
CALIBRATION_SAMPLES = 32_768
```

The target command must include `variant=pca384`. The CPU target function uses
16 CPUs and 131,072 MiB memory with no GPU. Define distinct GPU functions:

```python
@app.function(
    gpu="B200",
    volumes={VOLUME_ROOT_PATH: volume},
    cpu=16,
    memory=131072,
    timeout=4 * 60 * 60,
)
def execute_b200(action):
    assert action in {"smoke-b200", "calibrate-b200", "full-b200"}
    volume.reload()
    _prepare_workspace()
    return _run_training(action)

@app.function(
    gpu="H100!",
    volumes={VOLUME_ROOT_PATH: volume},
    cpu=16,
    memory=131072,
    timeout=4 * 60 * 60,
)
def execute_h100(action):
    assert action in {"calibrate-h100", "full-h100"}
    volume.reload()
    _prepare_workspace()
    return _run_training(action)

@app.function(
    gpu="H200",
    volumes={VOLUME_ROOT_PATH: volume},
    cpu=16,
    memory=131072,
    timeout=4 * 60 * 60,
)
def execute_h200(action):
    assert action in {"smoke-h200", "full-h200"}
    volume.reload()
    _prepare_workspace()
    return _run_training(action)
```

The local entrypoint routes only the declared actions; it exposes no submit
action.

```python
@app.function(
    volumes={VOLUME_ROOT_PATH: volume},
    cpu=16,
    memory=131072,
    timeout=30 * 60,
)
def build_target():
    volume.reload()
    _prepare_workspace()
    return _build_target()


@app.local_entrypoint()
def main(action="build-target"):
    if action == "build-target":
        result = build_target.remote()
    elif action in {"smoke-b200", "calibrate-b200", "full-b200"}:
        result = execute_b200.remote(action)
    elif action in {"calibrate-h100", "full-h100"}:
        result = execute_h100.remote(action)
    elif action in {"smoke-h200", "full-h200"}:
        result = execute_h200.remote(action)
    else:
        raise ValueError(f"unknown action: {action}")
    print(json.dumps(result, indent=2, sort_keys=True))
```

- [ ] **Step 2: Freeze mode-specific config rewrites**

- `smoke-b200`: batch 8, 1,024 samples, two local views, probes off, MolCap
  diagnostics on, ramp `[0.0, 0.25]`.
- `calibrate-b200` / `calibrate-h100`: batch 128, 32,768 samples, two global
  and eight local views, probes off, diagnostics off, original ramp fractions,
  no checkpoint/eval during the short run.
- `full-b200` / fallbacks: only volume paths and output directory change from
  the committed config.

Every GPU result records:

```python
{
    "gpu_name": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    "train_loop_wall_seconds": summary["train_loop_wall_seconds"],
    "flops_per_sec": summary["flops_per_sec"],
    "visible_patches_per_sec": summary["visible_patches_per_sec"],
}
```

- [ ] **Step 3: Verify the harness locally**

```powershell
python -m py_compile .superpowers/sdd/modal_molcap_biomed_pca384.py
$env:PYTHONUTF8='1'
modal run .superpowers/sdd/modal_molcap_biomed_pca384.py --help
git check-ignore -v .superpowers/sdd/modal_molcap_biomed_pca384.py
git status --short
```

Expected: compilation/help exit 0, the harness is ignored, and the tracked
worktree is clean.

- [ ] **Step 4: Run the CPU target build**

```powershell
$env:PYTHONUTF8='1'
modal run .superpowers/sdd/modal_molcap_biomed_pca384.py --action build-target
```

Require before continuing:

- `status=passed`, `published=true`;
- retained PCA variance at least `0.99`;
- deterministic target/report rebuild hashes;
- exact `11428 x 384`, finite unit vectors;
- exact canonical identity/captions and MiniLM replay;
- every unchanged geometry gate passes;
- 9,389/9,389 FINO coverage.

If any gate fails, stop. Retrieve the failed report, write the durable negative
result, do not run Tasks 3.6 onward, and do not submit.

- [ ] **Step 5: Retrieve and independently audit target/report**

Download the PCA target, PCA report, canonical MiniLM bank, and FINO metadata.
Independently recompute:

- file hashes and non-pickled keys;
- patient/caption elementwise equality;
- target shape, finiteness, norms, coverage;
- PCA retained variance from the reported eigenvalues;
- component/eigenvalue/mean hash formats;
- pre/post-PCA and corrected geometry;
- every threshold comparison.

Write the exact independent audit to
`.superpowers/sdd/pca384-target-audit.json`.

- [ ] **Step 6: Run CPU integration against the real PCA target**

Load the downloaded target through `load_molcap_bank(target_dim=384)`, perform a
real `MolCapHead(8, 384)` forward/backward and checkpoint restore, and require
finite nonzero trunk/head gradients and 100% fixture coverage.

- [ ] **Step 7: Run the B200 compatibility smoke**

```powershell
$env:PYTHONUTF8='1'
modal run .superpowers/sdd/modal_molcap_biomed_pca384.py --action smoke-b200
```

Require 1,024 samples, finite active loss, finite gradient cosine/norm ratio,
100% target coverage, checkpoint/summary persistence, no probe, and recorded
B200/CUDA identity. An infrastructure-only B200 failure activates H200 then
H100 fallback without changing the experiment.

- [ ] **Step 8: Run paired current-code calibration**

```powershell
modal run .superpowers/sdd/modal_molcap_biomed_pca384.py --action calibrate-b200
modal run .superpowers/sdd/modal_molcap_biomed_pca384.py --action calibrate-h100
```

From each `metrics.jsonl`, discard entries before step 60 and compute the median
logged FLOP/s from step 60 onward. Record `F_B200`, `F_H100`, their ratio, train
loop seconds, patches/s, GPU identity, memory, and target coverage in
`.superpowers/sdd/pca384-hardware-calibration.json`.

- [ ] **Step 9: Re-run the complete local gate before full compute**

```powershell
python -m pytest -q
python -m py_compile reembed_molcap_targets.py build_molcap_targets.py dataloader.py model.py train.py .superpowers/sdd/modal_molcap_biomed_pca384.py
git diff --check
git status --short
git diff --name-only b5869cb..HEAD -- model.py dataloader.py train.py probe.py benchmarking/
```

Expected: all checks pass, worktree clean, locked-path output empty.

---

### Task 4: Full Public Run, Timing Projection, Result, and Labless Submission

**Files:**
- Create after execution: `docs/results/2026-07-12-molcap-biomed-pca384-s7777.md`
- Modify after submission: the same result document with the Labless response/ID
- Never modify: historical result documents or locked probe code

**Interfaces:**
- Consumes: passed PCA target/report, calibration JSON, committed config/source.
- Produces: full checkpoint/summary/metrics/source snapshot, comparison decision, H100 timing projection, durable report, Labless submission ID.

- [ ] **Step 1: Launch exactly one full public run**

Preferred command:

```powershell
$env:PYTHONUTF8='1'
modal run .superpowers/sdd/modal_molcap_biomed_pca384.py --action full-b200
```

Do not duplicate or restart a live run. Use `full-h200` then `full-h100` only if
the prior accelerator is unavailable or incompatible, and record the reason.

- [ ] **Step 2: Monitor through the full sample budget and locked probe**

Require:

- 999,936 tile presentations / 7,812 full-batch steps;
- stop reason `max_train_samples`;
- MolCap activation at 50% and full scale at 75%;
- finite losses, gradients, and 100% target coverage;
- persisted final checkpoint, summary, metrics, Modal result, and
  `labless_source`;
- complete locked final probe with no changed dataset mapping.

- [ ] **Step 3: Retrieve artifacts and independently verify completion**

Download `summary.json`, `metrics.jsonl`, `modal_result.json`, the lightweight
source snapshot, and checkpoint metadata. Confirm source commit/config, status,
sample budget, score, all category metrics, target hash, and actual GPU.

- [ ] **Step 4: Compute the H100-equivalent timing projection**

Using the frozen formula:

```python
projected_h100_seconds = observed_b200_train_seconds * (
    median_b200_flops_per_sec / median_h100_flops_per_sec
)
margin_seconds = 7200.0 - projected_h100_seconds
```

Report projected minutes, margin, calibration ratio, historical H100 training
`3594.60417402` seconds, historical full function `5439.499483931` seconds, and
any discrepancy. If a fallback GPU ran, report observed timing and do not
invent a B200 projection.

- [ ] **Step 5: Compute the predeclared metric comparison**

Compare every final category against the exact MiniLM and frontier values in
the approved spec. Evaluate:

```python
primary_mean = (molecular_auc + survival_cindex) / 2
mini_primary_mean = (0.611648962395231 + 0.5784693835941472) / 2
semantic_support = (
    molecular_auc > 0.611648962395231
    and survival_cindex > 0.5784693835941472
    and primary_mean - mini_primary_mean >= 0.003
    and linear_mean_f1 - 0.8075887875494312 > -0.003
    and knn_mean_f1 - 0.7521066270862816 > -0.003
)
tile_guard = (
    linear_mean_f1 - 0.8075887875494312 > -0.003
    and knn_mean_f1 - 0.7521066270862816 > -0.003
)
promotion = final_score >= 0.6719107210 and tile_guard
```

Do not reinterpret the endpoint after seeing results.

- [ ] **Step 6: Write and commit the durable result report**

Include:

- source/config/target/report hashes and pinned revisions;
- full PCA eigenvalue/variance audit and gate table;
- B200 smoke and B200/H100 calibration;
- actual and projected hardware timings;
- complete category values and deltas versus MiniLM/frontier;
- primary endpoint and promotion decisions;
- honest next action: additional seeds only at promotion, otherwise EMA
  patient-centroids;
- explicit statement that preprocessing is excluded from training time.

Run a fresh full test/compile/diff gate, then:

```powershell
git add docs/results/2026-07-12-molcap-biomed-pca384-s7777.md
git commit -m "docs: record MolCap PCA-384 result"
```

- [ ] **Step 7: Stage only submission artifacts locally**

Stage `summary.json`, `metrics.jsonl`, and `labless_source` under a local
submission directory. Do not include the target bank, checkpoint, smoke, or
calibration outputs.

- [ ] **Step 8: Dry-run Labless submission**

Set `$RUN_DIR` to the staged full-run directory and `$HARDWARE` to the actual
accelerator, then run:

```powershell
python .\labless\submit_to_labless.py "output_dir=$RUN_DIR" "run_name=molcap-bio-pca384-s7777" "notes=Width- and head-capacity-controlled MolCap A/B: identical captions/training, pinned S-PubMedBERT PCA 768-to-384 before frozen isotropy." "review_config=configs/molcap-biomed-pca384-s7777.yaml" "hardware=$HARDWARE" "dry_run=true"
```

Require completed full status, exact score/config/source, and no policy error.
A PR or H100 is not a submission prerequisite.

- [ ] **Step 9: Submit the completed run regardless of score**

Repeat Step 8 without `dry_run=true`, complete GitHub device authentication if
prompted, and persist the Labless response/ID. This action is authorized for
any completed full run, including null or negative results.

- [ ] **Step 10: Update the report and commit the submission record**

Add the Labless ID/URL, submitted score, hardware, timestamp, and response to
the result document. Run `git diff --check`, then commit:

```powershell
git add docs/results/2026-07-12-molcap-biomed-pca384-s7777.md
git commit -m "docs: record MolCap PCA-384 submission"
```

- [ ] **Step 11: Final verification**

Run:

```powershell
python -m pytest -q
python -m py_compile reembed_molcap_targets.py build_molcap_targets.py dataloader.py model.py train.py
git diff --check
git status --short
git diff --name-only b5869cb..HEAD -- model.py dataloader.py train.py probe.py benchmarking/
```

Verify Modal artifact existence, stopped app state, exact source commit, target
hash, result metrics, and Labless ID. Then request final code review and use the
finishing-a-development-branch workflow.
