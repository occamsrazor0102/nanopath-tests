# MolCap Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a schema-correct, controlled MolCap experiment on the exact `bsc-s7777-k10` frontier.

**Architecture:** Offline preprocessing converts patient metadata into auditable 384-dimensional text, structured, or shuffled targets stored in a non-pickled NPZ. Training routes the target through a small head attached to mean patch tokens, keeps FINO unchanged, schedules the loss by sample progress, and logs shared-trunk gradient conflict.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, NumPy, sentence-transformers, pytest, YAML, SLURM/H100.

## Global Constraints

- Start from commit `11582fd55ba2153c63cdab8fee4fe6d72fc7905d` and the exact `bsc-s7777-k10` recipe.
- Do not modify `probe.py` or `benchmarking/`.
- Preserve the one-million tile-presentation and `1e18` FLOP caps.
- Keep FINO, JEPA, DINO, KDE, block-strided readout, and seed 7777 unchanged.
- MolCap is optional and the disabled path consumes no parameters or RNG.
- A full run is successful only at `mean_probe_score >= 0.6719107210` with linear and kNN declines below `0.003` each.

---

### Task 1: Schema-correct target builder

**Files:**
- Create: `build_molcap_targets.py`
- Modify: `pyproject.toml`
- Test: `tests/test_molcap_targets.py`

**Interfaces:**
- Produces: `aggregate_patients(path: Path) -> pandas.DataFrame`
- Produces: `render_captions(frame: pandas.DataFrame) -> list[str]`
- Produces: `structured_targets(frame: pandas.DataFrame, dim: int, seed: int) -> numpy.ndarray`
- Produces NPZ keys: `patient_ids`, `targets`, `captions`, `mode`

- [ ] **Step 1: Write failing tests for actual column names, deterministic aggregation, missing-value omission, normalization, shuffle determinism, and NPZ safety.**

```python
def synthetic_metadata(tmp_path):
    path = tmp_path / "meta.csv"
    pd.DataFrame([
        {"submitter_id": "TCGA-AA-0001", "cancer_type": "BRCA", "cbio_subtype": "BRCA_LumA", "ajcc_pathologic_stage": "Stage II", "cbio_msi_score": 0.2},
        {"submitter_id": "TCGA-AA-0001", "cancer_type": "BRCA", "cbio_subtype": "BRCA_LumA", "ajcc_pathologic_stage": "Stage II", "cbio_msi_score": 0.2},
        {"submitter_id": "TCGA-BB-0002", "cancer_type": "COAD", "cbio_subtype": np.nan, "ajcc_pathologic_stage": np.nan, "cbio_msi_score": 0.8},
    ]).to_csv(path, index=False)
    return path

def test_caption_uses_real_schema_and_omits_missing(tmp_path):
    patients = aggregate_patients(synthetic_metadata(tmp_path))
    captions = render_captions(patients)
    assert patients.submitter_id.tolist() == ["TCGA-AA-0001", "TCGA-BB-0002"]
    assert "stage nan" not in " ".join(captions).lower()
    assert "BRCA" in captions[0]

def test_structured_targets_are_deterministic_unit_vectors(tmp_path):
    patients = aggregate_patients(synthetic_metadata(tmp_path))
    a = structured_targets(patients, 384, 7777)
    b = structured_targets(patients, 384, 7777)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0)
```

- [ ] **Step 2: Run the target tests and confirm they fail because the builder does not exist.**

Run: `python -m pytest tests/test_molcap_targets.py -q`

Expected: collection failure for `build_molcap_targets`.

- [ ] **Step 3: Implement deterministic patient aggregation, caption rendering, structured projection, frozen MiniLM encoding, shuffle mode, geometry audit, and NPZ output.**

```python
patients = raw.groupby("submitter_id", sort=True, as_index=False).first()
targets = isotropize(encoder.encode(captions, normalize_embeddings=True)).astype("float32")
np.savez(output, patient_ids=ids, targets=targets, captions=captions, mode=np.array(mode))
```

- [ ] **Step 4: Run the target tests and full compile check.**

Run: `python -m pytest tests/test_molcap_targets.py -q && python -m py_compile build_molcap_targets.py`

Expected: all target tests pass; compile exits zero.

- [ ] **Step 5: Commit the target builder.**

```bash
git add build_molcap_targets.py pyproject.toml tests/test_molcap_targets.py
git commit -m "feat: build auditable MolCap targets"
```

### Task 2: Target loading and MolCap primitives

**Files:**
- Modify: `dataloader.py`
- Modify: `model.py`
- Modify: `train.py`
- Test: `tests/test_molcap_training.py`

**Interfaces:**
- Produces: `MolCapHead(in_dim: int, target_dim: int)`
- Produces: `molcap_loss(head, features, targets, present, views) -> torch.Tensor`
- Produces: `linear_ramp(progress: float, start: float, length: float) -> float`
- Produces batch keys: `molcap_target`, `molcap_present`

- [ ] **Step 1: Write failing tests for target lookup, missing-patient masking, crop-major repetition, ramp endpoints, gradients, and disabled-path behavior.**

```python
def test_patch_route_reaches_head_and_trunk_not_cls():
    patches = torch.randn(4, 16, 8, requires_grad=True)
    cls = torch.randn(4, 8, requires_grad=True)
    loss = molcap_loss(MolCapHead(8, 6), patches.mean(1), targets, present, views=2)
    loss.backward()
    assert patches.grad is not None
    assert cls.grad is None

def test_ramp_uses_sample_progress():
    assert linear_ramp(0.49, 0.50, 0.25) == 0.0
    assert linear_ramp(0.625, 0.50, 0.25) == 0.5
    assert linear_ramp(0.75, 0.50, 0.25) == 1.0
```

- [ ] **Step 2: Run the training tests and confirm the missing symbols fail.**

Run: `python -m pytest tests/test_molcap_training.py -q`

Expected: import failures for `MolCapHead`, `molcap_loss`, and `linear_ramp`.

- [ ] **Step 3: Implement the smallest target loader, head, masked cosine loss, and ramp that satisfy the tests.**

```python
class MolCapHead(nn.Module):
    def __init__(self, in_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, target_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
```

- [ ] **Step 4: Run focused tests, then the full test suite.**

Run: `python -m pytest tests/test_molcap_training.py -q && python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit primitives.**

```bash
git add dataloader.py model.py train.py tests/test_molcap_training.py
git commit -m "feat: add patch-routed MolCap objective"
```

### Task 3: Frontier training integration and diagnostics

**Files:**
- Modify: `train.py`
- Create: `configs/molcap-text-s7777.yaml`
- Test: `tests/test_molcap_integration.py`

**Interfaces:**
- Consumes: NPZ target bank and MolCap primitives from Tasks 1-2
- Produces: checkpoint key `molcap_head`
- Produces metrics: `molcap`, `molcap_scale`, `molcap_coverage`, `molcap_grad_cosine`, `molcap_grad_norm_ratio`, `meta`

- [ ] **Step 1: Write a failing CPU integration test covering forward, backward, checkpoint round-trip, and finite gradient diagnostics.**

```python
def test_molcap_cpu_step_and_checkpoint_round_trip(tmp_path):
    model = DinoV2ViT(variant_cfg=(8, 1, 2, 2, "mlp", True, "unused", 0))
    head = MolCapHead(8, 6)
    out = model(torch.randn(2, 3, 28, 28))
    targets = F.normalize(torch.randn(2, 6), dim=-1)
    aux = molcap_loss(head, out["x_norm_patchtokens"].mean(1), targets, torch.ones(2), views=1)
    aux.backward()
    assert torch.isfinite(aux)
    assert model.blocks[0].attn.qkv.weight.grad is not None
    torch.save({"molcap_head": head.state_dict()}, tmp_path / "step.pt")
    MolCapHead(8, 6).load_state_dict(torch.load(tmp_path / "step.pt")["molcap_head"])
```

- [ ] **Step 2: Run the integration test and confirm it fails before wiring.**

Run: `python -m pytest tests/test_molcap_integration.py -q`

Expected: failure because training integration and diagnostics are absent.

- [ ] **Step 3: Add optional model setup, optimizer/checkpoint integration, patch routing, shared-trunk gradient diagnostics at log steps, and W&B/JSON logging.**

```python
molcap_scale = linear_ramp(sfrac, molcap_cfg["ramp_start"], molcap_cfg["ramp_len"])
molcap = molcap_cfg["weight"] * molcap_scale * molcap_loss(...)
total_loss = dino_loss_value + jepa_loss + kde + meta_loss + molcap
```

- [ ] **Step 4: Add the exact frontier config with `weight: 0.03`, `ramp_start: 0.50`, and `ramp_len: 0.25`; run YAML and integration checks.**

Run: `python -m pytest tests/test_molcap_integration.py -q && python -c "import yaml; yaml.safe_load(open('configs/molcap-text-s7777.yaml'))"`

Expected: integration passes and YAML parses.

- [ ] **Step 5: Commit training integration.**

```bash
git add train.py configs/molcap-text-s7777.yaml tests/test_molcap_integration.py
git commit -m "experiment: wire MolCap into frontier recipe"
```

### Task 4: Build and audit the real target bank

**Files:**
- Generated outside git: `/data/$USER/nanopath/molcap_text_384.npz`

- [ ] **Step 1: Install the locked environment and build text targets from the committed metadata.**

Run: `uv sync && uv run python build_molcap_targets.py metadata/tcga_master_dataset.csv /data/$USER/nanopath/molcap_text_384.npz text`

Expected: audit reports at least 95% patient coverage, finite unit vectors, target standard deviation above `0.01`, and more than 32 effective dimensions.

- [ ] **Step 2: Run the builder twice and compare SHA-256 hashes.**

Run: `sha256sum /data/$USER/nanopath/molcap_text_384.npz /data/$USER/nanopath/molcap_text_384_repeat.npz`

Expected: identical hashes.

### Task 5: Smoke verification and full H100 run

**Files:**
- Generated outside git: `/data/$USER/nanopath/molcap/molcap-text-s7777/`

- [ ] **Step 1: Run the complete local verification gate.**

Run: `python -m pytest -q && python -m py_compile model.py dataloader.py train.py build_molcap_targets.py && git diff --check`

Expected: zero failures and zero compile/diff errors.

- [ ] **Step 2: Run a short GPU smoke with reduced caps and probes disabled in a temporary copied config.**

Run: `python train.py configs/molcap-smoke.yaml`

Expected: finite MolCap loss, nonzero target coverage, finite gradient diagnostics, checkpoint save, and no probe execution.

- [ ] **Step 3: Launch the full run on one H100.**

Run: `./submit/train_1gpu.sbatch configs/molcap-text-s7777.yaml output_dir=/data/$USER/nanopath/molcap/molcap-text-s7777`

Expected: job submission succeeds and W&B begins logging.

- [ ] **Step 4: Monitor until training and the final probe complete; inspect `summary.json`, `metrics.jsonl`, utilization, and Labless eligibility.**

Run: `squeue -u $USER; tail -f "$(ls -t slurm/*.out | head -1)"`

Expected: `stop_reason=max_train_samples`, `tile_presentations <= 1000000`, `max_train_flops=1e18`, all eight probe metrics present.

- [ ] **Step 5: Compare to `bsc-s7777-k10` and decide whether to replicate, tune, or stop.**

Run: `python -c "import json; print(json.load(open('/data/$USER/nanopath/molcap/molcap-text-s7777/summary.json')))"`

Expected: evidence-backed decision using the spec thresholds.

## Self-review

- Every design requirement maps to a task above.
- All production functions are introduced after a failing test.
- Target widths, NPZ keys, batch keys, checkpoint keys, and metric names are consistent across tasks.
- No locked probe files are modified.
