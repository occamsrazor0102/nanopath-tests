# MolCap Biomedical Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, audit, train, probe, and submit a strict seed-7777 MolCap encoder-only A/B using the exact canonical MiniLM captions re-embedded with pinned S-PubMedBert-MS-MARCO.

**Architecture:** An offline helper loads the canonical NPZ rather than metadata, re-encodes the identical strings with pinned generic and biomedical sentence encoders, independently applies the existing isotropy function, enforces predeclared geometry/coverage gates, and writes a deterministic 768-dimensional bank plus JSON report. Training uses the existing encoder-agnostic MolCap path with a config that differs from the MiniLM arm only in labels, target path, and target dimension. A local ignored Modal harness builds the artifact, runs an H100 smoke, executes the full locked probe, and stages the completed run for Labless submission.

**Tech Stack:** Python 3.12, NumPy, sentence-transformers, pandas, PyTorch, pytest, YAML, Modal H100, Labless submission wrapper.

## Global Constraints

- Canonical MiniLM target SHA-256 is `2F6648A4155B96757A136335A253E3FAEB6029A92A7E6356380CE80805011577`.
- MiniLM model/revision is `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Biomedical model/revision is `pritamdeka/S-PubMedBert-MS-MARCO@96786c7024f95c5aac7f2b9a18086c7b97b23036` and output width is exactly `768`.
- Each encoder independently uses the same mean removal, `0.05` eigenvalue floor, `-0.1` covariance power, rotation-back, and final L2 normalization.
- Caption strings, patient order, seed 7777, data split, crops, FINO, weight `0.03`, ramp `[0.50, 0.75]`, and locked probes do not change.
- Training is forbidden if any target gate in the approved spec fails; thresholds are not retuned after observing PubMedBERT geometry.
- `probe.py` and `benchmarking/` remain untouched.
- The completed full run is submitted to Labless even if null or negative; smoke runs are never submitted.

---

### Task 1: Deterministic Re-embedding and Geometry Audit

**Files:**
- Create: `reembed_molcap_targets.py`
- Create: `tests/test_molcap_reembed.py`
- Modify: `.gitignore`
- Reuse: `build_molcap_targets.py:isotropize,save_target_bank`

**Interfaces:**
- Consumes: canonical NPZ keys `patient_ids`, `targets`, `captions`, `mode`; committed `metadata/fino_meta.json`; encoder objects exposing `encode(captions, normalize_embeddings=True, ...)`.
- Produces: `geometry_metrics(targets) -> dict[str, int | float]`; `fino_patient_ids(path) -> set[str]`; `validate_candidate(reference, candidate, patient_ids, fino_ids) -> dict[str, object]`; `build_reembedded_bank(source, output, report, fino_path, minilm_encoder, biomedical_encoder, expected_source_sha) -> dict[str, object]`. `validate_candidate` derives its required row count from `len(patient_ids)`; the production builder separately requires the canonical count of 11,428.

- [ ] **Step 1: Write the failing geometry tests**

Create `tests/test_molcap_reembed.py` with hand-computable unit vectors and explicit gate failures:

```python
import hashlib
import json

import numpy as np

from build_molcap_targets import save_target_bank
from reembed_molcap_targets import (
    BIOMED_DIM,
    BIOMED_MODEL,
    BIOMED_REVISION,
    ISOTROPY_FLOOR,
    ISOTROPY_POWER,
    MINILM_MODEL,
    MINILM_REVISION,
    build_reembedded_bank,
    fino_patient_ids,
    geometry_metrics,
    validate_candidate,
)


def test_geometry_metrics_for_orthogonal_vectors():
    targets = np.eye(4, dtype=np.float32)
    metrics = geometry_metrics(targets)
    assert metrics["rows"] == 4
    assert metrics["width"] == 4
    assert metrics["mean_off_diagonal_cosine"] == 0.0
    assert metrics["effective_rank"] == 3.0
    assert metrics["participation_ratio"] == 3.0
    assert metrics["max_unit_norm_error"] == 0.0


def test_constants_pin_models_and_shared_isotropy():
    assert MINILM_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert MINILM_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert BIOMED_MODEL == "pritamdeka/S-PubMedBert-MS-MARCO"
    assert BIOMED_REVISION == "96786c7024f95c5aac7f2b9a18086c7b97b23036"
    assert BIOMED_DIM == 768
    assert ISOTROPY_FLOOR == 0.05
    assert ISOTROPY_POWER == 0.1
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_molcap_reembed.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'reembed_molcap_targets'`.

- [ ] **Step 3: Implement geometry metrics and frozen constants**

Create `reembed_molcap_targets.py` with:

```python
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BIOMED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
BIOMED_REVISION = "96786c7024f95c5aac7f2b9a18086c7b97b23036"
BIOMED_DIM = 768
ISOTROPY_FLOOR = 0.05
ISOTROPY_POWER = 0.1
CANONICAL_SHA256 = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"


def geometry_metrics(targets):
    x = np.asarray(targets, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    centered = x - x.mean(0, keepdims=True)
    values = np.linalg.eigvalsh(centered.T @ centered / max(1, len(x) - 1)).clip(0)
    weights = values / values.sum()
    effective_rank = np.exp(-(weights[weights > 0] * np.log(weights[weights > 0])).sum())
    participation = values.sum() ** 2 / np.square(values).sum()
    variance = x.var(0, ddof=1)
    off_diagonal = (np.square(x.sum(0)).sum() - len(x)) / (len(x) * (len(x) - 1))
    return {
        "rows": len(x),
        "width": x.shape[1],
        "mean_off_diagonal_cosine": float(off_diagonal),
        "effective_rank": float(effective_rank),
        "normalized_effective_rank": float(effective_rank / x.shape[1]),
        "participation_ratio": float(participation),
        "normalized_participation_ratio": float(participation / x.shape[1]),
        "variance_min": float(variance.min()),
        "variance_median": float(np.median(variance)),
        "variance_max": float(variance.max()),
        "variance_cv": float(variance.std() / variance.mean()),
        "max_unit_norm_error": float(np.abs(norms - 1).max()),
    }
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m pytest tests/test_molcap_reembed.py::test_geometry_metrics_for_orthogonal_vectors tests/test_molcap_reembed.py::test_constants_pin_models_and_shared_isotropy -q`

Expected: `2 passed`.

- [ ] **Step 5: Add failing canonical-copy, coverage, gate, and deterministic-output tests**

Append fixtures using 40-dimensional fake embeddings so corrected geometry can exceed the rank gates. The test must:

```python
def test_fino_patient_ids_unions_every_mapping(tmp_path):
    path = tmp_path / "fino.json"
    path.write_text(json.dumps({"discrete": {"a": {"P1": 1}}, "continuous": {"b": {"P2": 0.2}}}))
    assert fino_patient_ids(path) == {"P1", "P2"}


def test_validate_candidate_rejects_each_hard_gate():
    reference = {
        "normalized_effective_rank": 0.10,
        "normalized_participation_ratio": 0.06,
    }
    candidate = {
        "rows": 2, "width": 768, "mean_off_diagonal_cosine": 0.0,
        "effective_rank": 40.0, "participation_ratio": 20.0,
        "normalized_effective_rank": 0.052, "normalized_participation_ratio": 0.03,
        "variance_cv": 0.3, "max_unit_norm_error": 1e-7,
    }
    report = validate_candidate(reference, candidate, np.array(["P1", "P2"]), {"P1", "P2"})
    assert report["coverage_fraction"] == 1.0
```

Parameterize mutations for width 767, norm error `2e-5`, cosine `0.02`, effective rank 31, participation 15, variance CV `0.8`, normalized-rank ratios below `0.5` and above `2.0`, and missing FINO patient; each must raise an assertion naming the failed gate.

Use fake encoders that record received captions/revision and return deterministic full-rank arrays. Build twice from the same canonical fixture and assert:

```python
assert first_ids.tolist() == source_ids.tolist()
assert first_captions.tolist() == source_captions.tolist()
assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
assert json.loads(report.read_text())["models"]["biomedical"]["revision"] == BIOMED_REVISION
```

- [ ] **Step 6: Run the expanded test and verify RED**

Run: `python -m pytest tests/test_molcap_reembed.py -q`

Expected: failures for undefined `fino_patient_ids`, `validate_candidate`, and `build_reembedded_bank`.

- [ ] **Step 7: Implement loading, encoding, gates, deterministic artifacts, and CLI**

Implementation requirements:

```python
def fino_patient_ids(path):
    payload = json.loads(Path(path).read_text())
    return {patient for group in ("discrete", "continuous") for mapping in payload[group].values() for patient in mapping}


def encode(encoder, captions, expected_dim):
    raw = encoder.encode(
        captions, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64, convert_to_numpy=True,
    )
    raw = np.asarray(raw, dtype=np.float32)
    assert raw.shape == (len(captions), expected_dim), raw.shape
    return raw
```

`build_reembedded_bank` must verify the canonical file hash when `expected_source_sha` is not `None`, load with `allow_pickle=False`, assert exact required keys, re-encode both models, call the shared `isotropize` function separately for each raw matrix, require regenerated MiniLM targets to match canonical targets at `atol=2e-5, rtol=0`, validate geometry and FINO coverage, write the biomedical bank through `save_target_bank`, write sorted JSON atomically, write a second temporary NPZ from the same arrays, and assert equal hashes before replacing the requested output.

The key-value CLI is:

```powershell
python reembed_molcap_targets.py source=$env:TEMP\molcap_text_384.npz output=$env:TEMP\molcap_biomed_768.npz report=$env:TEMP\molcap_biomed_768.geometry.json fino=metadata\fino_meta.json device=cpu
```

It constructs each `SentenceTransformer` with the pinned `revision` and requested `device`; model/revision overrides are not exposed.

- [ ] **Step 8: Exclude the helper from Labless and verify all Task 1 tests**

Add `reembed_molcap_targets.py` to `.gitignore`, extend `test_development_helpers_are_excluded_from_labless_snapshot`, then run:

`python -m pytest tests/test_molcap_reembed.py tests/test_molcap_targets.py tests/test_molcap_config.py -q`

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add -f reembed_molcap_targets.py tests/test_molcap_reembed.py
git add .gitignore tests/test_molcap_config.py
git commit -m "feat: build audited biomedical MolCap targets"
```

---

### Task 2: Strict Biomedical Config and 768-Dimensional Integration

**Files:**
- Create: `configs/molcap-biomed-s7777.yaml`
- Modify: `tests/test_molcap_config.py`
- Modify: `tests/test_molcap_integration.py`

**Interfaces:**
- Consumes: existing `configs/molcap-text-s7777.yaml`; existing `load_molcap_bank`, `MolCapHead`, checkpoint, and patch-routed loss interfaces.
- Produces: a full locked config whose semantic diff from MiniLM is exactly project labels, target path, and `target_dim`.

- [ ] **Step 1: Write the failing exact-diff config test**

Add a recursive leaf-diff helper and assert:

```python
def test_biomedical_config_is_encoder_only_ab():
    generic = yaml.safe_load(Path("configs/molcap-text-s7777.yaml").read_text())
    biomedical = yaml.safe_load(Path("configs/molcap-biomed-s7777.yaml").read_text())
    assert changed_leaves(generic, biomedical) == {
        "project.name", "project.output_dir", "molcap.targets", "molcap.target_dim"
    }
    assert biomedical["project"]["name"] == "molcap-biomed-s7777"
    assert biomedical["molcap"]["targets"] == "/data/$USER/nanopath/molcap_biomed_768.npz"
    assert biomedical["molcap"]["target_dim"] == 768
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_molcap_config.py::test_biomedical_config_is_encoder_only_ab -q`

Expected: failure because `configs/molcap-biomed-s7777.yaml` does not exist.

- [ ] **Step 3: Create the biomedical config with only permitted changes**

Copy `configs/molcap-text-s7777.yaml` and change exactly:

```yaml
project:
  name: molcap-biomed-s7777
  output_dir: /data/$USER/nanopath/molcap/molcap-biomed-s7777
molcap:
  targets: /data/$USER/nanopath/molcap_biomed_768.npz
  target_dim: 768
```

- [ ] **Step 4: Verify the config test passes**

Run: `python -m pytest tests/test_molcap_config.py -q`

Expected: all config tests pass.

- [ ] **Step 5: Write and run a failing 768-dimensional integration assertion**

Extend the tiny integration fixture to accept `target_dim`. Add a test that saves one 768-dimensional target, loads a training sample, constructs `MolCapHead(8, 768)`, performs patch-routed cosine loss and backward, saves/restores the head, and asserts finite nonzero gradients. Run it before fixture support and confirm the expected signature/shape failure.

- [ ] **Step 6: Implement fixture support and verify GREEN**

Update only test helpers; production training is already dimension-agnostic. Run:

`python -m pytest tests/test_molcap_integration.py tests/test_molcap_training.py -q`

Expected: all integration/training tests pass.

- [ ] **Step 7: Run the complete local gate and commit Task 2**

```powershell
python -m pytest -q
python -m py_compile reembed_molcap_targets.py build_molcap_targets.py dataloader.py model.py train.py
git diff --check
git add configs/molcap-biomed-s7777.yaml tests/test_molcap_config.py tests/test_molcap_integration.py
git commit -m "experiment: add paired biomedical MolCap arm"
```

Expected: at least 17 existing tests plus all new tests pass; worktree is clean.

---

### Task 3: Build, Smoke, Full Run, Report, and Submit

**Files:**
- Create locally but keep ignored: `.superpowers/sdd/modal_molcap_biomed.py`
- Create after results: `docs/results/2026-07-12-molcap-biomed-s7777.md`
- Never modify: `probe.py`, `benchmarking/`

**Interfaces:**
- Consumes: committed source, canonical MiniLM NPZ on Modal volume, `/data/repo-data/nanopath_parquet/fino_meta.json`, biomedical config, and persistent nanopath data/probe roots.
- Produces: audited target and geometry report, smoke artifacts, final checkpoint/summary/metrics, comparison report, and Labless submission ID.

- [ ] **Step 1: Build an ignored Modal execution harness**

Adapt the proven `.superpowers/sdd/modal_molcap.py` pattern with actions:

- `build-target`: H100 function that loads `/data/repo-data/molcap_text_384.npz`, runs the pinned re-embed CLI against `/data/repo-data/nanopath_parquet/fino_meta.json`, writes `/data/repo-data/molcap_biomed_768.npz` and `/data/repo-data/molcap_biomed_768.geometry.json`, rebuilds to a temporary second artifact, asserts matching hashes, and commits the volume.
- `smoke`: batch 8, 1,024 samples, two local views, probes off, ramp `[0.0, 0.25]`, diagnostics on, output `/data/experiments/readout-local-context/smoke/molcap-biomed-s7777`.
- `full`: unchanged committed config except volume path rewrites and output `/data/experiments/readout-local-context/full/molcap-biomed-s7777`.

Use exact H100 allocation, W&B offline mode, the existing persistent volume, and a four-hour timeout. Do not expose a Labless auto-submit path inside Modal.

- [ ] **Step 2: Run target build and enforce all audit gates**

Run: `modal run .superpowers/sdd/modal_molcap_biomed.py --action build-target`

Expected: 11,428 x 768 finite unit vectors, 9,389/9,389 coverage, all geometry ratios inside their frozen ranges, deterministic hashes, and a persisted JSON report. If any gate fails, stop and report; do not tune isotropy.

- [ ] **Step 3: Retrieve and independently inspect the target report**

Download the geometry JSON and target NPZ with `modal volume get`. Recompute hashes, dimensions, unit norms, coverage, and geometry locally without trusting the remote summary. Confirm exact patient/caption equality to the MiniLM bank.

- [ ] **Step 4: Run the H100 smoke and inspect diagnostics**

Run: `modal run .superpowers/sdd/modal_molcap_biomed.py --action smoke`.

Require 1,024 samples, 100% coverage, finite active MolCap loss, finite gradient cosine/norm ratio, persisted checkpoint/summary, no probe, and no memory regression inconsistent with the wider head.

- [ ] **Step 5: Re-run local verification and launch the full experiment**

Run the complete pytest/compile/diff gate, require a clean worktree, then:

`modal run .superpowers/sdd/modal_molcap_biomed.py --action full`

Monitor the one-million-sample training through the 50%-75% ramp and the complete locked final probe. Do not duplicate or restart a live job.

- [ ] **Step 6: Retrieve artifacts and compute predeclared comparisons**

Download `summary.json`, `metrics.jsonl`, `modal_result.json`, and the lightweight source snapshot. Compare every category against:

- MiniLM: overall `0.6652239193862792`, linear `0.8075887875494312`, kNN `0.7521066270862816`, few-shot `0.68987434293451`, slide `0.6702551834130781`, segmentation `0.33196826873936`, molecular `0.611648962395231`, survival `0.5784693835941472`, robustness `0.8798797993781943`.
- Frontier: overall `0.6659107210081501`, linear `0.8076442781000113`, kNN `0.7510353902180573`, few-shot `0.6904189435158125`, slide `0.6671170772047965`, segmentation `0.3341096331514823`, molecular `0.6139796512930842`, survival `0.5825118967310817`, robustness `0.8804688978508746`.

Evaluate the primary two-endpoint mean and tile guard exactly as specified; do not reinterpret after observing results.

- [ ] **Step 7: Write and commit the durable result report**

Create `docs/results/2026-07-12-molcap-biomed-s7777.md` with target hashes/provenance, raw/corrected geometry table, execution metadata, all category values/deltas, primary decision, and the next action (additional seeds only at the promotion gate; otherwise patient centroid). Commit it after a fresh full test pass.

- [ ] **Step 8: Dry-run and submit the completed run to Labless**

Stage only `summary.json`, `metrics.jsonl`, and `labless_source` locally. Run:

```powershell
python .\labless\submit_to_labless.py "output_dir=$RUN_DIR" "run_name=molcap-bio-s7777" "notes=Strict encoder-only MolCap A/B: identical captions and routing, S-PubMedBert 768-d with independently refit frozen isotropy." "review_config=configs/molcap-biomed-s7777.yaml" "hardware=NVIDIA H100 80GB" "dry_run=true"
```

Require `status=completed` and the exact final metric, then repeat without `dry_run=true`, complete GitHub device authentication, record the Labless response/ID in the result report, and commit that record.

- [ ] **Step 9: Final verification**

Run `python -m pytest -q`, compilation, `git diff --check`, `git status --short`, inspect the exact experiment source commit and Modal artifact existence, then report the honest result and branch state.
