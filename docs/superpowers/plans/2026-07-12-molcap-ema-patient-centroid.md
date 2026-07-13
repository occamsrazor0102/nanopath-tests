# MolCap Probe-Route and EMA Patient-Centroid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build, verify, train, probe, time-calibrate, and submit a paired seed-7777 route-control and hierarchical EMA patient-centroid MolCap experiment that isolates historical teacher substitution.

**Architecture:** Both arms extract the frozen probe feature family from an additional RNG-isolated unmasked student forward and the existing unmasked EMA-teacher forward, average global views per tile, then pool tiles to slides and slides equally to patients. Arm R forwards the current teacher patient value through an identity-gradient student estimator; Arm C replaces only that forward value with a deterministic online slide-EMA/equal-slide patient value. The complex state remains in top-level, importable units in train.py because Labless permits reviewed production changes only in train.py, model.py, dataloader.py, and prepare.py.

**Tech Stack:** Python 3.12, PyTorch 2.8/CUDA 12.9, NumPy, PyArrow, pytest, YAML, Modal single-GPU B200/H100/H200, Weights & Biases, Labless.

## Global Constraints

- Approved specification: docs/superpowers/specs/2026-07-12-molcap-ema-patient-centroid-design.md.
- Implementation baseline: 70b9afe62cf9b888880617964719f02be98a3fb2.
- Locked-path reference: 01c1cdf8017a0481636a28ab58a0ddc67d6e0a06.
- Canonical MiniLM target SHA-256: 2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577.
- Canonical target: exactly 11,428 unique patients, 384 columns, finite unit rows, mode text.
- Both scored arms use seed 7777, one million samples, batch 128, two global views, eight local views, the existing FINO/DINO/JEPA/KDE recipe, MolCap weight 0.03, and ramp 0.50 through 0.75.
- Exact readout blocks are 4, 6, 8, and 11; ViT-S input width is 1,536; MolCap hidden/output widths are 512 and 384.
- Arm R and Arm C share current-teacher forward source and identity-gradient current-student backward source. Only historical substitution and its deterministic state differ.
- History is slide EMA with momentum 0.9, first-copy initialization, equal-slide patient pooling, and single-GPU execution.
- WORLD_SIZE greater than one aborts.
- Arm C incremental peak memory over Arm R must not exceed 0.5 GiB.
- Full checkpoints contain authoritative history state; probe checkpoints contain no MolCap head or history state.
- Scored runs start from scratch. Resume safety is implemented and tested but is not used to claim paired bit identity.
- probe.py, every tracked file under benchmarking/, target builders, target artifacts, and historical result records remain byte-identical.
- No new production Python file is allowed: Labless review accepts training changes only in train.py, model.py, dataloader.py, and prepare.py.
- The paired scored configs differ at exactly project.name, project.recipe_id, project.output_dir, and molcap.history.enabled.
- Smoke and calibration runs are never submitted. Every completed full run is submitted regardless of score.
- Public execution prefers one B200. An infrastructure fallback uses one H200 and then one exact H100; no multi-GPU path is added.
- B300 and B200+ are excluded because the pinned stack is CUDA 12.9.
- Preprocessing and indexing time are excluded from training time.

## File Structure

- Modify dataloader.py: canonical target provenance, dense train-only patient/slide mapping, two contiguous batch indices.
- Modify model.py: one shared block-strided probe-readout path available from forward without changing its default output.
- Modify train.py: hierarchy, STE, state, geometry, gates, routing, strict checkpoints, diagnostics, runner-only sample cap.
- Create configs/molcap-probe-route-s7777.yaml: scored Arm R.
- Create configs/molcap-ema-centroid-s7777.yaml: scored Arm C.
- Create tests/test_molcap_centroid.py: pure hierarchy/state/geometry/gate tests; ignored by Labless snapshots.
- Modify tests/test_molcap_training.py: target API and 1,536-D seed-neutral head regressions.
- Modify tests/test_molcap_integration.py: dataset/readout/RNG/checkpoint/mechanics integration.
- Modify tests/test_molcap_config.py: exact config leaves and locked-path manifest.
- Create locally, keep ignored: .superpowers/sdd/modal_molcap_centroid.py.
- Create after execution: docs/results/2026-07-12-molcap-probe-route-ema-centroid-s7777.md.

---

### Task 1: Freeze the Paired Config Contract

**Files:**
- Create: configs/molcap-probe-route-s7777.yaml
- Create: configs/molcap-ema-centroid-s7777.yaml
- Modify: tests/test_molcap_config.py

**Interfaces:**
- Consumes: configs/molcap-text-s7777.yaml as the unchanged recipe base.
- Produces: two full scored configs whose recursive diff is exactly the four approved leaves.

- [ ] **Step 1: Write the failing exact-config tests**

Add these assertions to tests/test_molcap_config.py:

~~~python
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
~~~

- [ ] **Step 2: Run the tests and confirm the configs are absent**

Run:

~~~powershell
python -m pytest tests/test_molcap_config.py::test_route_and_centroid_configs_differ_at_exactly_four_leaves tests/test_molcap_config.py::test_route_and_centroid_configs_freeze_registered_contract -q
~~~

Expected: both tests fail with FileNotFoundError for the two new YAML files.

- [ ] **Step 3: Create both complete configs from the MiniLM config**

Copy configs/molcap-text-s7777.yaml twice, then use these exact project values:

~~~yaml
# Arm R
project:
  name: molcap-probe-route-s7777
  family: nanopath
  recipe_id: dinov2-vits14-reg-jepa-mask10-molcap-probe-route
  output_dir: /data/$USER/nanopath/molcap/molcap-probe-route-s7777

# Arm C
project:
  name: molcap-ema-centroid-s7777
  family: nanopath
  recipe_id: dinov2-vits14-reg-jepa-mask10-molcap-ema-centroid
  output_dir: /data/$USER/nanopath/molcap/molcap-ema-centroid-s7777
~~~

Replace the MolCap block in both files with the same block below, changing only history.enabled:

~~~yaml
molcap:
  enabled: true
  targets: /data/$USER/nanopath/molcap_text_384.npz
  target_sha256: 2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577
  target_dim: 384
  weight: 0.03
  ramp_start: 0.5
  ramp_len: 0.25
  diagnose: true
  route: probe_cls_hierarchical
  feature_blocks: [4, 6, 8, 11]
  input_dim: 1536
  head_hidden_dim: 512
  forward_source: teacher
  gradient_source: student_identity_ste
  history:
    enabled: false
    level: slide_then_patient
    momentum: 0.9
    min_slide_updates: 2
    min_sample_weighted_coverage: 0.95
    min_geometry_patients: 512
    min_effective_rank: 32
    min_participation_ratio: 16
    max_mean_offdiag_cosine: 0.95
    min_centroid_norm: 1.0e-6
~~~

Arm C uses enabled: true. Preserve every other YAML leaf byte-for-byte from configs/molcap-text-s7777.yaml.

- [ ] **Step 4: Run and commit**

~~~powershell
python -m pytest tests/test_molcap_config.py -q
git add configs/molcap-probe-route-s7777.yaml configs/molcap-ema-centroid-s7777.yaml tests/test_molcap_config.py
git commit -m "test: freeze paired MolCap centroid configs"
~~~

Expected: all config tests pass before the commit.

---

### Task 2: Add Dense Train-Only Identity Mapping

**Files:**
- Modify: dataloader.py:56-61, 102-156, 218-248
- Modify: tests/test_molcap_training.py
- Modify: tests/test_molcap_integration.py

**Interfaces:**
- Produces: load_molcap_bank(path, target_dim, return_patient_ids=False).
- Produces dataset attributes: molcap_patient_ids, molcap_slide_ids, molcap_slide_to_patient, molcap_mapping_digest, molcap_target_sha256.
- Produces batch scalars: molcap_slide_idx and molcap_patient_idx.

- [ ] **Step 1: Write failing target-order and dense-index tests**

Extend tests/test_molcap_training.py:

~~~python
def test_target_bank_can_return_canonical_patient_order(tmp_path):
    path = tmp_path / "targets.npz"
    save_target_bank(
        path,
        ["TCGA-BB-0002", "TCGA-AA-0001"],
        np.eye(2, 4, dtype=np.float32),
        ["second", "first"],
        "structured",
    )
    bank, patient_ids = load_molcap_bank(path, 4, return_patient_ids=True)
    assert patient_ids == ("TCGA-BB-0002", "TCGA-AA-0001")
    assert set(bank) == set(patient_ids)
~~~

Add a two-patient, three-slide parquet fixture to tests/test_molcap_integration.py and assert canonical-filtered patient order, lexical slide order, slide_to_patient [1, 0, 0], a 64-character digest, and int64 item indices.

- [ ] **Step 2: Run the focused tests**

~~~powershell
python -m pytest tests/test_molcap_training.py::test_target_bank_can_return_canonical_patient_order tests/test_molcap_integration.py::test_dataset_emits_deterministic_dense_centroid_indices -q
~~~

Expected: failure because return_patient_ids and dense indices do not exist.

- [ ] **Step 3: Extend target loading without breaking old callers**

~~~python
def load_molcap_bank(path, target_dim, return_patient_ids=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"patient_ids", "targets", "captions", "mode"}
        patient_ids = tuple(str(x) for x in artifact["patient_ids"])
        targets = artifact["targets"]
        mode = str(artifact["mode"])
    assert len(set(patient_ids)) == len(patient_ids)
    assert targets.ndim == 2 and targets.shape == (len(patient_ids), target_dim)
    assert np.isfinite(targets).all()
    bank = {patient_id: target.astype(np.float32) for patient_id, target in zip(patient_ids, targets)}
    return (bank, patient_ids) if return_patient_ids else bank
~~~

During the existing path scan, collect only train patient and slide sets. After the scan:

~~~python
ordered_patients = tuple(pid for pid in canonical_patient_ids if pid in train_patients)
assert len(ordered_patients) == len(train_patients)
patient_index = {pid: i for i, pid in enumerate(ordered_patients)}
ordered_slides = tuple(sorted(train_slides))
slide_index = {sid: i for i, sid in enumerate(ordered_slides)}
slide_to_patient = np.asarray(
    [patient_index["-".join(sid.split("-")[:3])] for sid in ordered_slides],
    dtype=np.int64,
)
mapping_payload = {
    "version": 1,
    "patient_ids": ordered_patients,
    "slide_ids": ordered_slides,
    "slide_to_patient": slide_to_patient.tolist(),
}
mapping_digest = hashlib.sha256(
    json.dumps(mapping_payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
~~~

Store ordered tuples, lookup dictionaries, slide_to_patient, digest, and target-file SHA on routed training datasets. Assert every training patient has a target. Historical patch configs and validation retain current behavior.

For routed configs, also assert the actual file SHA equals molcap.target_sha256, there are exactly 11,428 patient rows, mode is text, every row is finite, and maximum unit-norm error is at most 1e-5. Add a test that corrupting the configured SHA fails during dataset construction.

- [ ] **Step 4: Emit indices from the final selected path**

~~~python
if self.molcap_identity_enabled:
    molcap_keys["molcap_slide_idx"] = torch.tensor(self.molcap_slide_index[slide_stem], dtype=torch.int64)
    molcap_keys["molcap_patient_idx"] = torch.tensor(self.molcap_patient_index[patient_id], dtype=torch.int64)
~~~

Use the final rel after tissue resampling.

- [ ] **Step 5: Run and commit**

~~~powershell
python -m pytest tests/test_molcap_training.py tests/test_molcap_integration.py -q
git add dataloader.py tests/test_molcap_training.py tests/test_molcap_integration.py
git commit -m "feat: add deterministic MolCap training identities"
~~~

---

### Task 3: Share the Exact Probe Readout

**Files:**
- Modify: model.py:118-204
- Modify: tests/test_molcap_integration.py

**Interfaces:**
- Produces: PROBE_FEATURE_BLOCKS = (4, 6, 8, 11).
- Extends: DinoV2ViT.forward(x, masks=None, checkpoint=False, feature_blocks=()).
- Adds x_norm_probe_features only when feature_blocks is nonempty.

- [ ] **Step 1: Write the independent readout regression**

~~~python
def test_shared_probe_readout_matches_probe_and_independent_oracle():
    torch.manual_seed(13)
    model = DinoV2ViT(variant_cfg=(8, 12, 2, 2, "mlp", True, "unused", 0)).eval()
    x = torch.randn(3, 3, 28, 28)
    with torch.no_grad():
        xt, expected = model._prepare_tokens(x), []
        for i, block in enumerate(model.blocks):
            xt = block(xt)
            if i in (4, 6, 8, 11):
                expected.append(model.norm(xt)[:, 0])
        expected = torch.cat(expected, dim=-1)
        default = model(x)
        routed = model(x, feature_blocks=(4, 6, 8, 11))
    assert set(default) == {"x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens"}
    assert routed["x_norm_probe_features"].shape == (3, 32)
    torch.testing.assert_close(routed["x_norm_probe_features"], expected, atol=2e-5, rtol=0)
    torch.testing.assert_close(model.probe_features(x), expected, atol=2e-5, rtol=0)
~~~

- [ ] **Step 2: Run and observe TypeError**

~~~powershell
python -m pytest tests/test_molcap_integration.py::test_shared_probe_readout_matches_probe_and_independent_oracle -q
~~~

- [ ] **Step 3: Refactor the existing block loop**

~~~python
PROBE_FEATURE_BLOCKS = (4, 6, 8, 11)

def forward(self, x, masks=None, checkpoint=False, feature_blocks=()):
    x = self._prepare_tokens(x, masks)
    selected = []
    for i, blk in enumerate(self.blocks):
        if checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:
            x = blk(x)
        if i in feature_blocks:
            selected.append(self.norm(x)[:, 0])
    x = self.norm(x)
    output = {
        "x_norm_clstoken": x[:, 0],
        "x_norm_regtokens": x[:, 1 : 1 + self.registers],
        "x_norm_patchtokens": x[:, 1 + self.registers :],
    }
    if feature_blocks:
        assert len(selected) == len(feature_blocks)
        output["x_norm_probe_features"] = torch.cat(selected, dim=-1)
    return output

def probe_features(self, x):
    return self(x, feature_blocks=PROBE_FEATURE_BLOCKS)["x_norm_probe_features"]
~~~

- [ ] **Step 4: Run and commit**

~~~powershell
python -m pytest tests/test_molcap_integration.py tests/test_molcap_training.py -q
git add model.py tests/test_molcap_integration.py
git commit -m "feat: expose exact probe readout during training"
~~~

---

### Task 4: Implement Hierarchy, STE, and Transactional EMA State

**Files:**
- Modify: train.py after update_ema
- Create: tests/test_molcap_centroid.py

**Interfaces:**
- Produces: Hierarchy and CentroidProposal NamedTuples.
- Produces: crop_major_tile_mean, hierarchical_means, patient_targets_from_tiles, teacher_value_student_gradient.
- Produces: HierarchicalCentroidBank.propose and commit.

- [ ] **Step 1: Write failing hierarchy and STE tests**

~~~python
import pytest
import torch

from train import (
    HierarchicalCentroidBank,
    crop_major_tile_mean,
    hierarchical_means,
    patient_targets_from_tiles,
    teacher_value_student_gradient,
)


def test_hierarchy_uses_tile_mean_then_equal_slide_mean_and_is_order_invariant():
    features = torch.tensor([[0.0], [0.0], [6.0], [10.0]])
    slides = torch.tensor([0, 0, 1, 2])
    slide_to_patient = torch.tensor([0, 0, 1])
    first = hierarchical_means(features, slides, slide_to_patient)
    second = hierarchical_means(features.flip(0), slides.flip(0), slide_to_patient)
    torch.testing.assert_close(first.slide_means, torch.tensor([[0.0], [6.0], [10.0]]))
    torch.testing.assert_close(first.patient_means, torch.tensor([[3.0], [10.0]]))
    torch.testing.assert_close(first.patient_means, second.patient_means)
    torch.testing.assert_close(first.slide_tile_counts, torch.tensor([2, 1, 1]))


def test_crop_major_views_restore_tile_identity():
    crop_major = torch.tensor([[1.0], [2.0], [3.0], [5.0]])
    torch.testing.assert_close(
        crop_major_tile_mean(crop_major, views=2, batch_size=2),
        torch.tensor([[2.0], [3.5]]),
    )


def test_teacher_forward_student_identity_gradient():
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[7.0, 11.0]], requires_grad=True)
    routed = teacher_value_student_gradient(student, teacher)
    torch.testing.assert_close(routed, teacher)
    routed.backward(torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(student.grad, torch.tensor([[3.0, 5.0]]))
    assert teacher.grad is None


def test_patient_targets_group_once_and_require_consistency():
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    present = torch.ones(3)
    tile_patients = torch.tensor([0, 0, 1])
    patient_ids = torch.tensor([0, 1])
    grouped, grouped_present = patient_targets_from_tiles(
        targets, present, tile_patients, patient_ids
    )
    torch.testing.assert_close(grouped, torch.eye(2))
    torch.testing.assert_close(grouped_present, torch.ones(2))
~~~

- [ ] **Step 2: Write failing state tests**

Use these exact state transitions:

~~~python
teacher = hierarchical_means(
    torch.tensor([[2.0], [4.0], [10.0]]),
    torch.tensor([0, 0, 1]),
    torch.tensor([0, 0]),
)
bank = HierarchicalCentroidBank(torch.tensor([0, 0]), feature_dim=1, momentum=0.9)
first = bank.propose(teacher)
assert first.base_state_step == 0
torch.testing.assert_close(first.next_slide_centroids, torch.tensor([[3.0], [10.0]]))
assert bank.slide_counts.sum() == 0
bank.commit(first, step=1)
torch.testing.assert_close(bank.slide_counts, torch.tensor([1, 1]))
torch.testing.assert_close(bank.slide_tile_presentations, torch.tensor([2, 1]))
second_teacher = hierarchical_means(
    torch.tensor([[13.0]]), torch.tensor([0]), torch.tensor([0, 0])
)
second = bank.propose(second_teacher)
torch.testing.assert_close(second.next_slide_centroids, torch.tensor([[4.0]]))
torch.testing.assert_close(second.patient_centroids, torch.tensor([[7.0]]))
bank.commit(second, step=2)
with pytest.raises(AssertionError):
    bank.commit(second, step=2)
assert bank.centroid_state_step.item() == 2
~~~

- [ ] **Step 3: Run and confirm ImportError**

~~~powershell
python -m pytest tests/test_molcap_centroid.py -q
~~~

- [ ] **Step 4: Add the pure hierarchy and STE**

~~~python
class Hierarchy(NamedTuple):
    slide_ids: torch.Tensor
    slide_means: torch.Tensor
    slide_tile_counts: torch.Tensor
    patient_ids: torch.Tensor
    patient_means: torch.Tensor


def crop_major_tile_mean(features, views, batch_size):
    assert features.shape[0] == views * batch_size
    return features.reshape(views, batch_size, -1).float().mean(0)


def hierarchical_means(features, slide_ids, slide_to_patient):
    unique_slides, tile_inverse = torch.unique(slide_ids, sorted=True, return_inverse=True)
    tile_counts = torch.bincount(tile_inverse, minlength=len(unique_slides))
    slide_sums = features.new_zeros((len(unique_slides), features.shape[-1]), dtype=torch.float32)
    slide_sums.index_add_(0, tile_inverse, features.float())
    slide_means = slide_sums / tile_counts[:, None]
    slide_patients = slide_to_patient[unique_slides]
    unique_patients, slide_inverse = torch.unique(slide_patients, sorted=True, return_inverse=True)
    patient_sums = slide_means.new_zeros((len(unique_patients), slide_means.shape[-1]))
    patient_sums.index_add_(0, slide_inverse, slide_means)
    patient_counts = torch.bincount(slide_inverse, minlength=len(unique_patients))
    return Hierarchy(unique_slides, slide_means, tile_counts, unique_patients, patient_sums / patient_counts[:, None])


def patient_targets_from_tiles(targets, present, tile_patient_ids, patient_ids):
    inverse = torch.searchsorted(patient_ids, tile_patient_ids)
    assert torch.equal(patient_ids[inverse], tile_patient_ids)
    counts = torch.bincount(inverse, minlength=len(patient_ids))
    grouped = targets.new_zeros((len(patient_ids), targets.shape[-1]))
    grouped.index_add_(0, inverse, targets)
    grouped = grouped / counts[:, None]
    assert torch.allclose(targets, grouped[inverse], atol=1e-6, rtol=0)
    grouped_present = present.new_zeros(len(patient_ids))
    grouped_present.index_add_(0, inverse, present)
    return grouped, grouped_present / counts


def teacher_value_student_gradient(student, teacher):
    return student + (teacher.detach() - student).detach()
~~~

- [ ] **Step 5: Add the bank interface**

~~~python
class CentroidProposal(NamedTuple):
    base_state_step: int
    slide_ids: torch.Tensor
    next_slide_centroids: torch.Tensor
    slide_tile_counts: torch.Tensor
    patient_ids: torch.Tensor
    patient_centroids: torch.Tensor
    drift_cosines: torch.Tensor
    historical_tile_fraction: torch.Tensor


class HierarchicalCentroidBank(nn.Module):
    def __init__(self, slide_to_patient, feature_dim, momentum):
        super().__init__()
        assert momentum == 0.9
        self.momentum = float(momentum)
        self.register_buffer("slide_to_patient", slide_to_patient.long(), persistent=False)
        self.register_buffer("slide_centroids", torch.zeros(len(slide_to_patient), feature_dim))
        self.register_buffer("slide_counts", torch.zeros(len(slide_to_patient), dtype=torch.int64))
        self.register_buffer("slide_tile_presentations", torch.zeros(len(slide_to_patient), dtype=torch.int64))
        self.register_buffer("centroid_state_step", torch.zeros((), dtype=torch.int64))
        patient_count = int(slide_to_patient.max().item()) + 1
        self.register_buffer("patient_sums", torch.zeros(patient_count, feature_dim), persistent=False)
        self.register_buffer("patient_slide_counts", torch.zeros(patient_count, dtype=torch.int64), persistent=False)

    def propose(self, teacher):
        slide_ids = teacher.slide_ids
        old = self.slide_centroids[slide_ids]
        seen = self.slide_counts[slide_ids] > 0
        next_values = torch.where(
            seen[:, None],
            self.momentum * old + (1.0 - self.momentum) * teacher.slide_means.detach(),
            teacher.slide_means.detach(),
        )
        slide_patients = self.slide_to_patient[slide_ids]
        patient_ids, inverse = torch.unique(slide_patients, sorted=True, return_inverse=True)
        sums = self.patient_sums[patient_ids].clone()
        counts = self.patient_slide_counts[patient_ids].clone()
        deltas = next_values - torch.where(seen[:, None], old, torch.zeros_like(old))
        sums.index_add_(0, inverse, deltas)
        counts.index_add_(0, inverse, (~seen).long())
        return CentroidProposal(
            int(self.centroid_state_step),
            slide_ids.detach(),
            next_values.detach(),
            teacher.slide_tile_counts.detach(),
            patient_ids.detach(),
            (sums / counts[:, None]).detach(),
            F.cosine_similarity(old[seen], next_values[seen], dim=-1).detach(),
            (
                teacher.slide_tile_counts[seen].sum()
                / teacher.slide_tile_counts.sum()
            ).detach(),
        )

    @torch.no_grad()
    def commit(self, proposal, step):
        assert proposal.base_state_step == int(self.centroid_state_step)
        assert int(step) == int(self.centroid_state_step) + 1
        slide_ids = proposal.slide_ids
        old = self.slide_centroids[slide_ids].clone()
        seen = self.slide_counts[slide_ids] > 0
        patients = self.slide_to_patient[slide_ids]
        self.slide_centroids[slide_ids] = proposal.next_slide_centroids
        self.slide_counts[slide_ids] += 1
        self.slide_tile_presentations[slide_ids] += proposal.slide_tile_counts
        self.patient_sums.index_add_(
            0,
            patients,
            proposal.next_slide_centroids - torch.where(seen[:, None], old, torch.zeros_like(old)),
        )
        self.patient_slide_counts.index_add_(0, patients, (~seen).long())
        self.centroid_state_step.fill_(step)
~~~

- [ ] **Step 6: Run and commit**

~~~powershell
python -m pytest tests/test_molcap_centroid.py -q
git add train.py tests/test_molcap_centroid.py
git commit -m "feat: add hierarchical MolCap centroid state"
~~~

---

### Task 5: Add Geometry, Maturity, and Strict State Restore

**Files:**
- Modify: train.py near HierarchicalCentroidBank
- Modify: tests/test_molcap_centroid.py

**Interfaces:**
- Produces: centroid_geometry, centroid_audit, require_centroid_gate.
- Produces: export_state and restore_state.

- [ ] **Step 1: Write failing geometry, coverage, and strict-restore tests**

Use an independent NumPy oracle over non-unit rows. Assert centered covariance eigenvalues, entropy effective rank, participation ratio, signed off-diagonal cosine, minimum norm, and row-order invariance. Test all-observed hard population versus mature-only diagnostics. Parameterize failures for every threshold and for every state/metadata/step mismatch.

~~~python
def test_centroid_geometry_matches_nonunit_oracle():
    x = torch.tensor([[2.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    metrics = centroid_geometry(x)
    raw = x.numpy().astype(np.float64)
    centered = raw - raw.mean(0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / 2).clip(0)
    p = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    expected_cosine = ((unit.sum(0) ** 2).sum() - len(unit)) / (len(unit) * (len(unit) - 1))
    assert metrics["effective_rank"] == pytest.approx(np.exp(-(p * np.log(p)).sum()))
    assert metrics["participation_ratio"] == pytest.approx(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())
    assert metrics["mean_offdiag_cosine"] == pytest.approx(expected_cosine)


def test_sample_weighted_coverage_uses_presentations():
    bank = committed_coverage_bank()
    expected = (
        bank.slide_tile_presentations[bank.slide_counts >= 2].sum()
        / bank.slide_tile_presentations.sum()
    )
    assert bank.sample_weighted_mature_coverage(2) == pytest.approx(float(expected))
~~~

- [ ] **Step 2: Run and observe missing APIs**

~~~powershell
python -m pytest tests/test_molcap_centroid.py -q
~~~

- [ ] **Step 3: Implement exact geometry**

~~~python
def centroid_geometry(patient_centroids):
    x = patient_centroids.detach().cpu().double()
    assert x.ndim == 2 and x.shape[0] >= 2 and torch.isfinite(x).all()
    norms = x.norm(dim=1)
    assert torch.all(norms > 0)
    centered = x - x.mean(0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / (x.shape[0] - 1)).clamp_min_(0)
    total = eigenvalues.sum()
    assert total > 0
    p = eigenvalues[eigenvalues > 0] / total
    effective_rank = torch.exp(-(p * p.log()).sum())
    participation = total.square() / eigenvalues.square().sum()
    unit = x / norms[:, None]
    offdiag = (unit.sum(0).square().sum() - x.shape[0]) / (x.shape[0] * (x.shape[0] - 1))
    return {
        "patient_count": int(x.shape[0]),
        "min_norm": float(norms.min()),
        "effective_rank": float(effective_rank),
        "participation_ratio": float(participation),
        "mean_offdiag_cosine": float(offdiag),
    }
~~~

- [ ] **Step 4: Implement audit and hard gate**

Add these exact population methods and gate:

~~~python
def patient_centroids(self, min_slide_updates=1):
    eligible = self.slide_counts >= min_slide_updates
    slide_ids = eligible.nonzero().flatten()
    patients = self.slide_to_patient[slide_ids]
    patient_ids, inverse = torch.unique(patients, sorted=True, return_inverse=True)
    sums = self.slide_centroids.new_zeros((len(patient_ids), self.slide_centroids.shape[-1]))
    sums.index_add_(0, inverse, self.slide_centroids[slide_ids])
    counts = torch.bincount(inverse, minlength=len(patient_ids))
    return patient_ids, sums / counts[:, None]


def sample_weighted_mature_coverage(self, min_slide_updates=2):
    total = self.slide_tile_presentations.sum()
    assert total > 0
    mature = self.slide_tile_presentations[self.slide_counts >= min_slide_updates].sum()
    return float(mature / total)


def centroid_audit(bank, min_slide_updates=2):
    _, observed = bank.patient_centroids(1)
    _, mature = bank.patient_centroids(min_slide_updates)
    return {
        "sample_weighted_mature_coverage": bank.sample_weighted_mature_coverage(min_slide_updates),
        "all_observed": centroid_geometry(observed),
        "mature_only": centroid_geometry(mature),
    }


def require_centroid_gate(audit, history_cfg):
    hard = audit["all_observed"]
    assert audit["sample_weighted_mature_coverage"] >= history_cfg["min_sample_weighted_coverage"]
    assert hard["patient_count"] >= history_cfg["min_geometry_patients"]
    assert hard["effective_rank"] >= history_cfg["min_effective_rank"]
    assert hard["participation_ratio"] >= history_cfg["min_participation_ratio"]
    assert hard["mean_offdiag_cosine"] < history_cfg["max_mean_offdiag_cosine"]
    assert hard["min_norm"] > history_cfg["min_centroid_norm"]
~~~

- [ ] **Step 5: Implement authoritative export/restore**

Use this payload and rebuild rule:

~~~python
def export_state(self, metadata):
    return {
        "metadata": dict(metadata),
        "slide_centroids": self.slide_centroids.detach().cpu().clone(),
        "slide_counts": self.slide_counts.detach().cpu().clone(),
        "slide_tile_presentations": self.slide_tile_presentations.detach().cpu().clone(),
        "centroid_state_step": self.centroid_state_step.detach().cpu().clone(),
    }


@torch.no_grad()
def restore_state(self, payload, expected_metadata, expected_step):
    assert set(payload) == {
        "metadata", "slide_centroids", "slide_counts",
        "slide_tile_presentations", "centroid_state_step",
    }
    assert payload["metadata"] == dict(expected_metadata)
    assert int(payload["centroid_state_step"]) == int(expected_step)
    for name in ("slide_centroids", "slide_counts", "slide_tile_presentations", "centroid_state_step"):
        source = payload[name]
        target = getattr(self, name)
        assert source.shape == target.shape and source.dtype == target.dtype
        target.copy_(source.to(target.device))
    self.patient_sums.zero_()
    self.patient_slide_counts.zero_()
    observed = self.slide_counts > 0
    patients = self.slide_to_patient[observed]
    self.patient_sums.index_add_(0, patients, self.slide_centroids[observed])
    self.patient_slide_counts.index_add_(0, patients, torch.ones_like(patients))
~~~

- [ ] **Step 6: Run and commit**

~~~powershell
python -m pytest tests/test_molcap_centroid.py tests/test_molcap_training.py -q
git add train.py tests/test_molcap_centroid.py
git commit -m "feat: gate and checkpoint MolCap centroid history"
~~~

---

### Task 6: Integrate the Paired Route into Training

**Files:**
- Modify: train.py:9-40, 218-330, 384-515, 620-848
- Modify: tests/test_molcap_integration.py
- Modify: tests/test_molcap_config.py

**Interfaces:**
- Consumes dense identities, x_norm_probe_features, hierarchy/state APIs.
- Produces Arm R/Arm C loss, transactional commit, ramp gate, logs, summaries, and checkpoints.
- Adds runner-only environment control NANOPATH_RUNNER_STOP_AFTER_SAMPLES while schedules retain the one-million denominator.

- [ ] **Step 1: Write failing RNG and mechanics tests**

Add an isolated RNG test that snapshots CPU state, performs the auxiliary train-mode forward, and proves the next random draw is unchanged. Add a CUDA version skipped when CUDA is absent. Add a nonzero-loss mechanics test with matching 12-block student/teacher models that proves:

- crop-major readout restoration;
- Arm R and empty-bank Arm C forward equality;
- finite nonzero head and final-block student gradients;
- no teacher gradient;
- history proposal remains detached until commit.

~~~python
def test_probe_route_head_initialization_is_seed_neutral_at_1536_input():
    torch.manual_seed(7777)
    expected = torch.rand(4)
    torch.manual_seed(7777)
    seed_neutral_molcap_head(1536, 384, "cpu")
    torch.testing.assert_close(torch.rand(4), expected)


def test_auxiliary_forward_restores_cpu_rng():
    torch.manual_seed(19)
    model = DinoV2ViT(variant_cfg=(8, 12, 2, 2, "mlp", True, "unused", 0)).train()
    x = torch.randn(2, 3, 28, 28)
    state = torch.random.get_rng_state()
    with isolated_torch_rng(123, torch.device("cpu")):
        model(x, feature_blocks=(4, 6, 8, 11))
    actual = torch.rand(3)
    torch.random.set_rng_state(state)
    expected = torch.rand(3)
    torch.testing.assert_close(actual, expected)
~~~

The mechanics test snapshots bank state before a validation-style call with MolCap arguments absent and asserts every history buffer is bitwise unchanged afterward.

- [ ] **Step 2: Run and confirm integration failures**

~~~powershell
python -m pytest tests/test_molcap_integration.py -q
~~~

- [ ] **Step 3: Add single-GPU and runner-cap setup**

At main start:

~~~python
assert int(os.environ.get("WORLD_SIZE", "1")) == 1
runner_stop_after_samples = int(
    os.environ.get("NANOPATH_RUNNER_STOP_AFTER_SAMPLES", train_cfg["max_train_samples"])
)
assert 0 < runner_stop_after_samples <= int(train_cfg["max_train_samples"])
assert runner_stop_after_samples == int(train_cfg["max_train_samples"]) or not probe_enabled(cfg)
~~~

All schedules use max_train_samples. Loop termination uses runner_stop_after_samples. Summary retains max_train_samples = 1,000,000, reports runner_stop_after_samples, and uses stop_reason runner_stop_after_samples for capped non-scored work. Full runs have the environment variable absent.

- [ ] **Step 4: Construct head and history after mapping exists**

Use molcap.input_dim for routed head creation. Build train_ds before restoring history. For Arm C, instantiate HierarchicalCentroidBank from train_ds.molcap_slide_to_patient on CUDA. Use exact metadata:

~~~python
history_metadata = {
    "version": 1,
    "arm": "centroid",
    "target_sha256": train_ds.molcap_target_sha256,
    "mapping_digest": train_ds.molcap_mapping_digest,
    "feature_blocks": tuple(molcap_cfg["feature_blocks"]),
    "feature_width": int(molcap_cfg["input_dim"]),
    "momentum": float(molcap_cfg["history"]["momentum"]),
    "hierarchy": molcap_cfg["history"]["level"],
    "ste": molcap_cfg["gradient_source"],
    "weight": float(molcap_cfg["weight"]),
    "ramp_start": float(molcap_cfg["ramp_start"]),
    "ramp_len": float(molcap_cfg["ramp_len"]),
}
~~~

Assert molcap.head_hidden_dim equals 512 because MolCapHead fixes that registered width. Restore model/head/optimizer at the existing point. Restore molcap_history only after dataset construction and assert checkpoint step equality. Arm C resume requires history; Arm R rejects unexpected history.

- [ ] **Step 5: Add the RNG-isolated unmasked student readout**

Complete the existing teacher, masked-global student, and local-student forwards first. Request configured feature blocks from the existing teacher pass. Save CPU/CUDA RNG state, seed the local forward with train seed + 1,000,003 times completed_step, run student_backbone on gf without masks and with feature_blocks, and restore both states in a finally block. Validation supplies no MolCap arguments and performs no auxiliary forward or state mutation.

~~~python
@contextlib.contextmanager
def isolated_torch_rng(seed, device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
~~~

- [ ] **Step 6: Compute paired patient loss**

Average crop-major student/teacher readouts to tiles, build identical hierarchies, assert matching patient IDs, and select one identical target per patient.

Arm R:

~~~python
patient_features = teacher_value_student_gradient(
    student_hierarchy.patient_means,
    teacher_hierarchy.patient_means,
)
pending_history = None
~~~

Arm C:

~~~python
pending_history = centroid_bank.propose(teacher_hierarchy)
assert torch.equal(pending_history.patient_ids, student_hierarchy.patient_ids)
patient_features = teacher_value_student_gradient(
    student_hierarchy.patient_means,
    pending_history.patient_centroids,
)
~~~

Call existing molcap_loss with views=1. Compute the path at scale zero so Arm C warms from sample zero.

~~~python
student_targets, student_present = patient_targets_from_tiles(
    molcap_target,
    molcap_present,
    molcap_patient_idx,
    student_hierarchy.patient_ids,
)
assert torch.all(student_present == 1)
molcap = float(molcap_cfg["weight"]) * molcap_scale * molcap_loss(
    molcap_head, patient_features, student_targets, student_present, views=1
)
~~~

- [ ] **Step 7: Gate before first positive scale**

Immediately before the first Arm C step with positive molcap_scale, audit the committed bank, require all gates, write output_dir/molcap_centroid_ramp_gate.json using strict finite JSON, and mark the in-memory gate passed. Failure raises before nonzero centroid supervision. Capped smoke/calibration end before this boundary.

~~~python
if centroid_bank is not None and molcap_scale > 0 and not centroid_gate_passed:
    centroid_gate_report = centroid_audit(
        centroid_bank, int(molcap_cfg["history"]["min_slide_updates"])
    )
    require_centroid_gate(centroid_gate_report, molcap_cfg["history"])
    assert math.isfinite(float(centroid_gate_report["all_observed"]["effective_rank"]))
    (output_dir / "molcap_centroid_ramp_gate.json").write_text(
        json.dumps(centroid_gate_report, allow_nan=False, indent=2) + "\n"
    )
    centroid_gate_passed = True
~~~

- [ ] **Step 8: Make optimizer and bank commit transactional**

Assert finite total loss before backward and all non-None optimized gradients finite afterward. Run optimizer.step. Commit pending history only after it returns, require centroid_state_step equals completed_step, then perform the existing backbone/DINO-head EMA updates.

~~~python
assert torch.isfinite(total_loss)
total_loss.backward()
optimized = [p for group in opt.param_groups for p in group["params"] if p.grad is not None]
assert all(torch.isfinite(p.grad).all() for p in optimized)
grad_norm = nn.utils.clip_grad_norm_(clipped, dino_cfg["clip_grad"])
assert torch.isfinite(grad_norm)
opt.step()
if pending_history is not None:
    centroid_bank.commit(pending_history, completed_step)
    assert int(centroid_bank.centroid_state_step) == completed_step
m = cosine_schedule(0.994, 1.0, reg_frac)
with torch.no_grad():
    update_ema(student_backbone, teacher_backbone, m)
    update_ema(student_dino_head, teacher_dino_head, m)
~~~

- [ ] **Step 9: Extend full checkpoints only**

Keep the full=False early return unchanged. For full Arm C payloads:

~~~python
if full and centroid_bank is not None:
    payload["molcap_history"] = centroid_bank.export_state(history_metadata)
~~~

Test that probe payloads contain neither molcap_head nor molcap_history.

- [ ] **Step 10: Add pairing and state diagnostics**

Hash the first 8,192 sample_idx values in presentation order using int64 little-endian bytes. Log and summarize sample-order digest/count, target/mapping/source hashes, train patient/slide counts, bank bytes, current groups, target coverage, gradient cosine/norm ratio, maturity coverage, update quantiles, nonhistorical fraction, centroid-caption cosine, teacher drift, geometry, runner cap, and peak memory. Diagnostics add no stochastic forward or mutation.

~~~python
train_log.update({
    "molcap_unique_patients": int(student_hierarchy.patient_ids.numel()),
    "molcap_current_slides": int(student_hierarchy.slide_ids.numel()),
    "molcap_mapping_digest": train_ds.molcap_mapping_digest,
    "molcap_target_sha256": train_ds.molcap_target_sha256,
    "molcap_sample_order_digest": sample_order_hasher.hexdigest(),
    "molcap_sample_order_count": sample_order_count,
    "molcap_history_enabled": centroid_bank is not None,
    "molcap_history_state_step": int(centroid_bank.centroid_state_step) if centroid_bank is not None else 0,
    "molcap_historical_tile_fraction": (
        float(pending_history.historical_tile_fraction) if pending_history is not None else 0.0
    ),
    "molcap_teacher_drift_mean": (
        float(pending_history.drift_cosines.mean())
        if pending_history is not None and pending_history.drift_cosines.numel() else 1.0
    ),
})
~~~

- [ ] **Step 11: Run and commit**

~~~powershell
python -m pytest tests/test_molcap_centroid.py tests/test_molcap_integration.py tests/test_molcap_config.py -q
git add train.py tests/test_molcap_centroid.py tests/test_molcap_integration.py tests/test_molcap_config.py
git commit -m "feat: integrate paired MolCap centroid training"
~~~

---

### Task 7: Lock Source Policy and Run Full Local Verification

**Files:**
- Modify: tests/test_molcap_config.py
- Verify unchanged: probe.py
- Verify unchanged: benchmarking/

**Interfaces:**
- Produces locked-path regression anchored to 01c1cdf and a clean committed source.

- [ ] **Step 1: Add locked-path regression**

For probe.py and every tracked path returned by git ls-tree -r --name-only 01c1cdf -- benchmarking, compare local bytes with git show 01c1cdf:path. Also assert source wiring tokens for history state, ramp gate, WORLD_SIZE, and shared feature blocks.

~~~python
import subprocess


def git_bytes(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"])


def test_locked_probe_and_benchmarking_match_preexperiment_commit():
    revision = "01c1cdf8017a0481636a28ab58a0ddc67d6e0a06"
    paths = ["probe.py"] + subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "--", "benchmarking/"],
        text=True,
    ).splitlines()
    for path in paths:
        assert Path(path).read_bytes() == git_bytes(revision, path)
~~~

- [ ] **Step 2: Run all verification**

~~~powershell
python -m pytest -q
python -m py_compile dataloader.py model.py train.py
git diff --check
git diff --exit-code 01c1cdf8017a0481636a28ab58a0ddc67d6e0a06 HEAD -- probe.py benchmarking/
~~~

Expected: all pass and locked diff is empty.

- [ ] **Step 3: Review the complete implementation**

~~~powershell
git diff --stat 70b9afe62cf9b888880617964719f02be98a3fb2..HEAD
git diff 70b9afe62cf9b888880617964719f02be98a3fb2..HEAD -- dataloader.py model.py train.py configs tests
~~~

Check default forward keys, strict route pairing, no target-builder change, and no validation mutation.

- [ ] **Step 4: Commit the policy test if changed**

~~~powershell
git add tests/test_molcap_config.py
git commit -m "test: lock MolCap centroid submission surface"
~~~

Do not create an empty commit.

---

### Task 8: Build the Ignored Modal Harness

**Files:**
- Reuse as reference: .superpowers/sdd/modal_molcap_biomed_pca384.py
- Create locally, keep ignored: .superpowers/sdd/modal_molcap_centroid.py
- Create locally, keep ignored: .superpowers/sdd/molcap-centroid-calibration.json

**Interfaces:**
- Produces smoke, calibration, and full actions for both arms.
- Provides H200 then exact-H100 full fallbacks only after B200 infrastructure failure.

- [ ] **Step 1: Copy proven image/workspace/volume setup**

Use Python 3.12, CUDA 12.9, uv sync, volume nanopath-readout-local-context at /data, dataset /data/repo-data/nanopath_parquet, target /data/repo-data/molcap_text_384.npz, and run root /data/experiments/readout-local-context.

- [ ] **Step 2: Add source and target preflight**

Require supplied committed HEAD, clean mount, exact target SHA/keys/shape/identities/unit rows/mode, and probe.py/benchmarking hashes matching 01c1cdf before each action.

~~~python
TARGET_SHA = "2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577"
LOCKED_COMMIT = "01c1cdf8017a0481636a28ab58a0ddc67d6e0a06"


def require_canonical_target(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TARGET_SHA
    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"patient_ids", "targets", "captions", "mode"}
        patient_ids = tuple(str(x) for x in artifact["patient_ids"])
        targets = artifact["targets"]
        assert str(artifact["mode"]) == "text"
    assert len(patient_ids) == len(set(patient_ids)) == 11_428
    assert targets.shape == (11_428, 384)
    assert np.isfinite(targets).all()
    assert np.max(np.abs(np.linalg.norm(targets, axis=1) - 1.0)) <= 1e-5
~~~

- [ ] **Step 3: Generate hidden non-scored configs**

Smoke preserves the one-million denominator, disables probes, uses batch 8/two locals, and sets runner cap 1,024. Calibration changes only output and probe.enabled=false, retains batch 128/all views/precision/objectives, and sets cap 32,768.

~~~python
ACTIONS = {
    "smoke-route-b200": ("route", "B200", 1_024, False),
    "smoke-centroid-b200": ("centroid", "B200", 1_024, False),
    "calibrate-route-b200": ("route", "B200", 32_768, False),
    "calibrate-centroid-b200": ("centroid", "B200", 32_768, False),
    "calibrate-route-h100": ("route", "H100!", 32_768, False),
    "calibrate-centroid-h100": ("centroid", "H100!", 32_768, False),
    "full-route-b200": ("route", "B200", 1_000_000, True),
    "full-centroid-b200": ("centroid", "B200", 1_000_000, True),
}


def runner_environment(sample_cap, full):
    env = {"NANOPATH_SOURCE_COMMIT": SOURCE_COMMIT}
    if not full:
        env["NANOPATH_RUNNER_STOP_AFTER_SAMPLES"] = str(sample_cap)
    return env
~~~

- [ ] **Step 4: Add nonzero CUDA mechanics and integrated smoke**

Each smoke first runs the CUDA mechanics test with nonzero local loss, then capped integrated training. Require finite student/head gradients, no teacher gradient, Arm R/first-observation C equality, full target coverage, exact mapping, finite bank, and no ramp gate.

- [ ] **Step 5: Add paired calibration extraction**

Require 256 batch-128 steps. Compute median logged FLOP/s over steps 60 through 256. Record GPU, versions, wall time, visible patches/s, FLOP/s, peak memory, hashes, route, and sample digest.

~~~python
def steady_calibration(metrics_path):
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    rows = [row for row in rows if 60 <= int(row.get("step", -1)) <= 256 and "flops_per_sec" in row]
    assert rows and max(row["examples_seen"] for row in rows) == 32_768
    return {
        "median_flops_per_sec": statistics.median(row["flops_per_sec"] for row in rows),
        "peak_memory_gib": max(row["gpu_peak_mem_gb"] for row in rows),
        "sample_order_digest": rows[-1]["molcap_sample_order_digest"],
    }
~~~

- [ ] **Step 6: Add full actions**

Use committed scored configs with runner cap absent and fresh outputs. Require 7,812 steps, 999,936 presentations, max_train_samples stop, complete final probe/checkpoint/source, exact hashes/coverage/mapping, and paired diagnostics. Arm C requires passed ramp gate and final bank state step.

- [ ] **Step 7: Compile the ignored harness**

~~~powershell
$env:PYTHONUTF8='1'
python -m py_compile .superpowers/sdd/modal_molcap_centroid.py
modal run .superpowers/sdd/modal_molcap_centroid.py --help
git check-ignore -v .superpowers/sdd/modal_molcap_centroid.py
~~~

---

### Task 9: Run B200 Smokes and B200/H100 Calibration

**Files:**
- Local ignored evidence: .superpowers/sdd/molcap-centroid-calibration.json

**Interfaces:**
- Produces mechanics, compatibility, timing, memory, and pairing evidence.

- [ ] **Step 1: Run fresh preflight**

~~~powershell
python -m pytest -q
python -m py_compile dataloader.py model.py train.py .superpowers/sdd/modal_molcap_centroid.py
git status --short
git diff --check
git diff --exit-code 01c1cdf8017a0481636a28ab58a0ddc67d6e0a06 HEAD -- probe.py benchmarking/
~~~

- [ ] **Step 2: Run both B200 smokes**

~~~powershell
modal run .superpowers/sdd/modal_molcap_centroid.py --action smoke-route-b200
modal run .superpowers/sdd/modal_molcap_centroid.py --action smoke-centroid-b200
~~~

- [ ] **Step 3: Run all four calibrations**

~~~powershell
modal run .superpowers/sdd/modal_molcap_centroid.py --action calibrate-route-b200
modal run .superpowers/sdd/modal_molcap_centroid.py --action calibrate-centroid-b200
modal run .superpowers/sdd/modal_molcap_centroid.py --action calibrate-route-h100
modal run .superpowers/sdd/modal_molcap_centroid.py --action calibrate-centroid-h100
~~~

- [ ] **Step 4: Certify pairing and resource bounds**

Require route/centroid sample-order digests to match per hardware and Arm C minus Arm R peak memory no more than 0.5 GiB. Persist strict JSON with both arms/GPUs, versions, throughput, memory, hashes, sample digests, and:

~~~python
projected_h100_seconds = observed_b200_seconds * (
    median_b200_flops_per_second / median_h100_flops_per_second
)
margin_seconds = 7200.0 - projected_h100_seconds
~~~

Do not commit the ignored evidence.

---

### Task 10: Run, Verify, and Submit Arm R

**Files:**
- Remote: /data/experiments/readout-local-context/full/molcap-probe-route-s7777
- Local staged directory outside repository.

**Interfaces:**
- Produces completed route artifacts, probes, timing, and Labless submission.

- [ ] **Step 1: Launch full B200 route**

~~~powershell
modal run .superpowers/sdd/modal_molcap_centroid.py --action full-route-b200
~~~

Use H200 then exact H100 only after recorded B200 infrastructure failure.

- [ ] **Step 2: Retrieve lightweight evidence**

Retrieve summary.json, metrics.jsonl, modal_result.json, and labless_source. Do not stage checkpoint or target.

- [ ] **Step 3: Verify completion**

Require 999,936 presentations, 7,812 steps, max_train_samples stop, complete probe, exact hashes/mapping/source, 100% coverage, finite route diagnostics, and locked source bytes. Recompute H100-equivalent time from observed B200 train seconds.

- [ ] **Step 4: Dry-run Labless**

~~~powershell
python .\labless\submit_to_labless.py "output_dir=$ROUTE_RUN_DIR" "run_name=molcap-route-s7777" "notes=Probe-CLS hierarchical MolCap route control with current-teacher forward and identity student STE." "review_config=configs/molcap-probe-route-s7777.yaml" "hardware=$HARDWARE" "dry_run=true"
~~~

- [ ] **Step 5: Submit completed Arm R**

Repeat without dry_run=true, complete device authentication, and save submission ID/URL/response/metric/commit. Submit regardless of score.

---

### Task 11: Run, Verify, and Submit Arm C

**Files:**
- Remote: /data/experiments/readout-local-context/full/molcap-ema-centroid-s7777
- Local staged directory outside repository.

**Interfaces:**
- Produces completed centroid artifacts or durable pre-supervision gate failure.

- [ ] **Step 1: Launch Arm C regardless of Arm R score**

~~~powershell
modal run .superpowers/sdd/modal_molcap_centroid.py --action full-centroid-b200
~~~

- [ ] **Step 2: Inspect 50% boundary**

Require strict ramp-gate evidence before nonzero centroid supervision. On failure, preserve report/logs, change no threshold/momentum, and do not submit an incomplete run. On pass, require full continuation.

- [ ] **Step 3: Retrieve and verify completion**

Require full steps/samples/probe, passed gate, final bank step 7,812, exact hashes/mapping/source, full coverage, finite diagnostics, locked bytes, timing projection, and memory bound.

- [ ] **Step 4: Dry-run Labless**

~~~powershell
python .\labless\submit_to_labless.py "output_dir=$CENTROID_RUN_DIR" "run_name=molcap-cent-s7777" "notes=Paired hierarchical slide-to-patient EMA MolCap arm with historical-teacher forward and identity student STE." "review_config=configs/molcap-ema-centroid-s7777.yaml" "hardware=$HARDWARE" "dry_run=true"
~~~

- [ ] **Step 5: Submit completed Arm C**

Repeat without dry_run=true and save submission evidence. Submit regardless of score.

---

### Task 12: Apply Decisions and Record Results

**Files:**
- Create: docs/results/2026-07-12-molcap-probe-route-ema-centroid-s7777.md

**Interfaces:**
- Consumes both full records or Arm C gate failure, MiniLM result, and frontier.
- Produces mechanism decision, route interpretation, timing audit, and next action.

- [ ] **Step 1: Build four-column metric table**

Report all eight categories and overall for Arm R, Arm C, molcap-text-s7777, and bsc-s7777-k10. Include C-minus-R deltas and progression/mutation/survival mean.

- [ ] **Step 2: Apply primary rule exactly**

Arm C supports the mechanism only if progression, mutation, and survival each improve; their mean improves at least 0.003; and linear, kNN, few-shot each decline less than 0.003.

- [ ] **Step 3: Apply promotion independently**

Promotion requires overall at least 0.6719107210 and linear/kNN each declining less than 0.003 versus seed-matched bsc-s7777-k10.

- [ ] **Step 4: Record provenance**

Include hashes, config diff, mapping/bank, readout/RNG evidence, smoke/gate/calibration, timing/memory, submissions, and the registered STE caveat.

- [ ] **Step 5: Verify and commit report**

~~~powershell
python -m pytest -q
python -m py_compile dataloader.py model.py train.py
git diff --check
git diff --exit-code 01c1cdf8017a0481636a28ab58a0ddc67d6e0a06 HEAD -- probe.py benchmarking/
git add docs/results/2026-07-12-molcap-probe-route-ema-centroid-s7777.md
git commit -m "docs: record paired MolCap centroid results"
~~~

---

### Task 13: Conditional Two-Seed Promotion

**Files:**
- Create only on promotion: seed-7778 and seed-7779 copies of promoted config.
- Extend paired result report.

**Interfaces:**
- Runs only after promoted seed-7777 submission.
- Produces three-seed median without tuning objective constants.

- [ ] **Step 1: Freeze seeds 7778 and 7779**

Change only project labels/output plus train.seed and data.split_seed. Keep model, loss, history, ramp, crops, probes, and hardware unchanged.

~~~yaml
# Seed 7778 leaves
data:
  split_seed: 7778
train:
  seed: 7778

# Seed 7779 leaves
data:
  split_seed: 7779
train:
  seed: 7779
~~~

The config test must compare each promoted config with seed 7777 and allow only project.name, project.recipe_id, project.output_dir, data.split_seed, and train.seed.

- [ ] **Step 2: Run and submit both completions**

Verify full contract and submit each completed seed regardless of score.

- [ ] **Step 3: Report three-seed median**

Add per-seed metrics, median overall/component deltas, timing, and submission IDs. Do not sweep momentum, weight, ramp, head, sampler, or pooling.
