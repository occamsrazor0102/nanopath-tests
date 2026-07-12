import hashlib
import json
import sys
import types

import numpy as np
import pytest

import reembed_molcap_targets as reembed
from build_molcap_targets import isotropize, save_target_bank
from reembed_molcap_targets import (
    BIOMED_DIM,
    BIOMED_MODEL,
    BIOMED_REVISION,
    ISOTROPY_FLOOR,
    ISOTROPY_POWER,
    MINILM_MODEL,
    MINILM_REVISION,
    geometry_metrics,
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


def test_fino_patient_ids_unions_every_mapping(tmp_path):
    path = tmp_path / "fino.json"
    path.write_text(json.dumps({"discrete": {"a": {"P1": 1}}, "continuous": {"b": {"P2": 0.2}}}))
    assert reembed.fino_patient_ids(path) == {"P1", "P2"}


def valid_geometry():
    return {
        "rows": 2,
        "width": 768,
        "mean_off_diagonal_cosine": 0.0,
        "effective_rank": 40.0,
        "participation_ratio": 20.0,
        "normalized_effective_rank": 0.052,
        "normalized_participation_ratio": 0.03,
        "variance_cv": 0.3,
        "max_unit_norm_error": 1e-7,
    }


def test_validate_candidate_accepts_valid_geometry_and_coverage():
    reference = {"normalized_effective_rank": 0.10, "normalized_participation_ratio": 0.06}
    report = reembed.validate_candidate(reference, valid_geometry(), np.array(["P1", "P2"]), {"P1", "P2"})
    assert report["coverage_fraction"] == 1.0
    assert report["normalized_effective_rank_ratio"] == 0.52
    assert report["normalized_participation_ratio_ratio"] == 0.5


@pytest.mark.parametrize(
    ("mutation", "patient_ids", "fino_ids", "gate"),
    [
        ({"rows": 1}, np.array(["P1", "P2"]), {"P1", "P2"}, "row count"),
        ({}, np.array(["P1", "P1"]), {"P1"}, "unique patient"),
        ({"width": 767}, np.array(["P1", "P2"]), {"P1", "P2"}, "width"),
        ({"max_unit_norm_error": 2e-5}, np.array(["P1", "P2"]), {"P1", "P2"}, "unit norm"),
        ({"mean_off_diagonal_cosine": 0.02}, np.array(["P1", "P2"]), {"P1", "P2"}, "cosine"),
        ({"effective_rank": 31.0}, np.array(["P1", "P2"]), {"P1", "P2"}, "effective rank"),
        ({"participation_ratio": 15.0}, np.array(["P1", "P2"]), {"P1", "P2"}, "participation"),
        ({"variance_cv": 0.8}, np.array(["P1", "P2"]), {"P1", "P2"}, "variance CV"),
        ({"normalized_effective_rank": 0.049}, np.array(["P1", "P2"]), {"P1", "P2"}, "effective rank ratio"),
        ({"normalized_effective_rank": 0.201}, np.array(["P1", "P2"]), {"P1", "P2"}, "effective rank ratio"),
        ({"normalized_participation_ratio": 0.029}, np.array(["P1", "P2"]), {"P1", "P2"}, "participation ratio ratio"),
        ({"normalized_participation_ratio": 0.121}, np.array(["P1", "P2"]), {"P1", "P2"}, "participation ratio ratio"),
        ({"variance_cv": np.nan}, np.array(["P1", "P2"]), {"P1", "P2"}, "finite"),
        ({}, np.array(["P1", "P2"]), {"P1", "P3"}, "FINO coverage"),
    ],
)
def test_validate_candidate_rejects_each_hard_gate(mutation, patient_ids, fino_ids, gate):
    reference = {"normalized_effective_rank": 0.10, "normalized_participation_ratio": 0.06}
    candidate = valid_geometry() | mutation
    with pytest.raises(AssertionError, match=gate):
        reembed.validate_candidate(reference, candidate, patient_ids, fino_ids)


class FakeEncoder:
    def __init__(self, raw, revision):
        self.raw = raw
        self.revision = revision
        self.calls = []

    def encode(self, captions, **kwargs):
        self.calls.append({"captions": list(captions), "revision": self.revision, **kwargs})
        return self.raw.copy()


def test_build_reembedded_bank_copies_canonical_rows_and_writes_deterministically(tmp_path, monkeypatch):
    rows, dim = 128, 40
    patient_ids = np.array([f"P{i:03d}" for i in range(rows)])
    captions = np.array([f"caption {i}" for i in range(rows)])
    minilm_raw = np.random.default_rng(11).normal(size=(rows, dim)).astype(np.float32)
    biomedical_raw = np.random.default_rng(29).normal(size=(rows, dim)).astype(np.float32)
    minilm_raw /= np.linalg.norm(minilm_raw, axis=1, keepdims=True)
    biomedical_raw /= np.linalg.norm(biomedical_raw, axis=1, keepdims=True)
    source = tmp_path / "canonical.npz"
    save_target_bank(source, patient_ids, isotropize(minilm_raw), captions, "text")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    fino = tmp_path / "fino.json"
    fino.write_text(json.dumps({"discrete": {"all": dict.fromkeys(patient_ids.tolist(), 1)}, "continuous": {}}))
    monkeypatch.setattr(reembed, "CANONICAL_ROWS", rows)
    monkeypatch.setattr(reembed, "BIOMED_DIM", dim)
    minilm = FakeEncoder(minilm_raw, MINILM_REVISION)
    biomedical = FakeEncoder(biomedical_raw, BIOMED_REVISION)
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    report, second_report = tmp_path / "first.json", tmp_path / "second.json"

    returned = reembed.build_reembedded_bank(source, first, report, fino, minilm, biomedical, source_sha)
    reembed.build_reembedded_bank(source, second, second_report, fino, minilm, biomedical, source_sha)

    with np.load(first, allow_pickle=False) as bank:
        first_ids = bank["patient_ids"]
        first_captions = bank["captions"]
        assert bank["mode"].item() == "biomedical"
        assert bank["targets"].shape == (rows, dim)
    assert first_ids.tolist() == patient_ids.tolist()
    assert first_captions.tolist() == captions.tolist()
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    assert report.read_bytes() == second_report.read_bytes()
    assert json.loads(report.read_text())["models"]["biomedical"]["revision"] == BIOMED_REVISION
    assert returned == json.loads(report.read_text())
    expected_call = {
        "captions": captions.tolist(),
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "batch_size": 64,
        "convert_to_numpy": True,
    }
    assert minilm.calls == [{**expected_call, "revision": MINILM_REVISION}] * 2
    assert biomedical.calls == [{**expected_call, "revision": BIOMED_REVISION}] * 2


def test_build_reembedded_bank_rejects_wrong_hash_before_loading(tmp_path):
    source = tmp_path / "canonical.npz"
    source.write_bytes(b"not an npz")
    with pytest.raises(AssertionError, match="source SHA-256"):
        reembed.build_reembedded_bank(source, tmp_path / "out.npz", tmp_path / "report.json", tmp_path / "fino.json", None, None, "wrong")


def test_build_reembedded_bank_requires_exact_keys(tmp_path):
    source = tmp_path / "canonical.npz"
    np.savez(source, patient_ids=["P1"], targets=[[1.0]], captions=["caption"], mode="text", extra=[1])
    with pytest.raises(AssertionError, match="canonical keys"):
        reembed.build_reembedded_bank(source, tmp_path / "out.npz", tmp_path / "report.json", tmp_path / "fino.json", None, None, None)


def test_build_reembedded_bank_requires_canonical_row_count(tmp_path):
    source = tmp_path / "canonical.npz"
    save_target_bank(source, ["P1"], [[1.0]], ["caption"], "text")
    with pytest.raises(AssertionError, match="canonical row count"):
        reembed.build_reembedded_bank(source, tmp_path / "out.npz", tmp_path / "report.json", tmp_path / "fino.json", None, None, None)


def test_cli_pins_revisions_and_stays_offline(tmp_path, monkeypatch):
    constructed, build_args = [], []

    class FakeSentenceTransformer:
        def __init__(self, model, **kwargs):
            constructed.append((model, kwargs))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    monkeypatch.setattr(reembed, "build_reembedded_bank", lambda *args: build_args.append(args) or {"ok": True})
    argv = [
        f"source={tmp_path / 'source.npz'}",
        f"output={tmp_path / 'output.npz'}",
        f"report={tmp_path / 'report.json'}",
        f"fino={tmp_path / 'fino.json'}",
        "device=cpu",
    ]

    assert reembed.main(argv) == {"ok": True}
    assert constructed == [
        (MINILM_MODEL, {"revision": MINILM_REVISION, "device": "cpu", "local_files_only": True}),
        (BIOMED_MODEL, {"revision": BIOMED_REVISION, "device": "cpu", "local_files_only": True}),
    ]
    assert build_args[0][-1] == reembed.CANONICAL_SHA256
