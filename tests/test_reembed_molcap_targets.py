# Tests for reembed_molcap_targets.py — the parts runnable without the pinned encoders or the
# canonical NPZ: geometry/isotropy math, deterministic serialization, every hard gate's accept/reject
# behavior, and the failure-path hygiene (publish-on-pass / clear-on-fail, strict-JSON non-finite).
# The model-dependent stages (actual encode, MiniLM reproduction) run in the fork's environment.
# Run: `python tests/test_reembed_molcap_targets.py` (or `pytest tests/`). Excluded from the snapshot.

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reembed_molcap_targets as R


def test_effective_rank_and_participation_handvalues():
    assert abs(R._effective_rank(np.ones(4)) - 4) < 1e-9
    assert abs(R._participation_ratio(np.ones(4)) - 4) < 1e-9
    assert R._effective_rank(np.array([1.0, 0, 0, 0])) == 1.0
    assert R._participation_ratio(np.array([1.0, 0, 0, 0])) == 1.0
    assert abs(R._effective_rank(np.array([3.0, 3.0, 0, 0])) - 2) < 1e-9
    assert abs(R._participation_ratio(np.array([3.0, 3.0, 0, 0])) - 2) < 1e-9


def test_mean_offdiag_cosine_handvalue():
    rows = R._l2(np.array([[1.0, 0], [0.0, 1.0], [1.0, 0.0]]))
    assert abs(R._mean_offdiag_cosine(rows) - 1 / 3) < 1e-9


def test_isotropy_unitnorm_and_decorrelates():
    rng = np.random.default_rng(1)
    aniso = rng.normal(size=(800, 48)) * (np.arange(1, 49) ** 2.0)
    corr = R.fit_isotropy(aniso)
    assert np.abs(np.linalg.norm(corr, axis=1) - 1).max() < 1e-5
    assert R.geometry(corr)["effective_rank"] > R.geometry(aniso)["effective_rank"]
    assert np.array_equal(corr, R.fit_isotropy(aniso))


def test_deterministic_savez_hash_stable():
    rng = np.random.default_rng(2)
    d = {"patient_ids": np.array(["A", "B", "C"]), "captions": np.array(["x", "y", "z"]),
         "targets": R._l2(rng.normal(size=(3, 8))).astype(np.float32)}
    R.deterministic_savez("/tmp/_d1.npz", d)
    R.deterministic_savez("/tmp/_d2.npz", d)
    assert R._sha256("/tmp/_d1.npz") == R._sha256("/tmp/_d2.npz")


# ---- gate fixtures: minimal inputs that PASS, then one mutation per gate that must FAIL ----------
def _passing(tmp="/tmp/_gate_pass.npz"):
    n, w = R.CANON_ROWS, R.BIOMED_WIDTH
    ids = np.array([f"TCGA-{i:06d}" for i in range(n)])
    captions = np.array([f"cap {i}" for i in range(n)])
    corrected = np.zeros((n, w), np.float32); corrected[:, 0] = 1.0    # unit-norm, finite (geom passed explicitly)
    geom = {"mean_offdiag_cosine": 0.0008, "effective_rank": 74.0, "norm_effective_rank": 74.0 / w,
            "participation_ratio": 45.0, "norm_participation_ratio": 45.0 / w, "var_cv": 0.30,
            "max_unit_norm_error": 1e-8}
    tile_ids = ids[:R.TILE_PATIENTS]
    R.deterministic_savez(tmp, {"patient_ids": ids, "captions": captions, "targets": corrected})
    return dict(ids=ids, captions=captions, canon_ids=ids, canon_captions=captions,
                corrected=corrected, geom=geom, tile_ids=tile_ids, staging=tmp)


def _verdict(target_width=None, **over):
    a = _passing(); a.update(over)
    tw = R.BIOMED_WIDTH if target_width is None else target_width
    _, passed, first_failed = R.evaluate_gates(a["ids"], a["captions"], a["canon_ids"], a["canon_captions"],
                                               a["corrected"], a["geom"], a["tile_ids"], tw, a["staging"])
    return passed, first_failed


def test_gates_accept_valid_fixture():
    passed, first_failed = _verdict()
    assert passed and first_failed is None


def test_gates_reject_each_failure_mode_by_name():
    n, w = R.CANON_ROWS, R.BIOMED_WIDTH
    base = _passing()["geom"]
    cases = {
        "ids_match_canonical": dict(canon_ids=np.array([f"X-{i}" for i in range(n)])),
        "captions_match_canonical": dict(canon_captions=np.array([f"other {i}" for i in range(n)])),
        "width": dict(corrected=np.zeros((n, 384), np.float32)),
        "fino_coverage_missing": dict(tile_ids=np.array(["TCGA-999999-MISSING"])),
        "mean_offdiag_cosine": dict(geom={**base, "mean_offdiag_cosine": 0.05}),
        "effective_rank": dict(geom={**base, "effective_rank": 20.0}),
        "participation_ratio": dict(geom={**base, "participation_ratio": 10.0}),
        "var_cv": dict(geom={**base, "var_cv": 0.9}),
        "norm_effrank_ratio": dict(geom={**base, "effective_rank": 600.0, "norm_effective_rank": 600.0 / w}),
    }
    for expected_gate, over in cases.items():
        passed, first_failed = _verdict(**over)
        assert not passed and first_failed == expected_gate, f"{expected_gate}: got {first_failed}"


# Reproduces the real S-PubMedBERT abort: absolute geometry passes, but the width-normalized ratio
# gate fails (0.45 < 0.5) purely because the encoder is 768-d vs MiniLM's 384-d.
def test_gates_reproduce_biomed_width_confounded_failure():
    base = _passing()["geom"]
    geom = {**base, "effective_rank": 33.19, "norm_effective_rank": 33.19 / 768,
            "participation_ratio": 19.35, "norm_participation_ratio": 19.35 / 768}
    passed, first_failed = _verdict(geom=geom)
    assert not passed and first_failed == "norm_effrank_ratio"


# Width-controlled A/B: the SAME biomedical absolute geometry that failed the ratio gate at width 768
# (eff rank 33.19, participation 19.35) now PASSES once width-matched to 384 (ratio ~0.90 vs ~0.45).
def test_gates_width_matched_pass():
    n = R.CANON_ROWS
    ids = np.array([f"TCGA-{i:06d}" for i in range(n)]); captions = np.array([f"cap {i}" for i in range(n)])
    corrected = np.zeros((n, 384), np.float32); corrected[:, 0] = 1.0
    geom = {"mean_offdiag_cosine": 0.0007, "effective_rank": 33.19, "norm_effective_rank": 33.19 / 384,
            "participation_ratio": 19.35, "norm_participation_ratio": 19.35 / 384, "var_cv": 0.46,
            "max_unit_norm_error": 1e-8}
    R.deterministic_savez("/tmp/_wm.npz", {"patient_ids": ids, "captions": captions, "targets": corrected})
    _, passed, first_failed = R.evaluate_gates(ids, captions, ids, captions, corrected, geom,
                                               ids[:R.TILE_PATIENTS], 384, "/tmp/_wm.npz")
    assert passed and first_failed is None


def test_pca_reduce_shape_variance_determinism():
    rng = np.random.default_rng(7)
    lowrank = rng.normal(size=(400, 6)) @ rng.normal(size=(6, 64))   # rank-6 signal embedded in 64-d
    red, ret = R.pca_reduce(lowrank, 10)
    assert red.shape == (400, 10) and ret > 0.999                    # top-10 keeps the rank-6 signal
    assert np.allclose(red, R.pca_reduce(lowrank, 10)[0])            # deterministic
    assert R.pca_reduce(lowrank, 3)[1] < 0.95                        # under-reducing loses variance


def test_publish_on_pass_and_clear_on_fail():
    staging, final = "/tmp/_pub_staging.npz", "/tmp/_pub_final.npz"
    for p in (staging, final):
        if os.path.exists(p): os.remove(p)
    # pass: staging is published to final
    open(staging, "wb").write(b"NEW")
    published, stale = R.publish_or_clear(True, staging, final)
    assert published and not stale and os.path.exists(final) and not os.path.exists(staging)
    # fail with a stale target present: both cleared, stale flagged
    open(staging, "wb").write(b"NEW2"); open(final, "wb").write(b"STALE")
    published, stale = R.publish_or_clear(False, staging, final)
    assert not published and stale and not os.path.exists(final) and not os.path.exists(staging)
    # fail with no stale target: staging cleared, stale=False
    open(staging, "wb").write(b"NEW3")
    published, stale = R.publish_or_clear(False, staging, final)
    assert not published and not stale and not os.path.exists(staging)


def test_json_safe_replaces_nonfinite_and_serializes_strict():
    nf = []
    safe = R.json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2.0}}, nf)
    assert safe["a"] is None and safe["b"][1] is None and safe["c"]["d"] == 2.0
    assert set(nf) == {".a", ".b[1]"}
    json.dumps(safe, allow_nan=False)   # must not raise


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
