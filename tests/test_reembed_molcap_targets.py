# Tests for reembed_molcap_targets.py — the parts runnable without the pinned encoders or the
# canonical NPZ: geometry/isotropy math, deterministic serialization, and every hard gate's
# accept/reject behavior. The model-dependent stages (actual encode, MiniLM reproduction, coverage
# vs the 9,389 tile patients) run in the fork's environment. Run: `python tests/test_reembed_molcap_targets.py`
# (or `pytest tests/`). Excluded from the Labless training source snapshot.

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
    # two equal + two zero eigenvalues -> both metrics = 2
    assert abs(R._effective_rank(np.array([3.0, 3.0, 0, 0])) - 2) < 1e-9
    assert abs(R._participation_ratio(np.array([3.0, 3.0, 0, 0])) - 2) < 1e-9


def test_mean_offdiag_cosine_handvalue():
    rows = R._l2(np.array([[1.0, 0], [0.0, 1.0], [1.0, 0.0]]))
    assert abs(R._mean_offdiag_cosine(rows) - 1 / 3) < 1e-9   # pair cosines {0,1,0} -> mean 1/3


def test_isotropy_unitnorm_and_decorrelates():
    rng = np.random.default_rng(1)
    aniso = rng.normal(size=(800, 48)) * (np.arange(1, 49) ** 2.0)   # strongly anisotropic columns
    corr = R.fit_isotropy(aniso)
    assert np.abs(np.linalg.norm(corr, axis=1) - 1).max() < 1e-5
    assert R.geometry(corr)["effective_rank"] > R.geometry(aniso)["effective_rank"]
    assert np.array_equal(corr, R.fit_isotropy(aniso))            # deterministic


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
                corrected=corrected, geom=geom, tile_ids=tile_ids, biomed_path=tmp)


def _gate(**over):
    a = _passing()
    a.update(over)
    return R.apply_gates(a["ids"], a["captions"], a["canon_ids"], a["canon_captions"],
                         a["corrected"], a["geom"], a["tile_ids"], a["biomed_path"])


def test_gates_accept_valid_fixture():
    out = _gate()
    assert out["missing_patients"] == 0 and R.RATIO_LO <= out["er_ratio"] <= R.RATIO_HI


def _rejects(**over):
    try:
        _gate(**over)
    except AssertionError:
        return True
    return False


def test_gates_reject_each_failure_mode():
    n, w = R.CANON_ROWS, R.BIOMED_WIDTH
    assert _rejects(canon_ids=np.array([f"X-{i}" for i in range(n)]))                         # id mismatch
    assert _rejects(canon_captions=np.array([f"other {i}" for i in range(n)]))                # caption re-render
    assert _rejects(corrected=np.pad(np.ones((n, 1), np.float32), ((0, 0), (0, 383)))[:, :384])  # wrong width (384)
    assert _rejects(tile_ids=np.array(["TCGA-999999-MISSING"]))                               # coverage gap
    assert _rejects(geom={**_passing()["geom"], "mean_offdiag_cosine": 0.05})                 # anisotropy collapse
    assert _rejects(geom={**_passing()["geom"], "effective_rank": 20.0})                      # low effective rank
    assert _rejects(geom={**_passing()["geom"], "participation_ratio": 10.0})                 # low participation
    assert _rejects(geom={**_passing()["geom"], "var_cv": 0.9})                               # variance blowup
    assert _rejects(geom={**_passing()["geom"], "effective_rank": 600.0,                      # ratio > 2 vs MiniLM
                          "norm_effective_rank": 600.0 / w})


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
