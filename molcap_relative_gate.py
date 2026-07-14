# Deterministic CPU-float64 matched-latest centroid audit and strict relative gate.
# This module stays independent of training orchestration and legacy gate dispatch.

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def _translation_stable_center(values):
    delta = values - values[0:1]
    return delta - delta.mean(dim=0, keepdim=True)


def _centroid_spectral_geometry_with_availability(patient_centroids):
    assert isinstance(patient_centroids, torch.Tensor)
    x = patient_centroids.detach().to(device="cpu", dtype=torch.float64)
    null = _null_centroid_spectral_geometry()
    if x.ndim != 2 or x.shape[0] < 2:
        return null, ["geometry:insufficient_population"]
    if not bool(torch.isfinite(x).all()):
        return null, ["geometry:nonfinite"]
    norms = x.norm(dim=1)
    centered = _translation_stable_center(x)
    covariance = centered.T @ centered / (x.shape[0] - 1)
    spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = spectrum.sum()
    trace = float(centered.square().sum().item() / (x.shape[0] - 1))
    unavailable = []
    if bool(total > 0):
        probabilities = spectrum[spectrum > 0] / total
        effective_rank = float(
            torch.exp(-(probabilities * probabilities.log()).sum()).item()
        )
        participation = total.square() / spectrum.square().sum()
        if not bool(torch.isfinite(participation)):
            participation = 1.0 / probabilities.square().sum()
        participation_ratio = float(participation.item())
    else:
        effective_rank = None
        participation_ratio = None
        unavailable.extend(
            (
                "diagnostic:effective_rank:zero_trace",
                "diagnostic:participation_ratio:zero_trace",
            )
        )
    if bool(torch.all(norms > 0)):
        unit = x / norms[:, None]
        mean_offdiag_cosine = float(
            (
                (unit.sum(dim=0).square().sum() - x.shape[0])
                / (x.shape[0] * (x.shape[0] - 1))
            ).item()
        )
    else:
        mean_offdiag_cosine = None
        unavailable.append("diagnostic:mean_offdiag_cosine:zero_norm")
    geometry = {
        "compute_device": str(x.device),
        "compute_dtype": str(x.dtype),
        "trace": trace,
        "spectrum": spectrum.tolist(),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "mean_offdiag_cosine": mean_offdiag_cosine,
        "min_norm": float(norms.min().item()),
    }
    return geometry, unavailable


# Measure sample-covariance geometry deterministically on CPU in float64.
def centroid_spectral_geometry(patient_centroids):
    geometry, _ = _centroid_spectral_geometry_with_availability(patient_centroids)
    return geometry


def _relative_centroid_geometry_with_availability(
    ema_centroids, latest_centroids, *, ema_geometry=None, latest_geometry=None
):
    assert isinstance(ema_centroids, torch.Tensor)
    assert isinstance(latest_centroids, torch.Tensor)
    ema = ema_centroids.detach().to(device="cpu", dtype=torch.float64)
    latest = latest_centroids.detach().to(device="cpu", dtype=torch.float64)
    assert ema.shape == latest.shape and ema.ndim == 2 and ema.shape[0] >= 2
    assert torch.isfinite(ema).all() and torch.isfinite(latest).all()
    ema0 = _translation_stable_center(ema)
    latest0 = _translation_stable_center(latest)
    ema_norm, latest_norm = ema0.norm(), latest0.norm()
    ema_geometry = (
        centroid_spectral_geometry(ema) if ema_geometry is None else ema_geometry
    )
    latest_geometry = (
        centroid_spectral_geometry(latest)
        if latest_geometry is None
        else latest_geometry
    )
    unavailable = []
    trace_ratio = None
    if latest_geometry["trace"] is not None and latest_geometry["trace"] > 0:
        trace_ratio = ema_geometry["trace"] / latest_geometry["trace"]
    else:
        unavailable.append("relative.trace_ratio:latest_zero_trace")
    effective_rank_ratio = None
    if (
        ema_geometry["effective_rank"] is not None
        and latest_geometry["effective_rank"] is not None
    ):
        effective_rank_ratio = (
            ema_geometry["effective_rank"] / latest_geometry["effective_rank"]
        )
    else:
        unavailable.append("relative.effective_rank_ratio:input_unavailable")
    participation_ratio = None
    if (
        ema_geometry["participation_ratio"] is not None
        and latest_geometry["participation_ratio"] is not None
    ):
        participation_ratio = (
            ema_geometry["participation_ratio"]
            / latest_geometry["participation_ratio"]
        )
    else:
        unavailable.append("relative.participation_ratio:input_unavailable")
    alignment = None
    linear_cka = None
    if bool(ema_norm > 0 and latest_norm > 0):
        alignment = float(
            ((ema0 * latest0).sum() / (ema_norm * latest_norm)).item()
        )
        ema_scaled = ema0 / ema0.abs().max()
        latest_scaled = latest0 / latest0.abs().max()
        cross = ema_scaled.T @ latest_scaled
        ema_gram = ema_scaled.T @ ema_scaled
        latest_gram = latest_scaled.T @ latest_scaled
        linear_cka = float(
            (
                cross.square().sum()
                / torch.sqrt(
                    ema_gram.square().sum() * latest_gram.square().sum()
                )
            ).item()
        )
    else:
        unavailable.extend(
            (
                "relative.alignment:zero_centered_norm",
                "diagnostic:relative.linear_cka:zero_centered_norm",
            )
        )
    raw_cosine_delta = None
    if (
        ema_geometry["mean_offdiag_cosine"] is not None
        and latest_geometry["mean_offdiag_cosine"] is not None
    ):
        raw_cosine_delta = (
            ema_geometry["mean_offdiag_cosine"]
            - latest_geometry["mean_offdiag_cosine"]
        )
    else:
        unavailable.append(
            "diagnostic:relative.mean_offdiag_cosine_delta:input_unavailable"
        )
    geometry = {
        "ema": ema_geometry,
        "latest": latest_geometry,
        "trace_ratio": trace_ratio,
        "effective_rank_ratio": effective_rank_ratio,
        "participation_ratio": participation_ratio,
        "alignment": alignment,
        "linear_cka": linear_cka,
        "mean_offdiag_cosine_delta": raw_cosine_delta,
    }
    return geometry, unavailable


def relative_centroid_geometry(
    ema_centroids, latest_centroids, *, ema_geometry=None, latest_geometry=None
):
    geometry, _ = _relative_centroid_geometry_with_availability(
        ema_centroids,
        latest_centroids,
        ema_geometry=ema_geometry,
        latest_geometry=latest_geometry,
    )
    return geometry


def matched_latest_permutation_seed(target_sha256, mapping_digest):
    assert type(target_sha256) is str and len(target_sha256) == 64
    assert type(mapping_digest) is str and len(mapping_digest) == 64
    target_bytes, mapping_bytes = (
        bytes.fromhex(target_sha256),
        bytes.fromhex(mapping_digest),
    )
    assert target_bytes.hex() == target_sha256
    assert mapping_bytes.hex() == mapping_digest
    domain = "molcap-matched-latest-v1"
    digest = hashlib.sha256(target_bytes + mapping_bytes + domain.encode("ascii"))
    return {
        "digest": digest.hexdigest(),
        "seed": int.from_bytes(digest.digest()[:8], byteorder="big", signed=False),
        "seed_bytes": 8,
        "byte_order": "big",
        "unsigned": True,
        "domain": domain,
    }


def matched_latest_permutation_audit(
    ema_centroids,
    latest_centroids,
    target_sha256,
    mapping_digest,
    *,
    permutation_count=256,
):
    assert type(permutation_count) is int and permutation_count > 0
    ema = ema_centroids.detach().to(device="cpu", dtype=torch.float64)
    latest = latest_centroids.detach().to(device="cpu", dtype=torch.float64)
    assert ema.shape == latest.shape and ema.ndim == 2 and ema.shape[0] >= 2
    ema0 = _translation_stable_center(ema)
    latest0 = _translation_stable_center(latest)
    denominator = ema0.norm() * latest0.norm()
    assert torch.isfinite(ema0).all() and torch.isfinite(latest0).all()
    assert denominator > 0
    observed_alignment = float(((ema0 * latest0).sum() / denominator).item())
    seed = matched_latest_permutation_seed(target_sha256, mapping_digest)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed["seed"])
    identity = torch.arange(len(ema0), device="cpu")
    alignments, identity_draw_count, unique_permutations = [], 0, set()
    for _ in range(permutation_count):
        permutation = torch.randperm(
            len(ema0), generator=generator, device="cpu"
        )
        identity_draw_count += int(torch.equal(permutation, identity))
        unique_permutations.add(tuple(permutation.tolist()))
        alignments.append(
            float(((ema0 * latest0[permutation]).sum() / denominator).item())
        )
    exceedance_count = sum(
        value >= observed_alignment for value in alignments
    )
    return {
        "seed": seed,
        "count": permutation_count,
        "generator_device": "cpu",
        "randperm_device": "cpu",
        "draw_policy": "sequential; identities and duplicates retained",
        "identity_draw_count": identity_draw_count,
        "unique_permutation_count": len(unique_permutations),
        "alignments": alignments,
        "observed_alignment": observed_alignment,
        "exceedance_count": exceedance_count,
        "p_value": (1 + exceedance_count) / (permutation_count + 1),
        "p_value_formula": (
            "(1 + count(permuted_alignment >= observed_alignment)) / "
            f"{permutation_count + 1}"
        ),
    }



def _fixed_distribution(values, quantiles):
    assert isinstance(values, torch.Tensor) and values.ndim == 1
    values = values.detach().to(device="cpu", dtype=torch.float64)
    assert torch.isfinite(values).all()
    summary = {
        "count": int(values.numel()),
        "mean": float(values.mean().item()) if values.numel() else None,
    }
    if values.numel():
        levels = torch.tensor(
            [level for _, level in quantiles], dtype=torch.float64
        )
        quantile_values = torch.quantile(values, levels).tolist()
    else:
        quantile_values = [None] * len(quantiles)
    summary.update(
        {label: value for (label, _), value in zip(quantiles, quantile_values)}
    )
    return summary


def _boundary_teacher_drift(bank, boundary_proposal):
    drift = torch.empty(0, dtype=torch.float64)
    if boundary_proposal is not None:
        assert boundary_proposal.base_state_step + 1 == int(
            bank.centroid_state_step.item()
        )
        committed = bank.slide_centroids[boundary_proposal.slide_ids]
        proposed = boundary_proposal.next_slide_centroids
        assert committed.shape == proposed.shape and committed.dtype == proposed.dtype
        assert (
            committed.detach().cpu().contiguous().numpy().tobytes()
            == proposed.detach().cpu().contiguous().numpy().tobytes()
        )
        drift = boundary_proposal.drift_cosines
    return {
        "first_copy_excluded": True,
        **_fixed_distribution(
            drift,
            (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)),
        ),
    }


def _boundary_proposal_provenance(
    bank, boundary_proposal, expected_proposal_type
):
    state_step = int(bank.centroid_state_step.item())
    if boundary_proposal is None:
        return {
            "present": False,
            "type_exact": None,
            "transaction_valid": None,
            "committed_match": None,
            "state_step": state_step,
            **_boundary_teacher_drift(bank, None),
        }
    type_exact = type(boundary_proposal) is expected_proposal_type
    invalid = {
        "present": True,
        "type_exact": type_exact,
        "transaction_valid": False,
        "committed_match": False,
        "state_step": state_step,
        "first_copy_excluded": True,
        "count": None,
        "mean": None,
        "q10": None,
        "q50": None,
        "q90": None,
    }
    if not type_exact:
        return invalid
    try:
        assert type(boundary_proposal.base_state_step) is int
        assert boundary_proposal.base_state_step + 1 == state_step
        slide_ids = boundary_proposal.slide_ids
        proposed = boundary_proposal.next_slide_centroids
        slide_tile_counts = boundary_proposal.slide_tile_counts
        patient_ids = boundary_proposal.patient_ids
        patient_centroids = boundary_proposal.patient_centroids
        drift = boundary_proposal.drift_cosines
        historical_tile_fraction = boundary_proposal.historical_tile_fraction
        device = bank.slide_centroids.device
        assert isinstance(slide_ids, torch.Tensor)
        assert slide_ids.ndim == 1 and slide_ids.numel() > 0
        assert slide_ids.dtype == torch.int64 and slide_ids.device == device
        assert bool(torch.all(slide_ids >= 0)) and bool(
            torch.all(slide_ids < len(bank.slide_centroids))
        )
        assert len(slide_ids) == 1 or bool(
            torch.all(slide_ids[1:] > slide_ids[:-1])
        )
        assert isinstance(proposed, torch.Tensor)
        assert proposed.shape == (
            len(slide_ids),
            bank.slide_centroids.shape[-1],
        )
        assert proposed.dtype == bank.slide_centroids.dtype
        assert proposed.device == device and not proposed.requires_grad
        assert bool(torch.isfinite(proposed).all())
        assert isinstance(slide_tile_counts, torch.Tensor)
        assert slide_tile_counts.shape == slide_ids.shape
        assert slide_tile_counts.dtype == torch.int64
        assert slide_tile_counts.device == device
        assert bool(torch.all(slide_tile_counts > 0))
        assert bool(torch.all(bank.slide_counts[slide_ids] > 0))
        assert bool(
            torch.all(
                bank.slide_tile_presentations[slide_ids]
                >= slide_tile_counts
            )
        )
        expected_patient_ids = torch.unique(
            bank.slide_to_patient[slide_ids], sorted=True
        )
        assert isinstance(patient_ids, torch.Tensor)
        assert patient_ids.shape == expected_patient_ids.shape
        assert patient_ids.dtype == torch.int64 and patient_ids.device == device
        assert torch.equal(patient_ids, expected_patient_ids)
        assert isinstance(patient_centroids, torch.Tensor)
        assert patient_centroids.shape == (
            len(patient_ids),
            bank.slide_centroids.shape[-1],
        )
        assert patient_centroids.dtype == bank.slide_centroids.dtype
        assert patient_centroids.device == device
        assert not patient_centroids.requires_grad
        assert bool(torch.isfinite(patient_centroids).all())
        observed_patient_ids, observed_patient_centroids = bank.patient_centroids(1)
        patient_positions = torch.searchsorted(
            observed_patient_ids, patient_ids.detach().cpu()
        )
        assert bool(torch.all(patient_positions < len(observed_patient_ids)))
        assert torch.equal(
            observed_patient_ids[patient_positions], patient_ids.detach().cpu()
        )
        expected_patient_centroids = observed_patient_centroids[
            patient_positions
        ].to(device=device, dtype=bank.slide_centroids.dtype)
        assert torch.allclose(
            patient_centroids,
            expected_patient_centroids,
            atol=1.0e-6,
            rtol=0.0,
        )
        seen_before = bank.slide_counts[slide_ids] > 1
        assert isinstance(drift, torch.Tensor)
        assert drift.shape == (int(seen_before.sum().item()),)
        assert drift.dtype == bank.slide_centroids.dtype and drift.device == device
        assert not drift.requires_grad and bool(torch.isfinite(drift).all())
        cosine_tolerance = 1.0e-6
        assert bool(torch.all(drift >= -1.0 - cosine_tolerance))
        assert bool(torch.all(drift <= 1.0 + cosine_tolerance))
        assert isinstance(historical_tile_fraction, torch.Tensor)
        assert historical_tile_fraction.shape == torch.Size([])
        assert historical_tile_fraction.dtype == bank.slide_centroids.dtype
        assert historical_tile_fraction.device == device
        assert not historical_tile_fraction.requires_grad
        assert bool(torch.isfinite(historical_tile_fraction))
        assert bool(historical_tile_fraction >= 0.0)
        assert bool(historical_tile_fraction <= 1.0)
        expected_fraction = (
            slide_tile_counts[seen_before].sum().float()
            / slide_tile_counts.sum().float()
        )
        assert torch.allclose(
            historical_tile_fraction,
            expected_fraction,
            atol=1.0e-6,
            rtol=0.0,
        )
        summary = _boundary_teacher_drift(bank, boundary_proposal)
    except (AssertionError, AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return invalid
    return {
        "present": True,
        "type_exact": True,
        "transaction_valid": True,
        "committed_match": True,
        "state_step": state_step,
        **summary,
    }


def _missing_boundary_proposal_provenance(
    boundary_proposal, expected_proposal_type
):
    present = boundary_proposal is not None
    return {
        "present": present,
        "type_exact": (
            type(boundary_proposal) is expected_proposal_type if present else None
        ),
        "transaction_valid": False if present else None,
        "committed_match": False if present else None,
        "state_step": None,
        "first_copy_excluded": True,
        "count": None,
        "mean": None,
        "q10": None,
        "q50": None,
        "q90": None,
    }


def _boundary_proposal_pairing(
    ema_proposal, shadow_proposal, expected_proposal_type
):
    names = (
        "base_state_step_equal",
        "slide_ids_equal",
        "slide_tile_counts_equal",
        "patient_ids_equal",
        "historical_tile_fraction_equal",
    )
    applicable = (
        type(ema_proposal) is expected_proposal_type
        and type(shadow_proposal) is expected_proposal_type
    )
    if not applicable:
        return {
            "applicable": False,
            "matches": {name: None for name in names},
        }

    def tensors_equal(left, right):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        try:
            return bool(torch.equal(left, right))
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            return False

    matches = {
        "base_state_step_equal": (
            type(ema_proposal.base_state_step) is int
            and type(shadow_proposal.base_state_step) is int
            and ema_proposal.base_state_step == shadow_proposal.base_state_step
        ),
        "slide_ids_equal": tensors_equal(
            ema_proposal.slide_ids, shadow_proposal.slide_ids
        ),
        "slide_tile_counts_equal": tensors_equal(
            ema_proposal.slide_tile_counts,
            shadow_proposal.slide_tile_counts,
        ),
        "patient_ids_equal": tensors_equal(
            ema_proposal.patient_ids, shadow_proposal.patient_ids
        ),
        "historical_tile_fraction_equal": tensors_equal(
            ema_proposal.historical_tile_fraction,
            shadow_proposal.historical_tile_fraction,
        ),
    }
    assert all(type(value) is bool for value in matches.values())
    return {"applicable": True, "matches": matches}


def _centroid_bank_state_digest(bank):
    hasher = hashlib.sha256()
    formats = {
        "slide_to_patient": "<i8",
        "slide_centroids": "<f4",
        "slide_counts": "<i8",
        "slide_tile_presentations": "<i8",
        "centroid_state_step": "<i8",
    }
    for name, dtype in formats.items():
        array = np.asarray(
            getattr(bank, name).detach().cpu().contiguous().numpy(), dtype=dtype
        )
        hasher.update(name.encode("ascii") + b"\0")
        hasher.update(dtype.encode("ascii") + b"\0")
        hasher.update(len(array.shape).to_bytes(1, byteorder="little", signed=False))
        for dimension in array.shape:
            hasher.update(int(dimension).to_bytes(8, byteorder="little", signed=False))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()



def _ordered_int64_provenance(values):
    values = values.detach().to(device="cpu", dtype=torch.int64).contiguous()
    payload = np.asarray(values.numpy(), dtype="<i8").tobytes()
    return {"count": int(values.numel()), "sha256": hashlib.sha256(payload).hexdigest()}


def _centroid_checkpoint_tensor_bytes(bank):
    return sum(
        getattr(bank, name).numel() * getattr(bank, name).element_size()
        for name in (
            "slide_centroids",
            "slide_counts",
            "slide_tile_presentations",
            "centroid_state_step",
        )
    )


def _reported_scalars_finite(value):
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is dict:
        return all(_reported_scalars_finite(item) for item in value.values())
    if type(value) in (list, tuple):
        return all(_reported_scalars_finite(item) for item in value)
    return True


def _collect_nonfinite_paths(value, path, paths):
    if isinstance(value, torch.Tensor):
        try:
            value = value.detach().to(device="cpu").tolist()
        except (RuntimeError, TypeError, ValueError):
            return
    elif isinstance(value, np.ndarray):
        try:
            value = value.tolist()
        except (TypeError, ValueError):
            return
    if isinstance(value, np.floating):
        value = float(value)
    if type(value) is float and not math.isfinite(value):
        paths.append(path)
        return
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            paths.append(path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_nonfinite_paths(item, f"{path}[{index}]", paths)


def _normalize_world_size(world_size):
    path = "provenance.world_size"
    nonfinite_paths = []
    _collect_nonfinite_paths(world_size, path, nonfinite_paths)
    reason = None
    normalized = None
    if type(world_size) is int:
        normalized = world_size
    elif isinstance(world_size, np.integer) and not isinstance(world_size, np.bool_):
        normalized = int(world_size)
    elif isinstance(world_size, torch.Tensor):
        if world_size.ndim != 0:
            reason = "non_scalar"
        else:
            try:
                torch.iinfo(world_size.dtype)
            except TypeError:
                reason = "expected_integer"
            else:
                try:
                    normalized = int(
                        world_size.detach().to(device="cpu").item()
                    )
                except (NotImplementedError, RuntimeError, TypeError, ValueError):
                    reason = "unreadable"
    elif isinstance(world_size, np.ndarray):
        if world_size.ndim != 0:
            reason = "non_scalar"
        elif np.issubdtype(world_size.dtype, np.integer) and not np.issubdtype(
            world_size.dtype, np.bool_
        ):
            normalized = int(world_size.item())
        else:
            reason = "expected_integer"
    else:
        reason = "expected_integer"
    unavailable = [] if reason is None else [f"{path}:{reason}"]
    if nonfinite_paths:
        normalized = None
        if reason == "expected_integer":
            unavailable = []
        unavailable.append(f"{path}:nonfinite")
    return normalized, unavailable, nonfinite_paths


def _null_centroid_spectral_geometry():
    return {
        "compute_device": "cpu",
        "compute_dtype": "torch.float64",
        "trace": None,
        "spectrum": None,
        "effective_rank": None,
        "participation_ratio": None,
        "mean_offdiag_cosine": None,
        "min_norm": None,
    }


def _nullable_centroid_spectral_geometry(patient_centroids):
    return _centroid_spectral_geometry_with_availability(patient_centroids)


def _prefix_geometry_unavailable(bank_name, reasons):
    prefixed = []
    for reason in reasons:
        if reason.startswith("diagnostic:"):
            prefixed.append(
                f"diagnostic:{bank_name}.{reason.removeprefix('diagnostic:')}"
            )
        elif reason.startswith("geometry:"):
            prefixed.append(
                f"{bank_name}_geometry:{reason.removeprefix('geometry:')}"
            )
        else:
            prefixed.append(f"{bank_name}.{reason}")
    return prefixed


def _unavailable_permutation_audit(
    target_sha256, mapping_digest, permutation_count
):
    return {
        "seed": matched_latest_permutation_seed(target_sha256, mapping_digest),
        "count": permutation_count,
        "generator_device": "cpu",
        "randperm_device": "cpu",
        "draw_policy": "sequential; identities and duplicates retained",
        "identity_draw_count": None,
        "unique_permutation_count": None,
        "alignments": None,
        "observed_alignment": None,
        "exceedance_count": None,
        "p_value": None,
        "p_value_formula": (
            "(1 + count(permuted_alignment >= observed_alignment)) / 257"
        ),
    }


def _legacy_geometry_projection(geometry, patient_count):
    if geometry is None:
        return {
            "patient_count": patient_count,
            "min_norm": None,
            "effective_rank": None,
            "participation_ratio": None,
            "mean_offdiag_cosine": None,
        }
    return {
        "patient_count": patient_count,
        "min_norm": geometry["min_norm"],
        "effective_rank": geometry["effective_rank"],
        "participation_ratio": geometry["participation_ratio"],
        "mean_offdiag_cosine": geometry["mean_offdiag_cosine"],
    }


def _unavailable_relative_legacy_diagnostics(
    bank,
    min_slide_updates,
    observed_ids,
    mature_ids,
    boundary_proposal,
    *,
    observed_geometry=None,
    mature_geometry=None,
):
    counts = bank.slide_counts.detach().cpu()
    mapping = bank.slide_to_patient.detach().cpu()
    observed_slides, mature_slides = counts > 0, counts >= min_slide_updates
    observed_per_patient = torch.bincount(
        mapping[observed_slides], minlength=len(bank.patient_slide_counts)
    )
    observed_per_patient = observed_per_patient[observed_per_patient > 0]
    return {
        "sample_weighted_mature_coverage": bank.sample_weighted_mature_coverage(
            min_slide_updates
        ),
        "all_observed": _legacy_geometry_projection(
            observed_geometry, int(len(observed_ids))
        ),
        "mature_only": _legacy_geometry_projection(
            mature_geometry, int(len(mature_ids))
        ),
        "population_sizes": {
            "mature_min_slide_updates": min_slide_updates,
            "observed_slides": int(observed_slides.sum().item()),
            "mature_slides": int(mature_slides.sum().item()),
            "observed_patients": int(len(observed_ids)),
            "mature_patients": int(len(mature_ids)),
        },
        "slide_update_count_distribution": {
            "population": "observed_slides",
            **_fixed_distribution(
                counts[observed_slides],
                (("q0", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q100", 1.0)),
            ),
        },
        "observed_slides_per_patient_distribution": {
            "population": "observed_patients",
            **_fixed_distribution(
                observed_per_patient,
                (("q0", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q100", 1.0)),
            ),
        },
        "boundary_teacher_centroid_drift": _boundary_teacher_drift(
            bank, boundary_proposal
        ),
    }


def _missing_latest_centroid_audit(
    ema_bank,
    history_cfg,
    *,
    target_sha256,
    mapping_digest,
    history_metadata,
    shadow_metadata,
    world_size,
    world_size_unavailable,
    world_size_nonfinite_paths,
    boundary_proposal,
    boundary_shadow_proposal,
    expected_proposal_type,
    legacy_audit,
):
    min_slide_updates = 2
    ema_ids, ema_matrix = ema_bank.patient_centroids(1)
    ema_mature_ids, ema_mature = ema_bank.patient_centroids(min_slide_updates)
    ema_geometry, ema_unavailable = _nullable_centroid_spectral_geometry(ema_matrix)
    ema_boundary = _boundary_proposal_provenance(
        ema_bank, boundary_proposal, expected_proposal_type
    )
    shadow_boundary = _missing_boundary_proposal_provenance(
        boundary_shadow_proposal, expected_proposal_type
    )
    proposal_pairing = _boundary_proposal_pairing(
        boundary_proposal,
        boundary_shadow_proposal,
        expected_proposal_type,
    )
    safe_boundary_proposal = (
        boundary_proposal if ema_boundary["committed_match"] is True else None
    )
    ema_hard_unavailable = any(
        not reason.startswith("diagnostic:") for reason in ema_unavailable
    )
    if ema_unavailable:
        ema_mature_geometry, _ = _nullable_centroid_spectral_geometry(ema_mature)
        legacy = _unavailable_relative_legacy_diagnostics(
            ema_bank,
            min_slide_updates,
            ema_ids,
            ema_mature_ids,
            safe_boundary_proposal,
            observed_geometry=ema_geometry,
            mature_geometry=ema_mature_geometry,
        )
    else:
        legacy = legacy_audit(ema_bank, min_slide_updates, safe_boundary_proposal)
    ema_counts = ema_bank.slide_counts.detach().cpu()
    null_geometry = _null_centroid_spectral_geometry()
    null_relative = {
        name: None
        for name in (
            "trace_ratio",
            "effective_rank_ratio",
            "participation_ratio",
            "alignment",
            "linear_cka",
            "mean_offdiag_cosine_delta",
        )
    }
    empty_ids = _ordered_int64_provenance(torch.empty(0, dtype=torch.int64))
    return {
        **legacy,
        "gate_version": "matched_latest_v1",
        **(
            {"nonfinite_paths": world_size_nonfinite_paths}
            if world_size_nonfinite_paths
            else {}
        ),
        "provenance": {
            "target_sha256": target_sha256,
            "mapping_digest": mapping_digest,
            "target_sha256_match": history_metadata.get("target_sha256")
            == target_sha256
            and shadow_metadata.get("target_sha256") == target_sha256,
            "mapping_digest_match": history_metadata.get("mapping_digest")
            == mapping_digest
            and shadow_metadata.get("mapping_digest") == mapping_digest,
            "world_size": world_size,
        },
        "state": {
            "min_slide_updates": min_slide_updates,
            "ema_finite": bool(torch.isfinite(ema_bank.slide_centroids).all()),
            "latest_finite": False,
            "reported_scalars_finite": True,
            "matches": {
                name: False
                for name in (
                    "slide_mapping_equal",
                    "slide_counts_equal",
                    "tile_presentation_counts_equal",
                    "state_step_equal",
                    "observed_slides_equal",
                    "mature_slides_equal",
                    "patient_ids_equal",
                    "matrix_shapes_equal",
                )
            },
            "boundary_proposals": {
                "presence_equal": ema_boundary["present"]
                == shadow_boundary["present"],
                "ema": ema_boundary,
                "shadow": shadow_boundary,
                "paired": proposal_pairing,
            },
            "ema": {
                "momentum": ema_bank.momentum,
                "state_step": int(ema_bank.centroid_state_step.item()),
                "observed_slides": int((ema_counts > 0).sum().item()),
                "mature_slides": int(
                    (ema_counts >= min_slide_updates).sum().item()
                ),
                "bank_state_digest": _centroid_bank_state_digest(ema_bank),
            },
            "latest": None,
        },
        "population": {
            "ema_mature_coverage": ema_bank.sample_weighted_mature_coverage(
                min_slide_updates
            ),
            "latest_mature_coverage": None,
            "matched_patient_count": 0,
            "ema_patient_ids": _ordered_int64_provenance(ema_ids),
            "latest_patient_ids": empty_ids,
            "ema_mature_patient_ids": _ordered_int64_provenance(ema_mature_ids),
            "latest_mature_patient_ids": empty_ids,
            "ema_matrix_shape": list(ema_matrix.shape),
            "latest_matrix_shape": None,
        },
        "ema": ema_geometry,
        "latest": null_geometry,
        "relative": null_relative,
        "permutation": _unavailable_permutation_audit(
            target_sha256, mapping_digest, 256
        ),
        "shadow": {
            "audit_time_present": False,
            "checkpoint_payload_present": False,
            "checkpoint_tensor_payload_bytes": 0,
            "state_step": None,
            "bank_state_digest": None,
            "post_pass_action": "none",
            "boundary_proposal": shadow_boundary,
        },
        "unavailable": [
            *world_size_unavailable,
            "latest_shadow",
            *_prefix_geometry_unavailable("ema", ema_unavailable),
            "latest_geometry:missing_shadow",
            "relative_geometry",
            "permutation",
            *(["legacy_diagnostics"] if ema_hard_unavailable else []),
        ],
    }


def matched_latest_centroid_audit(
    ema_bank,
    latest_bank,
    history_cfg,
    *,
    target_sha256,
    mapping_digest,
    history_metadata,
    shadow_metadata,
    world_size,
    legacy_audit,
    boundary_proposal=None,
    boundary_shadow_proposal=None,
    expected_proposal_type=None,
):
    assert hasattr(ema_bank, "patient_centroids")
    assert ema_bank.momentum == 0.9
    (
        world_size,
        world_size_unavailable,
        world_size_nonfinite_paths,
    ) = _normalize_world_size(world_size)
    if latest_bank is None:
        return _missing_latest_centroid_audit(
            ema_bank,
            history_cfg,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=history_metadata,
            shadow_metadata=shadow_metadata,
            world_size=world_size,
            world_size_unavailable=world_size_unavailable,
            world_size_nonfinite_paths=world_size_nonfinite_paths,
            boundary_proposal=boundary_proposal,
            boundary_shadow_proposal=boundary_shadow_proposal,
            expected_proposal_type=expected_proposal_type,
            legacy_audit=legacy_audit,
        )
    assert hasattr(latest_bank, "patient_centroids")
    assert latest_bank.momentum == 0.0
    min_slide_updates = 2
    ema_ids, ema_matrix = ema_bank.patient_centroids(1)
    latest_ids, latest_matrix = latest_bank.patient_centroids(1)
    ema_mature_ids, ema_mature = ema_bank.patient_centroids(min_slide_updates)
    latest_mature_ids, latest_mature = latest_bank.patient_centroids(
        min_slide_updates
    )
    ema_counts = ema_bank.slide_counts.detach().cpu()
    latest_counts = latest_bank.slide_counts.detach().cpu()
    matches = {
        "slide_mapping_equal": torch.equal(
            ema_bank.slide_to_patient.detach().cpu(),
            latest_bank.slide_to_patient.detach().cpu(),
        ),
        "slide_counts_equal": torch.equal(ema_counts, latest_counts),
        "tile_presentation_counts_equal": torch.equal(
            ema_bank.slide_tile_presentations.detach().cpu(),
            latest_bank.slide_tile_presentations.detach().cpu(),
        ),
        "state_step_equal": torch.equal(
            ema_bank.centroid_state_step.detach().cpu(),
            latest_bank.centroid_state_step.detach().cpu(),
        ),
        "observed_slides_equal": torch.equal(ema_counts > 0, latest_counts > 0),
        "mature_slides_equal": torch.equal(
            ema_counts >= min_slide_updates,
            latest_counts >= min_slide_updates,
        ),
        "patient_ids_equal": torch.equal(ema_ids, latest_ids),
        "matrix_shapes_equal": ema_matrix.shape == latest_matrix.shape,
    }
    ema_geometry, ema_unavailable = _nullable_centroid_spectral_geometry(
        ema_matrix
    )
    latest_geometry, latest_unavailable = _nullable_centroid_spectral_geometry(
        latest_matrix
    )
    unavailable = [
        *world_size_unavailable,
        *_prefix_geometry_unavailable("ema", ema_unavailable),
        *_prefix_geometry_unavailable("latest", latest_unavailable),
    ]
    relative_inputs_available = (
        ema_geometry["trace"] is not None
        and latest_geometry["trace"] is not None
        and matches["matrix_shapes_equal"]
        and matches["patient_ids_equal"]
    )
    if relative_inputs_available:
        relative_geometry, relative_unavailable = (
            _relative_centroid_geometry_with_availability(
                ema_matrix,
                latest_matrix,
                ema_geometry=ema_geometry,
                latest_geometry=latest_geometry,
            )
        )
        relative = {
            name: value
            for name, value in relative_geometry.items()
            if name not in ("ema", "latest")
        }
        unavailable.extend(relative_unavailable)
        ema_centered = _translation_stable_center(ema_matrix)
        latest_centered = _translation_stable_center(latest_matrix)
        if bool(ema_centered.norm() > 0 and latest_centered.norm() > 0):
            permutation = matched_latest_permutation_audit(
                ema_matrix,
                latest_matrix,
                target_sha256,
                mapping_digest,
                permutation_count=256,
            )
        else:
            permutation = _unavailable_permutation_audit(
                target_sha256, mapping_digest, 256
            )
            unavailable.append("permutation:zero_centered_norm")
    else:
        relative = {
            name: None
            for name in (
                "trace_ratio",
                "effective_rank_ratio",
                "participation_ratio",
                "alignment",
                "linear_cka",
                "mean_offdiag_cosine_delta",
            )
        }
        permutation = _unavailable_permutation_audit(
            target_sha256, mapping_digest, 256
        )
        unavailable.extend(("relative_geometry", "permutation"))
    population = {
        "ema_mature_coverage": ema_bank.sample_weighted_mature_coverage(
            min_slide_updates
        ),
        "latest_mature_coverage": latest_bank.sample_weighted_mature_coverage(
            min_slide_updates
        ),
        "matched_patient_count": int(len(ema_ids)) if torch.equal(ema_ids, latest_ids) else 0,
        "ema_patient_ids": _ordered_int64_provenance(ema_ids),
        "latest_patient_ids": _ordered_int64_provenance(latest_ids),
        "ema_mature_patient_ids": _ordered_int64_provenance(ema_mature_ids),
        "latest_mature_patient_ids": _ordered_int64_provenance(latest_mature_ids),
        "ema_matrix_shape": list(ema_matrix.shape),
        "latest_matrix_shape": list(latest_matrix.shape),
    }
    provenance = {
        "target_sha256": target_sha256,
        "mapping_digest": mapping_digest,
        "target_sha256_match": (
            history_metadata.get("target_sha256") == target_sha256
            and shadow_metadata.get("target_sha256") == target_sha256
        ),
        "mapping_digest_match": (
            history_metadata.get("mapping_digest") == mapping_digest
            and shadow_metadata.get("mapping_digest") == mapping_digest
        ),
        "world_size": world_size,
    }
    ema_boundary_provenance = _boundary_proposal_provenance(
        ema_bank, boundary_proposal, expected_proposal_type
    )
    shadow_boundary_provenance = _boundary_proposal_provenance(
        latest_bank, boundary_shadow_proposal, expected_proposal_type
    )
    proposal_pairing = _boundary_proposal_pairing(
        boundary_proposal,
        boundary_shadow_proposal,
        expected_proposal_type,
    )
    safe_boundary_proposal = (
        boundary_proposal
        if ema_boundary_provenance["committed_match"] is True
        else None
    )
    ema_hard_unavailable = any(
        not reason.startswith("diagnostic:") for reason in ema_unavailable
    )
    if ema_unavailable:
        ema_mature_geometry, _ = _nullable_centroid_spectral_geometry(ema_mature)
        legacy = _unavailable_relative_legacy_diagnostics(
            ema_bank,
            min_slide_updates,
            ema_ids,
            ema_mature_ids,
            safe_boundary_proposal,
            observed_geometry=ema_geometry,
            mature_geometry=ema_mature_geometry,
        )
    else:
        legacy = legacy_audit(ema_bank, min_slide_updates, safe_boundary_proposal)
    if ema_hard_unavailable:
        unavailable.append("legacy_diagnostics")
    finite_payload = {
        "legacy": legacy,
        "population": population,
        "ema": ema_geometry,
        "latest": latest_geometry,
        "relative": relative,
        "permutation": permutation,
        "ema_boundary_proposal": ema_boundary_provenance,
        "shadow_boundary_proposal": shadow_boundary_provenance,
    }
    state = {
        "min_slide_updates": min_slide_updates,
        "ema_finite": bool(torch.isfinite(ema_bank.slide_centroids).all()),
        "latest_finite": bool(torch.isfinite(latest_bank.slide_centroids).all()),
        "reported_scalars_finite": _reported_scalars_finite(finite_payload),
        "matches": matches,
        "boundary_proposals": {
            "presence_equal": ema_boundary_provenance["present"]
            == shadow_boundary_provenance["present"],
            "ema": ema_boundary_provenance,
            "shadow": shadow_boundary_provenance,
            "paired": proposal_pairing,
        },
        "ema": {
            "momentum": ema_bank.momentum,
            "state_step": int(ema_bank.centroid_state_step.item()),
            "observed_slides": int((ema_counts > 0).sum().item()),
            "mature_slides": int((ema_counts >= min_slide_updates).sum().item()),
            "bank_state_digest": _centroid_bank_state_digest(ema_bank),
        },
        "latest": {
            "momentum": latest_bank.momentum,
            "state_step": int(latest_bank.centroid_state_step.item()),
            "observed_slides": int((latest_counts > 0).sum().item()),
            "mature_slides": int(
                (latest_counts >= min_slide_updates).sum().item()
            ),
            "bank_state_digest": _centroid_bank_state_digest(latest_bank),
        },
    }
    return {
        **legacy,
        "gate_version": "matched_latest_v1",
        **(
            {"nonfinite_paths": world_size_nonfinite_paths}
            if world_size_nonfinite_paths
            else {}
        ),
        "provenance": provenance,
        "state": state,
        "population": population,
        "ema": ema_geometry,
        "latest": latest_geometry,
        "relative": relative,
        "permutation": permutation,
        "shadow": {
            "audit_time_present": True,
            "checkpoint_payload_present": True,
            "checkpoint_tensor_payload_bytes": _centroid_checkpoint_tensor_bytes(
                latest_bank
            ),
            "state_step": int(latest_bank.centroid_state_step.item()),
            "bank_state_digest": _centroid_bank_state_digest(latest_bank),
            "post_pass_action": "discard_after_durable_pass_report",
            "boundary_proposal": shadow_boundary_provenance,
        },
        "unavailable": unavailable,
    }



def evaluate_matched_latest_gate(audit, history_cfg):
    thresholds = {
        "min_slide_updates": 2,
        "min_sample_weighted_coverage": 0.95,
        "min_geometry_patients": 512,
        "min_centroid_norm": 1.0e-6,
        "min_trace_ratio": 0.05263157894736842,
        "min_effective_rank_ratio": 0.5,
        "min_participation_ratio": 0.5,
        "min_alignment_exclusive": 0.0,
        "permutation_count": 256,
        "max_permutation_p_value": 0.01,
    }
    expected_config = {
        "min_slide_updates": 2,
        "min_sample_weighted_coverage": 0.95,
        "min_geometry_patients": 512,
        "min_centroid_norm": 1.0e-6,
        "permutation_count": 256,
        "permutation_seed_domain": "molcap-matched-latest-v1",
        "min_trace_ratio": 0.05263157894736842,
        "min_effective_rank_ratio": 0.5,
        "min_participation_ratio": 0.5,
        "min_alignment": 0.0,
        "max_permutation_p_value": 0.01,
    }
    failures = [
        f"config_{name}_exact"
        for name, value in expected_config.items()
        if type(history_cfg.get(name)) is not type(value)
        or history_cfg.get(name) != value
    ]
    provenance, state, population = (
        audit["provenance"],
        audit["state"],
        audit["population"],
    )
    ema, latest, relative, permutation = (
        audit["ema"],
        audit["latest"],
        audit["relative"],
        audit["permutation"],
    )
    def at_least(value, threshold):
        return type(value) in (int, float) and math.isfinite(value) and value >= threshold

    def greater_than(value, threshold):
        return type(value) in (int, float) and math.isfinite(value) and value > threshold

    def at_most(value, threshold):
        return type(value) in (int, float) and math.isfinite(value) and value <= threshold

    boundary_proposals = state.get("boundary_proposals")
    ema_boundary = None if boundary_proposals is None else boundary_proposals["ema"]
    shadow_boundary = (
        None if boundary_proposals is None else boundary_proposals["shadow"]
    )
    proposal_pairing = (
        None if boundary_proposals is None else boundary_proposals.get("paired")
    )
    proposal_pair_checks = (
        ()
        if proposal_pairing is None
        else tuple(
            (
                f"boundary_proposal_{name}",
                proposal_pairing["applicable"] is not True or value is True,
            )
            for name, value in proposal_pairing["matches"].items()
        )
    )
    checks = (
        ("target_sha256_match", provenance["target_sha256_match"]),
        ("mapping_digest_match", provenance["mapping_digest_match"]),
        (
            "latest_shadow_present",
            audit.get("shadow", {}).get("audit_time_present", True),
        ),
        (
            "boundary_proposal_presence_parity",
            boundary_proposals is None or boundary_proposals["presence_equal"],
        ),
        (
            "boundary_ema_proposal_type_exact",
            ema_boundary is None
            or not ema_boundary["present"]
            or ema_boundary["type_exact"] is True,
        ),
        (
            "boundary_ema_proposal_transaction_valid",
            ema_boundary is None
            or not ema_boundary["present"]
            or ema_boundary["transaction_valid"] is True,
        ),
        (
            "boundary_ema_proposal_committed",
            ema_boundary is None
            or not ema_boundary["present"]
            or ema_boundary["committed_match"] is True,
        ),
        (
            "boundary_shadow_proposal_type_exact",
            shadow_boundary is None
            or not shadow_boundary["present"]
            or shadow_boundary["type_exact"] is True,
        ),
        (
            "boundary_shadow_proposal_transaction_valid",
            shadow_boundary is None
            or not shadow_boundary["present"]
            or shadow_boundary["transaction_valid"] is True,
        ),
        (
            "boundary_shadow_proposal_committed",
            shadow_boundary is None
            or not shadow_boundary["present"]
            or shadow_boundary["committed_match"] is True,
        ),
        *proposal_pair_checks,
        (
            "world_size_one",
            type(provenance["world_size"]) is int
            and provenance["world_size"] == 1,
        ),
        ("min_slide_updates_exact", state["min_slide_updates"] == 2),
        ("ema_state_finite", state["ema_finite"]),
        ("latest_state_finite", state["latest_finite"]),
        *state["matches"].items(),
        (
            "ema_min_sample_weighted_coverage",
            at_least(population["ema_mature_coverage"], 0.95),
        ),
        (
            "latest_min_sample_weighted_coverage",
            at_least(population["latest_mature_coverage"], 0.95),
        ),
        ("min_geometry_patients", at_least(population["matched_patient_count"], 512)),
        ("ema_min_centroid_norm_strict", greater_than(ema["min_norm"], 1.0e-6)),
        ("latest_min_centroid_norm_strict", greater_than(latest["min_norm"], 1.0e-6)),
        ("ema_trace_positive", greater_than(ema["trace"], 0.0)),
        ("latest_trace_positive", greater_than(latest["trace"], 0.0)),
        ("all_reported_scalars_finite", state["reported_scalars_finite"]),
        (
            "min_trace_ratio",
            at_least(relative["trace_ratio"], 0.05263157894736842),
        ),
        ("min_effective_rank_ratio", at_least(relative["effective_rank_ratio"], 0.5)),
        ("min_participation_ratio", at_least(relative["participation_ratio"], 0.5)),
        ("min_alignment_strict", greater_than(relative["alignment"], 0.0)),
        ("permutation_count_exact", permutation["count"] == 256),
        ("max_permutation_p_value", at_most(permutation["p_value"], 0.01)),
    )
    checks = tuple(
        (name, passed if type(passed) is bool else False)
        for name, passed in checks
    )
    assert all(type(passed) is bool for _, passed in checks)
    failures.extend(name for name, passed in checks if passed is not True)
    failures.extend(
        f"audit_available:{name}"
        for name in audit["unavailable"]
        if not name.startswith("diagnostic:")
    )
    observed = {
        "ema_mature_coverage": population["ema_mature_coverage"],
        "latest_mature_coverage": population["latest_mature_coverage"],
        "matched_patient_count": population["matched_patient_count"],
        "ema_min_norm": ema["min_norm"],
        "latest_min_norm": latest["min_norm"],
        "ema_trace": ema["trace"],
        "latest_trace": latest["trace"],
        **relative,
        "permutation_p_value": permutation["p_value"],
    }
    return {
        **audit,
        "thresholds": thresholds,
        "diagnostic_absolute_thresholds": {
            "min_effective_rank": 32.0,
            "min_participation_ratio": 16.0,
            "max_mean_offdiag_cosine": 0.95,
        },
        "observed": observed,
        "failures": failures,
        "passed": not failures,
    }



def _fsync_parent_directory(directory, *, platform_name=None):
    platform_name = os.name if platform_name is None else platform_name
    if platform_name == "nt":
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _normalize_report_nonfinite(
    value, path, nonfinite_paths, unsupported_paths
):
    if isinstance(value, torch.Tensor):
        try:
            value = value.detach().to(device="cpu").tolist()
        except (RuntimeError, TypeError, ValueError):
            unsupported_paths.append(path or "$")
            return None
    elif isinstance(value, np.ndarray):
        try:
            value = value.tolist()
        except (TypeError, ValueError):
            unsupported_paths.append(path or "$")
            return None
    if isinstance(value, np.bool_):
        value = bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        value = int(value)
    if type(value) is float and not math.isfinite(value):
        nonfinite_paths.append(path or "$")
        return None
    if isinstance(value, complex):
        issue_paths = (
            nonfinite_paths
            if not (math.isfinite(value.real) and math.isfinite(value.imag))
            else unsupported_paths
        )
        issue_paths.append(path or "$")
        return None
    if isinstance(value, dict):
        normalized = {}
        for index, (key, item) in enumerate(value.items()):
            if type(key) is not str:
                unsupported_paths.append(
                    f"{path}.<key[{index}]>" if path else f"<key[{index}]>"
                )
                key = f"__unsupported_key_{index}__"
                while key in normalized:
                    key = f"_{key}"
            normalized[key] = _normalize_report_nonfinite(
                item,
                f"{path}.{key}" if path else str(key),
                nonfinite_paths,
                unsupported_paths,
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_report_nonfinite(
                item,
                f"{path}[{index}]",
                nonfinite_paths,
                unsupported_paths,
            )
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in (bool, int, float, str):
        return value
    unsupported_paths.append(path or "$")
    return None


def finalize_matched_latest_report(report):
    assert type(report) is dict
    detected_paths = []
    detected_unsupported_paths = []
    staged = _normalize_report_nonfinite(
        report, "", detected_paths, detected_unsupported_paths
    )
    prior_paths = staged.get("nonfinite_paths", [])
    prior_paths = prior_paths if isinstance(prior_paths, list) else []
    nonfinite_paths = list(dict.fromkeys((*prior_paths, *detected_paths)))
    staged["nonfinite_paths"] = nonfinite_paths
    prior_unsupported_paths = staged.get("unsupported_paths", [])
    prior_unsupported_paths = (
        prior_unsupported_paths
        if isinstance(prior_unsupported_paths, list)
        else []
    )
    unsupported_paths = list(
        dict.fromkeys((*prior_unsupported_paths, *detected_unsupported_paths))
    )
    if unsupported_paths:
        staged["unsupported_paths"] = unsupported_paths
    failures_value = staged.get("failures", [])
    failures = list(failures_value) if isinstance(failures_value, list) else []
    for path in nonfinite_paths:
        failure = f"report_nonfinite:{path}"
        if failure not in failures:
            failures.append(failure)
    for path in unsupported_paths:
        failure = f"report_unserializable:{path}"
        if failure not in failures:
            failures.append(failure)
    staged["failures"] = failures
    if nonfinite_paths or unsupported_paths:
        staged["passed"] = False
        if type(staged.get("state")) is dict:
            staged["state"]["reported_scalars_finite"] = False
    json.dumps(staged, allow_nan=False)
    return staged


def _windows_write_through_replace(
    source,
    destination,
    *,
    move_file_ex=None,
    get_last_error=None,
    format_error=None,
):
    import ctypes

    if move_file_ex is None:
        move_file_ex = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file_ex.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        move_file_ex.restype = ctypes.c_int
    source = str(Path(source).resolve())
    destination = str(Path(destination).resolve())
    flags = 0x00000001 | 0x00000008
    if move_file_ex(source, destination, flags):
        return
    error_code = (
        ctypes.get_last_error() if get_last_error is None else get_last_error()
    )
    message = (
        ctypes.FormatError(error_code)
        if format_error is None
        else format_error(error_code)
    )
    raise OSError(error_code, message, destination)


def _matched_latest_persistence(platform_name):
    if platform_name == "nt":
        strategy = "windows_movefileex_replace_existing_write_through"
    elif platform_name == "posix":
        strategy = "posix_temp_flush_fsync_replace_parent_fsync"
    else:
        raise OSError(f"unsupported durable report platform: {platform_name!r}")
    return {"strategy": strategy, "durable_before_return": True}


def _write_matched_latest_gate_report(
    report,
    report_path,
    *,
    platform_name=None,
    move_file_ex=None,
    get_last_error=None,
    format_error=None,
    replace_operation=None,
):
    report_path = Path(report_path)
    platform_name = os.name if platform_name is None else platform_name
    report = dict(report)
    report["persistence"] = _matched_latest_persistence(platform_name)
    report = finalize_matched_latest_report(report)
    serialized = json.dumps(report, allow_nan=False, indent=2) + "\n"
    temporary_path = None
    raw_descriptor = None
    primary_error = None
    try:
        raw_descriptor, temporary_name = tempfile.mkstemp(
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        handle = os.fdopen(
            raw_descriptor, "w", encoding="utf-8", newline="\n"
        )
        raw_descriptor = None
        with handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if platform_name == "nt":
            _windows_write_through_replace(
                temporary_path,
                report_path,
                move_file_ex=move_file_ex,
                get_last_error=get_last_error,
                format_error=format_error,
            )
        elif platform_name == "posix":
            replace = os.replace if replace_operation is None else replace_operation
            replace(temporary_path, report_path)
            _fsync_parent_directory(
                report_path.parent, platform_name=platform_name
            )
        else:
            raise OSError(
                f"unsupported durable report platform: {platform_name!r}"
            )
        return report
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error = None
        if raw_descriptor is not None:
            try:
                os.close(raw_descriptor)
            except Exception as error:
                cleanup_error = error
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error
