# Deterministic CPU-float64 matched-latest centroid audit and strict relative gate.
# This module stays independent of training orchestration and legacy gate dispatch.

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

# Measure sample-covariance geometry deterministically on CPU in float64.
def centroid_spectral_geometry(patient_centroids):
    assert isinstance(patient_centroids, torch.Tensor)
    x = patient_centroids.detach().to(device="cpu", dtype=torch.float64)
    assert x.ndim == 2 and x.shape[0] >= 2 and torch.isfinite(x).all()
    norms = x.norm(dim=1)
    assert torch.all(norms > 0)
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (x.shape[0] - 1)
    spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = spectrum.sum()
    assert total > 0
    probabilities = spectrum[spectrum > 0] / total
    unit = x / norms[:, None]
    return {
        "compute_device": str(x.device),
        "compute_dtype": str(x.dtype),
        "trace": float(centered.square().sum().item() / (x.shape[0] - 1)),
        "spectrum": spectrum.tolist(),
        "effective_rank": float(
            torch.exp(-(probabilities * probabilities.log()).sum()).item()
        ),
        "participation_ratio": float(
            (total.square() / spectrum.square().sum()).item()
        ),
        "mean_offdiag_cosine": float(
            (
                (unit.sum(dim=0).square().sum() - x.shape[0])
                / (x.shape[0] * (x.shape[0] - 1))
            ).item()
        ),
        "min_norm": float(norms.min().item()),
    }


def relative_centroid_geometry(
    ema_centroids, latest_centroids, *, ema_geometry=None, latest_geometry=None
):
    assert isinstance(ema_centroids, torch.Tensor)
    assert isinstance(latest_centroids, torch.Tensor)
    ema = ema_centroids.detach().to(device="cpu", dtype=torch.float64)
    latest = latest_centroids.detach().to(device="cpu", dtype=torch.float64)
    assert ema.shape == latest.shape and ema.ndim == 2 and ema.shape[0] >= 2
    assert torch.isfinite(ema).all() and torch.isfinite(latest).all()
    ema0 = ema - ema.mean(dim=0, keepdim=True)
    latest0 = latest - latest.mean(dim=0, keepdim=True)
    ema_norm, latest_norm = ema0.norm(), latest0.norm()
    assert ema_norm > 0 and latest_norm > 0
    cross = ema0.T @ latest0
    ema_gram, latest_gram = ema0.T @ ema0, latest0.T @ latest0
    ema_geometry = (
        centroid_spectral_geometry(ema) if ema_geometry is None else ema_geometry
    )
    latest_geometry = (
        centroid_spectral_geometry(latest)
        if latest_geometry is None
        else latest_geometry
    )
    return {
        "ema": ema_geometry,
        "latest": latest_geometry,
        "trace_ratio": ema_geometry["trace"] / latest_geometry["trace"],
        "effective_rank_ratio": (
            ema_geometry["effective_rank"] / latest_geometry["effective_rank"]
        ),
        "participation_ratio": (
            ema_geometry["participation_ratio"]
            / latest_geometry["participation_ratio"]
        ),
        "alignment": float(
            ((ema0 * latest0).sum() / (ema_norm * latest_norm)).item()
        ),
        "linear_cka": float(
            (
                cross.square().sum()
                / torch.sqrt(
                    ema_gram.square().sum() * latest_gram.square().sum()
                )
            ).item()
        ),
        "mean_offdiag_cosine_delta": (
            ema_geometry["mean_offdiag_cosine"]
            - latest_geometry["mean_offdiag_cosine"]
        ),
    }


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
    ema0 = ema - ema.mean(dim=0, keepdim=True)
    latest0 = latest - latest.mean(dim=0, keepdim=True)
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


def _boundary_shadow_provenance(bank, boundary_proposal):
    state_step = int(bank.centroid_state_step.item())
    if boundary_proposal is None:
        return {
            "present": False,
            "committed_match": None,
            "state_step": state_step,
            **_boundary_teacher_drift(bank, None),
        }
    invalid = {
        "present": True,
        "committed_match": False,
        "state_step": state_step,
        "first_copy_excluded": True,
        "count": None,
        "mean": None,
        "q10": None,
        "q50": None,
        "q90": None,
    }
    try:
        slide_ids = boundary_proposal.slide_ids
        proposed = boundary_proposal.next_slide_centroids
        drift = boundary_proposal.drift_cosines
        assert isinstance(slide_ids, torch.Tensor)
        assert slide_ids.ndim == 1 and slide_ids.numel() > 0
        assert slide_ids.dtype == torch.int64
        assert torch.all(slide_ids >= 0) and torch.all(
            slide_ids < len(bank.slide_centroids)
        )
        assert len(slide_ids) == 1 or torch.all(slide_ids[1:] > slide_ids[:-1])
        assert isinstance(proposed, torch.Tensor)
        assert isinstance(drift, torch.Tensor) and torch.isfinite(drift).all()
        summary = _boundary_teacher_drift(bank, boundary_proposal)
    except (AssertionError, AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return invalid
    return {
        "present": True,
        "committed_match": True,
        "state_step": state_step,
        **summary,
    }


def _missing_boundary_shadow_provenance(boundary_proposal):
    return {
        "present": boundary_proposal is not None,
        "committed_match": False if boundary_proposal is not None else None,
        "state_step": None,
        "first_copy_excluded": True,
        "count": None,
        "mean": None,
        "q10": None,
        "q50": None,
        "q90": None,
    }


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
    x = patient_centroids.detach().to(device="cpu", dtype=torch.float64)
    reason = None
    if x.ndim != 2 or x.shape[0] < 2:
        reason = "insufficient_population"
    elif not bool(torch.isfinite(x).all()):
        reason = "nonfinite"
    elif not bool(torch.all(x.norm(dim=1) > 0)):
        reason = "zero_norm"
    elif not bool((x - x.mean(dim=0, keepdim=True)).square().sum() > 0):
        reason = "zero_trace"
    if reason is None:
        return centroid_spectral_geometry(x), None
    return _null_centroid_spectral_geometry(), reason


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


def _unavailable_relative_legacy_diagnostics(
    bank, min_slide_updates, observed_ids, mature_ids, boundary_proposal
):
    counts = bank.slide_counts.detach().cpu()
    mapping = bank.slide_to_patient.detach().cpu()
    observed_slides, mature_slides = counts > 0, counts >= min_slide_updates
    observed_per_patient = torch.bincount(
        mapping[observed_slides], minlength=len(bank.patient_slide_counts)
    )
    observed_per_patient = observed_per_patient[observed_per_patient > 0]
    null_geometry = lambda patient_count: {
        "patient_count": patient_count,
        "min_norm": None,
        "effective_rank": None,
        "participation_ratio": None,
        "mean_offdiag_cosine": None,
    }
    return {
        "sample_weighted_mature_coverage": bank.sample_weighted_mature_coverage(
            min_slide_updates
        ),
        "all_observed": null_geometry(int(len(observed_ids))),
        "mature_only": null_geometry(int(len(mature_ids))),
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
    boundary_proposal,
    boundary_shadow_proposal,
    legacy_audit,
):
    min_slide_updates = 2
    ema_ids, ema_matrix = ema_bank.patient_centroids(1)
    ema_mature_ids, _ = ema_bank.patient_centroids(min_slide_updates)
    ema_geometry, ema_unavailable = _nullable_centroid_spectral_geometry(ema_matrix)
    legacy = (
        legacy_audit(ema_bank, min_slide_updates, boundary_proposal)
        if ema_unavailable is None
        else _unavailable_relative_legacy_diagnostics(
            ema_bank,
            min_slide_updates,
            ema_ids,
            ema_mature_ids,
            boundary_proposal,
        )
    )
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
            "boundary_proposal": _missing_boundary_shadow_provenance(
                boundary_shadow_proposal
            ),
        },
        "unavailable": [
            "latest_shadow",
            *([f"ema_geometry:{ema_unavailable}"] if ema_unavailable else []),
            "latest_geometry:missing_shadow",
            "relative_geometry",
            "permutation",
            *(["legacy_diagnostics"] if ema_unavailable else []),
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
):
    assert hasattr(ema_bank, "patient_centroids")
    assert ema_bank.momentum == 0.9
    if latest_bank is None:
        return _missing_latest_centroid_audit(
            ema_bank,
            history_cfg,
            target_sha256=target_sha256,
            mapping_digest=mapping_digest,
            history_metadata=history_metadata,
            shadow_metadata=shadow_metadata,
            world_size=world_size,
            boundary_proposal=boundary_proposal,
            boundary_shadow_proposal=boundary_shadow_proposal,
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
    unavailable = []
    if ema_unavailable is not None:
        unavailable.append(f"ema_geometry:{ema_unavailable}")
    if latest_unavailable is not None:
        unavailable.append(f"latest_geometry:{latest_unavailable}")
    relative_available = (
        ema_unavailable is latest_unavailable is None
        and matches["matrix_shapes_equal"]
        and matches["patient_ids_equal"]
    )
    if relative_available:
        relative_geometry = relative_centroid_geometry(
            ema_matrix,
            latest_matrix,
            ema_geometry=ema_geometry,
            latest_geometry=latest_geometry,
        )
        relative = {
            name: value
            for name, value in relative_geometry.items()
            if name not in ("ema", "latest")
        }
        permutation = matched_latest_permutation_audit(
            ema_matrix,
            latest_matrix,
            target_sha256,
            mapping_digest,
            permutation_count=256,
        )
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
    legacy = (
        legacy_audit(ema_bank, min_slide_updates, boundary_proposal)
        if ema_unavailable is None
        else _unavailable_relative_legacy_diagnostics(
            ema_bank,
            min_slide_updates,
            ema_ids,
            ema_mature_ids,
            boundary_proposal,
        )
    )
    if ema_unavailable is not None:
        unavailable.append("legacy_diagnostics")
    shadow_boundary_provenance = _boundary_shadow_provenance(
        latest_bank, boundary_shadow_proposal
    )
    finite_payload = {
        "legacy": legacy,
        "population": population,
        "ema": ema_geometry,
        "latest": latest_geometry,
        "relative": relative,
        "permutation": permutation,
        "shadow_boundary_proposal": shadow_boundary_provenance,
    }
    state = {
        "min_slide_updates": min_slide_updates,
        "ema_finite": bool(torch.isfinite(ema_bank.slide_centroids).all()),
        "latest_finite": bool(torch.isfinite(latest_bank.slide_centroids).all()),
        "reported_scalars_finite": _reported_scalars_finite(finite_payload),
        "matches": matches,
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

    checks = (
        ("target_sha256_match", provenance["target_sha256_match"]),
        ("mapping_digest_match", provenance["mapping_digest_match"]),
        (
            "latest_shadow_present",
            audit.get("shadow", {}).get("audit_time_present", True),
        ),
        (
            "boundary_shadow_proposal_committed",
            audit.get("shadow", {}).get("boundary_proposal") is None
            or not audit["shadow"]["boundary_proposal"]["present"]
            or audit["shadow"]["boundary_proposal"]["committed_match"] is True,
        ),
        ("world_size_one", provenance["world_size"] == 1),
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
    failures.extend(name for name, passed in checks if not passed)
    failures.extend(f"audit_available:{name}" for name in audit["unavailable"])
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


def _write_matched_latest_gate_report(report, report_path):
    report_path = Path(report_path)
    serialized = json.dumps(report, allow_nan=False, indent=2) + "\n"
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, report_path)
    _fsync_parent_directory(report_path.parent)
