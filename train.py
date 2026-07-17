# Continual DINOv2 pretraining on TCGA tiles (single-GPU). Three loss terms:
# DINO CLS self-distillation (Sinkhorn-Knopp centred teacher targets),
# I-JEPA patch-feature regression, and a KDE uniformity term on the
# L2-normalised CLS tokens. YAML drives the tunable knobs (backbone variant,
# LR + LR scheduler, drop path, layerwise decay, KDE weight + concentration,
# FLOP/sample budgets, batch size); other DINOv2 hyperparameters are hardcoded
# inline at their use sites.

import atexit
import contextlib
import fnmatch
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from dataloader import TCGATileDataset, TILE_SIZE
from model import DINOHead, DinoV2ViT, GradScale, JEPAPredictor, gradient_alignment, linear_ramp, load_dinov2_pretrained, molcap_loss, seed_neutral_molcap_head
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Read the YAML recipe and fail before any GPU work if the parquet tile dataset is absent.
# expandvars is necessary to resolve `$USER` for checked-in configs.
def load_config():
    if len(sys.argv) < 2:
        raise ValueError("usage: python train.py <config.yaml> [output_dir=<path>]")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    # Optional `key=value` overrides after the config; only output_dir is supported,
    # since it's the run identifier and routinely set per-submission from the CLI.
    for arg in sys.argv[2:]:
        key, _, value = arg.partition("=")
        if key != "output_dir":
            raise ValueError(f"unsupported override {arg!r}; only output_dir=<path> is supported")
        cfg["project"]["output_dir"] = os.path.expandvars(value)
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not any(dataset_dir.glob("shard-*.parquet")):
        raise FileNotFoundError(
            f"No parquet shards (shard-*.parquet) under {dataset_dir}. Pull the 4M-tile "
            f"parquet dataset from medarc/nanopath on HF by running "
            f"`python prepare.py {cfg['config_path']} download=True`. Follow the data setup in "
            f"README.md before launching train.py."
        )
    return cfg


# Arm Labless before any GPU work so direct `python train.py ...` gets the same
# no-scope GitHub device login path as the SLURM launcher. Noninteractive runs
# train locally unless the launcher passed a preauthorized token file.
def maybe_arm_labless_autosubmit(cfg, repo_dir):
    token_path = os.environ.get("LABLESS_AUTOSUBMIT_FILE", "")
    eligible = (
        bool(cfg["probe"]["enabled"])
        and int(cfg["probe"]["count"]) > 0
        and int(cfg["train"]["max_train_samples"]) == 1_000_000
        and int(cfg["train"]["max_train_flops"]) == 1_000_000_000_000_000_000
    )
    if token_path:
        atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
        return token_path
    if not eligible:
        return ""
    if not sys.stdin.isatty():
        if not os.environ.get("SLURM_JOB_ID"):
            print(f"{console_prefix()} Labless  no interactive stdin; training will run without auto-submit.", flush=True)
        return ""
    print("This looks like a full Labless-eligible run. Leave either prompt blank to train without auto-submit.", flush=True)
    run_name = input("Labless run name (<=20 chars): ").strip()
    notes = input("Labless notes: ").strip()
    if not run_name or not notes or len(run_name) > 20:
        print("Labless auto-submit skipped; run name and notes are required, and run name must be <=20 chars.", flush=True)
        return ""
    token_path = str(Path(str(Path(cfg["project"]["output_dir"]).expanduser().resolve()) + ".labless_autosubmit.json"))
    status = subprocess.run(
        [sys.executable, str(repo_dir / "labless" / "submit_to_labless.py"), "login_only=true", f"token_output={token_path}", f"run_name={run_name}", f"notes={notes}"],
        cwd=repo_dir,
    ).returncode
    if status != 0:
        print("Labless login did not complete; training will run without auto-submit.", flush=True)
        Path(token_path).unlink(missing_ok=True)
        return ""
    os.environ["LABLESS_AUTOSUBMIT_FILE"] = token_path
    atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
    return token_path


def finish_labless_autosubmit(token_path, output_dir, repo_dir):
    token_file = Path(token_path) if token_path else None
    if token_file is None or not token_file.exists():
        return
    token = json.loads(token_file.read_text())
    status = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "labless" / "submit_to_labless.py"),
            f"output_dir={output_dir.resolve()}",
            f"run_name={token['run_name']}",
            f"notes={token['notes']}",
            f"github_token_file={token_file}",
        ],
        cwd=repo_dir,
    ).returncode
    token_file.unlink(missing_ok=True)
    if status == 2:
        print(f"{console_prefix()} Labless  auto-submit skipped because the completed run did not satisfy submission restrictions.", flush=True)
    elif status != 0:
        raise SystemExit(status)


# Cosine schedule from `start` to `end` over fractional progress in [0, 1].
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(1.0, max(0.0, frac))))


# Sinkhorn-Knopp centring across this batch, used for DINO teacher targets.
def sinkhorn(x, temp):
    q = torch.exp(x.float() / temp).t()
    b = q.shape[1]
    k = q.shape[0]
    q /= q.sum()
    for _ in range(3):
        q /= q.sum(1, keepdim=True) * k
        q /= q.sum(0, keepdim=True) * b
    return (q * b).t()


# Cross-entropy between teacher distribution and softmax(student / 0.1).
def dino_ce(student, teacher):
    return -(teacher * F.log_softmax(student / 0.1, dim=-1)).sum(-1).mean()


# KDE uniformity loss on L2-normalised CLS tokens.
def kde_loss(x, concentration):
    x = F.normalize(x, p=2, dim=-1)
    sim = concentration * (x @ x.T)
    sim.fill_diagonal_(-float("inf"))
    return torch.logsumexp(sim, dim=1).mean() - math.log(max(1, sim.shape[1] - 1))


# I-JEPA target mask: contiguous square blocks so the predictor must infer missing tissue context.
def make_block_mask(batch, grid, device, n_blocks=4, block_scale=0.10):
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    side = max(1, round(grid * block_scale ** 0.5))
    for i in range(batch):
        for _ in range(n_blocks):
            top = random.randint(0, grid - side)
            left = random.randint(0, grid - side)
            masks[i, top : top + side, left : left + side] = True
    masks = masks.flatten(1)
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# AdamW parameter groups with layer-wise LR decay on the backbone:
# block i gets lr * layerwise_decay^(depth - 1 - i); patch_embed gets the deepest decay
# multiplied by patch_embed_lr_mult; biases and norms get no weight decay; the head's
# DINO final weight-norm last_layer parameters get an LR-freeze for the first dino.freeze_last_layer_fraction.
def build_param_groups(student_backbone, student_dino_head, student_predictor, layerwise_decay, patch_embed_lr_mult):
    depth = len(student_backbone.blocks)
    # Coalesce params that share (lr_mult, wd_mult, last_layer) into a single group each (~30 groups
    # instead of one-per-param), so AdamW's foreach path fuses the step across many tensors rather than
    # launching per-parameter kernels. Per-param lr/wd are unchanged, so the optimization is numerically identical.
    coalesced = {}
    modules = ((student_backbone, "backbone"), (student_dino_head, "dino_head"), (student_predictor, "jepa_predictor"))
    for module, kind in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if kind == "backbone" and name.startswith("blocks."):
                lr_mult = layerwise_decay ** (depth - 1 - int(name.split(".")[1]))
            elif kind == "backbone" and name.startswith("patch_embed."):
                lr_mult = (layerwise_decay ** depth) * patch_embed_lr_mult
            wd_mult = 0.0 if name.endswith("bias") or "norm" in name or p.ndim < 2 else 1.0
            key = (lr_mult, wd_mult, "last_layer" in name)
            coalesced.setdefault(key, {"params": [], "lr_mult": lr_mult, "wd_mult": wd_mult, "last_layer": key[2]})["params"].append(p)
    return list(coalesced.values())


# EMA-update teacher modules from student modules with a single multiplicative decay.
# Params are fused into two _foreach kernels (mul then add) instead of a Python per-tensor loop;
# numerically identical (pt = pt*m + ps*(1-m) per tensor). Called under torch.no_grad() by the caller.
def update_ema(student_module, teacher_module, momentum):
    teacher_params, student_params = list(teacher_module.parameters()), list(student_module.parameters())
    torch._foreach_mul_(teacher_params, momentum)
    torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)
    for bs, bt in zip(student_module.buffers(), teacher_module.buffers()):
        bt.copy_(bs)


class Hierarchy(NamedTuple):
    slide_ids: torch.Tensor
    slide_means: torch.Tensor
    patient_ids: torch.Tensor
    patient_means: torch.Tensor
    patient_sums: torch.Tensor
    patient_counts: torch.Tensor


class CentroidProposal(NamedTuple):
    slide_ids: torch.Tensor
    next_slide_centroids: torch.Tensor
    patient_ids: torch.Tensor
    patient_centroids: torch.Tensor
    patient_sums: torch.Tensor
    patient_counts: torch.Tensor


class CentroidStep(NamedTuple):
    loss: torch.Tensor
    proposal: CentroidProposal | None
    scale: float
    coverage: float
    patient_count: int

    @classmethod
    def zero(cls, reference: torch.Tensor):
        return cls(reference.new_zeros(()), None, 0.0, 0.0, 0)


def crop_major_tile_mean(features: torch.Tensor, views: int, batch_size: int) -> torch.Tensor:
    """Average crop-major features while returning one row per original tile."""
    if features.ndim < 1:
        raise ValueError("features must have a leading crop-major dimension")
    if not isinstance(views, Integral) or isinstance(views, bool) or views <= 0:
        raise ValueError("views must be a positive integer")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if features.shape[0] != views * batch_size:
        raise ValueError("crop-major features do not match views * batch_size")
    return features.reshape(views, batch_size, *features.shape[1:]).mean(dim=0)


def hierarchical_means(
    tile_features: torch.Tensor,
    tile_slide_idx: torch.Tensor,
    slide_to_patient: torch.Tensor,
) -> Hierarchy:
    """Build equal-tile, equal-slide, then equal-patient means deterministically."""
    if tile_features.ndim != 2:
        raise ValueError("tile_features must have shape [tiles, features]")
    if not tile_features.is_floating_point():
        raise ValueError("tile_features must be floating point")
    if tile_slide_idx.ndim != 1 or tile_slide_idx.dtype != torch.int64:
        raise ValueError("tile_slide_idx must be a one-dimensional int64 tensor")
    if tile_features.shape[0] != tile_slide_idx.shape[0]:
        raise ValueError("tile_features and tile_slide_idx must have the same length")
    if tile_features.device != tile_slide_idx.device:
        raise ValueError("tile_features and tile_slide_idx must be on the same device")
    if slide_to_patient.ndim != 1 or slide_to_patient.dtype != torch.int64:
        raise ValueError("slide_to_patient must be a one-dimensional int64 tensor")
    if slide_to_patient.numel() and torch.any(slide_to_patient < 0):
        raise ValueError("slide_to_patient IDs must be nonnegative")

    if tile_slide_idx.numel() and (
        torch.any(tile_slide_idx < 0) or torch.any(tile_slide_idx >= slide_to_patient.numel())
    ):
        raise ValueError("tile_slide_idx contains a slide outside slide_to_patient")

    slide_to_patient = slide_to_patient.to(device=tile_features.device)
    slide_ids, tile_to_slide = torch.unique(tile_slide_idx, sorted=True, return_inverse=True)
    slide_sums = tile_features.new_zeros((slide_ids.numel(), tile_features.shape[1]))
    slide_sums.index_add_(0, tile_to_slide, tile_features)
    slide_counts = tile_features.new_zeros((slide_ids.numel(), 1))
    slide_counts.index_add_(0, tile_to_slide, tile_features.new_ones((tile_features.shape[0], 1)))
    slide_means = slide_sums / slide_counts

    slide_patients = slide_to_patient.index_select(0, slide_ids)
    patient_ids, slide_to_patient_idx = torch.unique(slide_patients, sorted=True, return_inverse=True)
    patient_sums = tile_features.new_zeros((patient_ids.numel(), tile_features.shape[1]))
    patient_sums.index_add_(0, slide_to_patient_idx, slide_means)
    patient_counts = tile_features.new_zeros((patient_ids.numel(), 1))
    patient_counts.index_add_(0, slide_to_patient_idx, tile_features.new_ones((slide_ids.numel(), 1)))
    patient_means = patient_sums / patient_counts
    return Hierarchy(slide_ids, slide_means, patient_ids, patient_means, patient_sums, patient_counts)


def teacher_value_student_gradient(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError("student and teacher hierarchy shapes differ")
    return teacher.detach() + (student - student.detach())


class HierarchicalCentroidBank(nn.Module):
    """Transactional, non-checkpointed EMA history for slide and patient centroids."""

    _BUFFER_NAMES = (
        "slide_to_patient",
        "slide_centroids",
        "slide_updates",
        "patient_sums",
        "patient_slide_counts",
        "observed_tile_count",
        "caption_present_tile_count",
    )

    def __init__(self, slide_to_patient: torch.Tensor, feature_dim: int, momentum: float):
        super().__init__()
        if slide_to_patient.ndim != 1 or slide_to_patient.dtype != torch.int64:
            raise ValueError("slide_to_patient must be a one-dimensional int64 tensor")
        if slide_to_patient.numel() and torch.any(slide_to_patient < 0):
            raise ValueError("slide_to_patient IDs must be nonnegative")
        if not isinstance(feature_dim, Integral) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(momentum, (float, int)) or not math.isfinite(float(momentum)) or not 0.0 <= float(momentum) <= 1.0:
            raise ValueError("momentum must be finite and lie in [0, 1]")

        mapping = slide_to_patient.detach().clone()
        patient_ids = torch.unique(mapping, sorted=True)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.register_buffer("slide_to_patient", mapping, persistent=False)
        self.register_buffer(
            "slide_centroids",
            torch.zeros((mapping.numel(), self.feature_dim), dtype=torch.float32, device=mapping.device),
            persistent=False,
        )
        self.register_buffer(
            "slide_updates",
            torch.zeros(mapping.numel(), dtype=torch.int64, device=mapping.device),
            persistent=False,
        )
        self.register_buffer(
            "patient_sums",
            torch.zeros((patient_ids.numel(), self.feature_dim), dtype=torch.float32, device=mapping.device),
            persistent=False,
        )
        self.register_buffer(
            "patient_slide_counts",
            torch.zeros(patient_ids.numel(), dtype=torch.int64, device=mapping.device),
            persistent=False,
        )
        self.register_buffer("observed_tile_count", torch.zeros((), dtype=torch.int64, device=mapping.device), persistent=False)
        self.register_buffer("caption_present_tile_count", torch.zeros((), dtype=torch.int64, device=mapping.device), persistent=False)

    @staticmethod
    def _ids_are_sorted_and_unique(ids: torch.Tensor) -> bool:
        return ids.numel() == 0 or bool(torch.all(ids[1:] > ids[:-1]))

    def _patient_id_table(self) -> torch.Tensor:
        return torch.unique(self.slide_to_patient, sorted=True)

    def _patient_cache_indices(self, patient_ids: torch.Tensor) -> torch.Tensor:
        all_patient_ids = self._patient_id_table()
        cache_indices = torch.searchsorted(all_patient_ids, patient_ids)
        if torch.any(cache_indices >= all_patient_ids.numel()) or not torch.equal(
            all_patient_ids.index_select(0, cache_indices), patient_ids
        ):
            raise ValueError("proposal contains a patient outside slide_to_patient")
        return cache_indices

    def _validate_hierarchy(self, hierarchy: Hierarchy) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(hierarchy, Hierarchy):
            raise ValueError("propose expects a Hierarchy")
        slide_ids, slide_means, patient_ids, patient_means, patient_sums, patient_counts = hierarchy
        if slide_ids.ndim != 1 or slide_ids.dtype != torch.int64 or slide_ids.device != self.slide_centroids.device:
            raise ValueError("hierarchy slide IDs must be device-matched int64")
        if patient_ids.ndim != 1 or patient_ids.dtype != torch.int64 or patient_ids.device != self.slide_centroids.device:
            raise ValueError("hierarchy patient IDs must be device-matched int64")
        if not self._ids_are_sorted_and_unique(slide_ids) or not self._ids_are_sorted_and_unique(patient_ids):
            raise ValueError("hierarchy IDs must be sorted and unique")
        if slide_means.shape != (slide_ids.numel(), self.feature_dim) or patient_means.shape != (patient_ids.numel(), self.feature_dim) or patient_sums.shape != patient_means.shape or patient_counts.shape != (patient_ids.numel(), 1):
            raise ValueError("hierarchy feature shapes do not match this centroid bank")
        if not slide_means.is_floating_point() or not patient_means.is_floating_point() or not patient_sums.is_floating_point() or not patient_counts.is_floating_point():
            raise ValueError("hierarchy means must be floating point")
        if slide_means.device != self.slide_centroids.device or patient_means.device != self.slide_centroids.device or patient_sums.device != self.slide_centroids.device or patient_counts.device != self.slide_centroids.device:
            raise ValueError("hierarchy means must match the centroid bank device")
        if not torch.isfinite(slide_means).all() or not torch.isfinite(patient_means).all() or not torch.isfinite(patient_sums).all() or not torch.isfinite(patient_counts).all():
            raise ValueError("hierarchy means must be finite")
        if slide_ids.numel() and (torch.any(slide_ids < 0) or torch.any(slide_ids >= self.slide_to_patient.numel())):
            raise ValueError("hierarchy contains a slide outside slide_to_patient")

        slide_patients = self.slide_to_patient.index_select(0, slide_ids)
        expected_patient_ids = torch.unique(slide_patients, sorted=True)
        if not torch.equal(patient_ids, expected_patient_ids):
            raise ValueError("hierarchy patient IDs do not match slide_to_patient")
        if torch.any(patient_counts <= 0) or not torch.equal(patient_means, patient_sums / patient_counts):
            raise ValueError("hierarchy patient means do not match its slide means")
        return slide_ids, slide_means.to(dtype=torch.float32), slide_patients

    def _candidate_cache(
        self,
        slide_ids: torch.Tensor,
        next_slide_centroids: torch.Tensor,
        patient_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        old_slide_centroids = self.slide_centroids.index_select(0, slide_ids)
        old_slide_updates = self.slide_updates.index_select(0, slide_ids)
        slide_patients = self.slide_to_patient.index_select(0, slide_ids)
        expected_patient_ids, slide_to_patient_idx = torch.unique(slide_patients, sorted=True, return_inverse=True)
        if not torch.equal(patient_ids, expected_patient_ids):
            raise ValueError("proposal patient IDs do not match proposal slide IDs")

        patient_deltas = self.patient_sums.new_zeros((patient_ids.numel(), self.feature_dim))
        patient_deltas.index_add_(0, slide_to_patient_idx, next_slide_centroids - old_slide_centroids)
        new_slide_counts = self.patient_slide_counts.new_zeros(patient_ids.numel())
        new_slide_counts.index_add_(0, slide_to_patient_idx, (old_slide_updates == 0).to(torch.int64))
        cache_indices = self._patient_cache_indices(patient_ids)
        return (
            cache_indices,
            self.patient_sums.index_select(0, cache_indices) + patient_deltas,
            self.patient_slide_counts.index_select(0, cache_indices) + new_slide_counts,
        )

    def propose(self, hierarchy: Hierarchy) -> CentroidProposal:
        """Compute an EMA/cache update without changing any bank buffer."""
        slide_ids, slide_means, slide_patients = self._validate_hierarchy(hierarchy)
        old_centroids = self.slide_centroids.index_select(0, slide_ids)
        old_updates = self.slide_updates.index_select(0, slide_ids)
        next_slide_centroids = torch.where(
            (old_updates > 0).unsqueeze(1),
            old_centroids * self.momentum + slide_means * (1.0 - self.momentum),
            slide_means,
        )
        patient_ids, _ = torch.unique(slide_patients, sorted=True, return_inverse=True)
        _, patient_sums, patient_counts = self._candidate_cache(slide_ids, next_slide_centroids, patient_ids)
        patient_centroids = patient_sums / patient_counts.to(dtype=patient_sums.dtype).unsqueeze(1)
        return CentroidProposal(
            slide_ids, next_slide_centroids, patient_ids, patient_centroids, patient_sums, patient_counts
        )

    @staticmethod
    def _coverage_count(value: object, name: str) -> int:
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise ValueError(f"{name} must be a scalar integer")
            value = int(value.item())
        elif isinstance(value, Integral) and not isinstance(value, bool):
            value = int(value)
        else:
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return value

    def _validate_current_state(self) -> None:
        if not torch.isfinite(self.slide_centroids).all() or not torch.isfinite(self.patient_sums).all():
            raise ValueError("centroid cache contains non-finite values")
        if torch.any(self.slide_updates < 0) or torch.any(self.patient_slide_counts < 0):
            raise ValueError("centroid cache contains negative counts")
        if self.observed_tile_count.item() < 0 or self.caption_present_tile_count.item() < 0:
            raise ValueError("centroid coverage counters must be nonnegative")
        if self.caption_present_tile_count.item() > self.observed_tile_count.item():
            raise ValueError("caption coverage cannot exceed observed tiles")

    def _validate_proposal(self, proposal: CentroidProposal) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(proposal, CentroidProposal):
            raise ValueError("commit expects a CentroidProposal")
        slide_ids, next_slide_centroids, patient_ids, patient_centroids, candidate_sums, candidate_counts = proposal
        if slide_ids.ndim != 1 or slide_ids.dtype != torch.int64 or slide_ids.device != self.slide_centroids.device:
            raise ValueError("proposal slide IDs must be device-matched int64")
        if patient_ids.ndim != 1 or patient_ids.dtype != torch.int64 or patient_ids.device != self.slide_centroids.device:
            raise ValueError("proposal patient IDs must be device-matched int64")
        if not self._ids_are_sorted_and_unique(slide_ids) or not self._ids_are_sorted_and_unique(patient_ids):
            raise ValueError("proposal IDs must be sorted and unique")
        if next_slide_centroids.shape != (slide_ids.numel(), self.feature_dim) or patient_centroids.shape != (patient_ids.numel(), self.feature_dim) or candidate_sums.shape != patient_centroids.shape or candidate_counts.shape != (patient_ids.numel(),):
            raise ValueError("proposal centroid shapes do not match this centroid bank")
        if next_slide_centroids.dtype != torch.float32 or patient_centroids.dtype != torch.float32 or candidate_sums.dtype != torch.float32 or candidate_counts.dtype != torch.int64:
            raise ValueError("proposal centroids must be float32")
        if next_slide_centroids.device != self.slide_centroids.device or patient_centroids.device != self.slide_centroids.device or candidate_sums.device != self.slide_centroids.device or candidate_counts.device != self.slide_centroids.device:
            raise ValueError("proposal centroids must match the centroid bank device")
        if not torch.isfinite(next_slide_centroids).all() or not torch.isfinite(patient_centroids).all() or not torch.isfinite(candidate_sums).all():
            raise ValueError("proposal centroids must be finite")
        if slide_ids.numel() and (torch.any(slide_ids < 0) or torch.any(slide_ids >= self.slide_to_patient.numel())):
            raise ValueError("proposal contains a slide outside slide_to_patient")
        cache_indices = self._patient_cache_indices(patient_ids)
        if torch.any(candidate_counts <= 0):
            raise ValueError("proposal leaves a patient without cached slides")

        # Replay the exact staged float32 quotient, not a different CPU float64 calculation.
        reconstructed = candidate_sums / candidate_counts.to(dtype=candidate_sums.dtype).unsqueeze(1)
        if not torch.equal(reconstructed, patient_centroids):
            raise ValueError("proposal patient centroids do not match the cache reconstruction")
        return cache_indices, candidate_sums, candidate_counts, slide_ids

    def commit(self, proposal: CentroidProposal, observed_tiles: int, caption_present_tiles: int) -> None:
        """Validate every value, then atomically make the staged proposal visible."""
        observed = self._coverage_count(observed_tiles, "observed_tiles")
        caption_present = self._coverage_count(caption_present_tiles, "caption_present_tiles")
        if caption_present > observed:
            raise ValueError("caption_present_tiles cannot exceed observed_tiles")
        self._validate_current_state()
        cache_indices, candidate_sums, candidate_counts, slide_ids = self._validate_proposal(proposal)

        with torch.no_grad():
            self.slide_centroids.index_copy_(0, slide_ids, proposal.next_slide_centroids.detach())
            self.slide_updates.index_add_(0, slide_ids, torch.ones_like(slide_ids, dtype=torch.int64))
            self.patient_sums.index_copy_(0, cache_indices, candidate_sums.detach())
            self.patient_slide_counts.index_copy_(0, cache_indices, candidate_counts.detach())
            self.observed_tile_count.add_(observed)
            self.caption_present_tile_count.add_(caption_present)

    def export_state(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().clone() for name in self._BUFFER_NAMES}

    def restore_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore a complete exported snapshot after validating it without mutation."""
        if not isinstance(state, dict) or set(state) != set(self._BUFFER_NAMES):
            raise ValueError("centroid state must contain exactly the bank buffers")
        for name in self._BUFFER_NAMES:
            source, destination = state[name], getattr(self, name)
            if not isinstance(source, torch.Tensor):
                raise ValueError(f"centroid state {name} must be a tensor")
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError(f"centroid state {name} has a shape or dtype mismatch")
        candidate_mapping = state["slide_to_patient"].to(device=self.slide_to_patient.device)
        if not torch.equal(candidate_mapping, self.slide_to_patient):
            raise ValueError("centroid state slide_to_patient mapping mismatch")
        if not torch.isfinite(state["slide_centroids"]).all() or not torch.isfinite(state["patient_sums"]).all():
            raise ValueError("centroid state contains non-finite values")
        if torch.any(state["slide_updates"] < 0) or torch.any(state["patient_slide_counts"] < 0):
            raise ValueError("centroid state contains negative counts")
        observed = state["observed_tile_count"].item()
        caption_present = state["caption_present_tile_count"].item()
        if observed < 0 or caption_present < 0 or caption_present > observed:
            raise ValueError("centroid state has invalid coverage counters")

        with torch.no_grad():
            for name in self._BUFFER_NAMES:
                destination = getattr(self, name)
                destination.copy_(state[name].detach().to(device=destination.device))


@dataclass
class CentroidRuntime:
    head: nn.Module
    bank: HierarchicalCentroidBank
    target_digest: str
    mapping_digest: str
    gate_report: dict | None = None
    gate_checked: bool = False


_CENTROID_KEYS = {
    "enabled", "weight", "ramp_start", "ramp_len", "feature_blocks", "input_dim",
    "head_hidden_dim", "forward_source", "gradient_source", "history",
}
_CENTROID_HISTORY_KEYS = {
    "level", "momentum", "min_slide_updates", "min_sample_weighted_coverage",
    "min_geometry_patients", "min_effective_rank", "min_participation_ratio",
    "max_mean_offdiag_cosine", "gate_version",
}


def validate_centroid_config(raw_centroid_cfg, legacy_molcap_cfg=None):
    """Return an approved C1 block, or None for the bit-exact C0 path."""
    if raw_centroid_cfg is None:
        return None
    if not isinstance(raw_centroid_cfg, dict):
        raise ValueError("molcap.centroid must be a mapping")
    if not raw_centroid_cfg.get("enabled", False):
        return None
    if not legacy_molcap_cfg or not legacy_molcap_cfg.get("enabled", False):
        raise ValueError("an enabled centroid objective requires enabled legacy MolCap")
    if set(raw_centroid_cfg) != _CENTROID_KEYS:
        raise ValueError("enabled molcap.centroid must contain exactly the approved keys")
    history = raw_centroid_cfg["history"]
    if not isinstance(history, dict) or set(history) != _CENTROID_HISTORY_KEYS:
        raise ValueError("centroid history must contain exactly the approved keys")
    blocks = raw_centroid_cfg["feature_blocks"]
    if not isinstance(blocks, list) or len(blocks) != 4 or any(
        not isinstance(block, Integral) or isinstance(block, bool) for block in blocks
    ):
        raise ValueError("feature_blocks must contain four integer block IDs")
    if len(set(blocks)) != len(blocks) or any(block < 0 or block > 11 for block in blocks):
        raise ValueError("feature_blocks must be unique IDs in [0, 11]")
    if int(raw_centroid_cfg["input_dim"]) != 384 * len(blocks) or int(raw_centroid_cfg["input_dim"]) != 1536:
        raise ValueError("centroid input_dim must equal 384 * len(feature_blocks) == 1536")
    if int(raw_centroid_cfg["head_hidden_dim"]) != 512:
        raise ValueError("centroid head_hidden_dim must be 512")
    if raw_centroid_cfg["forward_source"] != "teacher":
        raise ValueError("centroid forward_source must be teacher")
    if raw_centroid_cfg["gradient_source"] != "student_identity_ste":
        raise ValueError("centroid gradient_source must be student_identity_ste")
    if history["level"] != "slide_then_patient" or history["gate_version"] != "matched_latest_v1":
        raise ValueError("centroid history level and gate_version are fixed")
    for name in ("weight", "ramp_start", "ramp_len"):
        if not isinstance(raw_centroid_cfg[name], (float, int)) or not math.isfinite(float(raw_centroid_cfg[name])):
            raise ValueError(f"centroid {name} must be finite")
    if float(raw_centroid_cfg["ramp_len"]) <= 0 or not 0.0 <= float(raw_centroid_cfg["ramp_start"]) <= 1.0:
        raise ValueError("centroid ramp must have a positive length and a start in [0, 1]")
    legacy_ramp_end = float(legacy_molcap_cfg["ramp_start"]) + float(legacy_molcap_cfg["ramp_len"])
    if float(raw_centroid_cfg["ramp_start"]) < legacy_ramp_end:
        raise ValueError("centroid ramp_start must be no earlier than the end of the legacy MolCap ramp")
    for name in ("momentum", "min_sample_weighted_coverage", "max_mean_offdiag_cosine"):
        if not isinstance(history[name], (float, int)) or not math.isfinite(float(history[name])):
            raise ValueError(f"centroid history {name} must be finite")
    if not 0.0 <= float(history["momentum"]) <= 1.0:
        raise ValueError("centroid history momentum must lie in [0, 1]")
    for name in ("min_slide_updates", "min_geometry_patients", "min_effective_rank", "min_participation_ratio"):
        if not isinstance(history[name], Integral) or isinstance(history[name], bool) or int(history[name]) <= 0:
            raise ValueError(f"centroid history {name} must be a positive integer")
    return raw_centroid_cfg


def centroid_scale(sfrac, centroid_cfg) -> float:
    """C1 activates strictly after the legacy MolCap ramp is complete."""
    start = float(centroid_cfg["ramp_start"])
    if float(sfrac) <= start:
        return 0.0
    return linear_ramp(float(sfrac), start, float(centroid_cfg["ramp_len"]))


def build_centroid_runtime(centroid_cfg, legacy_molcap_cfg, train_ds, target_dim, device):
    """Create C1-only state without touching C0's dataset/device/RNG state."""
    if centroid_cfg is None or not centroid_cfg.get("enabled", False):
        return None
    centroid_cfg = validate_centroid_config(centroid_cfg, legacy_molcap_cfg)
    return CentroidRuntime(
        head=seed_neutral_molcap_head(int(centroid_cfg["input_dim"]), target_dim, device),
        bank=HierarchicalCentroidBank(
            torch.as_tensor(train_ds.molcap_slide_to_patient, dtype=torch.int64, device=device),
            feature_dim=int(centroid_cfg["input_dim"]),
            momentum=float(centroid_cfg["history"]["momentum"]),
        ).to(device),
        target_digest=train_ds.molcap_target_digest,
        mapping_digest=train_ds.molcap_centroid_mapping_digest,
    )


def patient_caption_targets(tile_targets, tile_present, tile_slide_idx, slide_to_patient, patient_ids):
    """Pool only caption-bearing tiles into normalized targets for sorted patient IDs."""
    if tile_targets.ndim != 2 or tile_present.ndim != 1 or tile_slide_idx.ndim != 1:
        raise ValueError("caption targets, presence, and slide IDs must be rank 2, 1, and 1")
    if tile_targets.shape[0] != tile_present.numel() or tile_targets.shape[0] != tile_slide_idx.numel():
        raise ValueError("caption target inputs must agree on tile count")
    if tile_slide_idx.dtype != torch.int64 or patient_ids.ndim != 1 or patient_ids.dtype != torch.int64:
        raise ValueError("caption slide and patient IDs must be int64")
    if tile_targets.device != tile_present.device or tile_targets.device != tile_slide_idx.device:
        raise ValueError("caption target inputs must share a device")
    if patient_ids.device != tile_targets.device:
        raise ValueError("caption patient IDs must share the target device")
    if patient_ids.numel() and not bool(torch.all(patient_ids[1:] > patient_ids[:-1])):
        raise ValueError("caption patient IDs must be sorted and unique")
    mapping = slide_to_patient.to(device=tile_targets.device)
    if mapping.ndim != 1 or mapping.dtype != torch.int64:
        raise ValueError("slide_to_patient must be one-dimensional int64")
    if tile_slide_idx.numel() and (torch.any(tile_slide_idx < 0) or torch.any(tile_slide_idx >= mapping.numel())):
        raise ValueError("caption slide ID is outside slide_to_patient")
    tile_patient_ids = mapping.index_select(0, tile_slide_idx)
    patient_indices = torch.searchsorted(patient_ids, tile_patient_ids)
    if patient_indices.numel() and (torch.any(patient_indices >= patient_ids.numel()) or not torch.equal(
        patient_ids.index_select(0, patient_indices), tile_patient_ids
    )):
        raise ValueError("caption tile references a patient absent from patient_ids")
    present = tile_present.to(dtype=torch.bool)
    target_sums = tile_targets.new_zeros((patient_ids.numel(), tile_targets.shape[1]))
    target_counts = tile_targets.new_zeros((patient_ids.numel(), 1))
    target_sums.index_add_(0, patient_indices[present], tile_targets[present])
    target_counts.index_add_(0, patient_indices[present], tile_targets.new_ones((int(present.sum().item()), 1)))
    presence = (target_counts.squeeze(1) > 0).to(dtype=torch.float32)
    targets = F.normalize(target_sums, p=2, dim=-1)
    return targets, presence


def _centroid_coverage(bank: HierarchicalCentroidBank) -> float:
    observed = int(bank.observed_tile_count.item())
    return 0.0 if observed == 0 else float(bank.caption_present_tile_count.item()) / observed


def _finite_float(value: torch.Tensor | float | int) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("centroid gate produced a non-finite metric")
    return value


def run_centroid_gate(bank, centroid_cfg, target_digest, mapping_digest, output_path):
    """Validate C1 history and atomically publish a finite gate report on success."""
    final_path = Path(output_path)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    try:
        history = centroid_cfg["history"]
        bank._validate_current_state()
        mature_slides = bank.slide_updates >= int(history["min_slide_updates"])
        all_patient_ids = torch.unique(bank.slide_to_patient, sorted=True)
        mature_patient_ids = torch.unique(bank.slide_to_patient[mature_slides], sorted=True)
        mature_patient_count = int(mature_patient_ids.numel())
        cache_indices = torch.searchsorted(all_patient_ids, mature_patient_ids)
        if mature_patient_count:
            patient_centroids = (
                bank.patient_sums.index_select(0, cache_indices) /
                bank.patient_slide_counts.index_select(0, cache_indices).to(dtype=bank.patient_sums.dtype).unsqueeze(1)
            ).detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(patient_centroids).all():
                raise ValueError("centroid gate found non-finite patient centroids")
            singular_values = torch.linalg.svdvals(patient_centroids)
            spectrum = singular_values.square()
            spectrum_sum = spectrum.sum()
            if spectrum_sum > 0:
                probabilities = spectrum / spectrum_sum
                positive = probabilities > 0
                effective_rank = torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum())
                participation_ratio = spectrum_sum.square() / spectrum.square().sum()
            else:
                effective_rank = patient_centroids.new_zeros(())
                participation_ratio = patient_centroids.new_zeros(())
            if mature_patient_count > 1:
                normalized = F.normalize(patient_centroids, p=2, dim=-1)
                cosine = normalized @ normalized.T
                mean_offdiag_cosine = (cosine.sum() - cosine.diagonal().sum()) / (mature_patient_count * (mature_patient_count - 1))
            else:
                mean_offdiag_cosine = patient_centroids.new_zeros(())
        else:
            effective_rank = participation_ratio = mean_offdiag_cosine = 0.0
        coverage = _centroid_coverage(bank)
        report = {
            "sample_weighted_coverage": _finite_float(coverage),
            "mature_patient_count": mature_patient_count,
            "effective_rank": _finite_float(effective_rank),
            "participation_ratio": _finite_float(participation_ratio),
            "mean_offdiag_cosine": _finite_float(mean_offdiag_cosine),
            "target_digest": target_digest,
            "mapping_digest": mapping_digest,
            "gate_version": history["gate_version"],
        }
        failures = []
        for name, actual, minimum in (
            ("sample_weighted_coverage", report["sample_weighted_coverage"], float(history["min_sample_weighted_coverage"])),
            ("mature_patient_count", report["mature_patient_count"], int(history["min_geometry_patients"])),
            ("effective_rank", report["effective_rank"], float(history["min_effective_rank"])),
            ("participation_ratio", report["participation_ratio"], float(history["min_participation_ratio"])),
        ):
            if actual < minimum:
                failures.append(name)
        if report["mean_offdiag_cosine"] > float(history["max_mean_offdiag_cosine"]):
            failures.append("mean_offdiag_cosine")
        report["passed"] = not failures
        report["failure_reason"] = None if report["passed"] else ", ".join(failures)
        if not report["passed"]:
            raise RuntimeError(f"centroid gate failed: {report['failure_reason']}")
        tmp_path.write_text(json.dumps(report, allow_nan=False, sort_keys=True))
        os.replace(tmp_path, final_path)
        return report
    except Exception:
        tmp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise


def compute_centroid_loss(runtime, centroid_cfg, teacher_hierarchy, student_hierarchy, patient_targets, patient_present, sfrac, gate_path):
    """Stage a detached-teacher EMA proposal and C1-only STE loss without committing history."""
    detached_teacher = Hierarchy(*(tensor.detach() for tensor in teacher_hierarchy))
    proposal = runtime.bank.propose(detached_teacher)
    coverage = _centroid_coverage(runtime.bank)
    scale = centroid_scale(sfrac, centroid_cfg)
    if float(sfrac) > float(centroid_cfg["ramp_start"]) and not runtime.gate_checked:
        runtime.gate_report = run_centroid_gate(
            runtime.bank, centroid_cfg, runtime.target_digest, runtime.mapping_digest, gate_path,
        )
        runtime.gate_checked = True
    loss = student_hierarchy.patient_means.new_zeros(())
    if scale > 0.0 and runtime.gate_checked and runtime.gate_report["passed"]:
        forward_patient_features = teacher_value_student_gradient(
            student_hierarchy.patient_means, proposal.patient_centroids,
        )
        loss = float(centroid_cfg["weight"]) * scale * molcap_loss(
            runtime.head, forward_patient_features, patient_targets, patient_present, views=1,
        )
    return CentroidStep(loss, proposal, scale, coverage, int(proposal.patient_ids.numel()))


def attach_centroid_checkpoint_state(payload, runtime):
    """Keep C0 payload bytes/keys unchanged while adding the complete C1 state."""
    if runtime is None:
        return payload
    payload = dict(payload)
    payload.update({
        "centroid_head": {key: value.detach().cpu().clone() for key, value in runtime.head.state_dict().items()},
        "centroid_history": {key: value.detach().cpu().clone() for key, value in runtime.bank.export_state().items()},
        "centroid_gate_report": runtime.gate_report,
        "centroid_target_digest": runtime.target_digest,
        "centroid_mapping_digest": runtime.mapping_digest,
    })
    return payload


# Orchestrates one pretraining run: setup, train+probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    repo_dir = Path(__file__).resolve().parent
    labless_autosubmit_file = maybe_arm_labless_autosubmit(cfg, repo_dir)
    train_cfg = cfg["train"]
    dino_cfg = cfg["dino"]
    # FINO metadata-guidance: select factors + signs (float; + encourage M+ / - suppress M-). fino_meta (built or
    # copied beside the dataset by prepare.py) holds per-factor barcode maps + cardinalities (n) / vector dims.
    fino_cfg = cfg["fino"] if (cfg.get("fino") or {}).get("enabled") else None
    raw_molcap_cfg = cfg.get("molcap") or {}
    molcap_cfg = raw_molcap_cfg if raw_molcap_cfg.get("enabled") else None
    centroid_cfg = validate_centroid_config(raw_molcap_cfg.get("centroid"), molcap_cfg)
    fino_disc = [(f, float(s)) for f, s in fino_cfg.get("discrete", [])] if fino_cfg else []
    fino_cont = [(f, float(s)) for f, s in fino_cfg.get("continuous", [])] if fino_cfg else []
    fino_meta = json.loads((Path(cfg["data"]["dataset_dir"]) / "fino_meta.json").read_text()) if fino_cfg else {"n": {}, "cont_dim": {}}
    # FINO two-phase: freeze the backbone (except patch_embed) for the first this-fraction of the run so the DINO/JEPA
    # heads + metadata prototypes/predictors converge against a fixed target before they steer the encoder. 0 = off.
    freeze_backbone_frac = float(dino_cfg.get("freeze_backbone_fraction", 0.0))
    # JEPA-T: optionally condition the JEPA predictor on a discrete factor (must be in fino.discrete so its per-tile
    # label rides in the batch). cond_col indexes that factor's column in batch["meta_disc"].
    jepa_cond = fino_cfg.get("jepa_cond") if fino_cfg else None
    cond_col = [f for f, _ in fino_disc].index(jepa_cond) if jepa_cond else None
    save_every = train_cfg["save_every"]
    save_checkpoints = save_every is not None
    device = torch.device("cuda")
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    variant = cfg["model"]["type"]
    student_backbone = load_dinov2_pretrained(DinoV2ViT(variant=variant, drop_path_rate=dino_cfg["drop_path_rate"])).to(device)
    teacher_backbone = deepcopy(student_backbone)
    teacher_backbone.train(False)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    student_dino_head = DINOHead(student_backbone.embed_dim, 131072, dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3).to(device)
    teacher_dino_head = deepcopy(student_dino_head)
    student_predictor = JEPAPredictor(student_backbone.embed_dim, depth=int(dino_cfg["jepa_pred_depth"]), width=int(dino_cfg["jepa_pred_width"]), n_cond=(fino_meta["n"][jepa_cond] if jepa_cond else 0)).to(device)
    molcap_head = seed_neutral_molcap_head(student_backbone.embed_dim, int(molcap_cfg["target_dim"]), device) if molcap_cfg else None
    for p in teacher_dino_head.parameters():
        p.requires_grad = False
    backbone_activated_params = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    # FINO continuous-factor predictors (phi -> vector regressors); their params join the optimizer.
    predictors = {f: nn.Sequential(nn.Linear(student_backbone.embed_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, fino_meta.get("cont_dim", {}).get(f, 1))).to(device) for f, _ in fino_cont}
    # AdamW param groups carry per-parameter LR/WD multipliers (LWD + patch_embed + biases-no-WD).
    param_groups = build_param_groups(student_backbone, student_dino_head, student_predictor, dino_cfg["layerwise_decay"], dino_cfg["patch_embed_lr_mult"])
    if predictors:
        param_groups.append({"params": [p for m in predictors.values() for p in m.parameters()], "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    if molcap_head:
        param_groups.append({"params": list(molcap_head.parameters()), "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    opt = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, dino_cfg["adam_beta2"]))
    # FINO prototype banks: one unit vector per discrete-factor value, EMA-updated from teacher CLS in compute_losses.
    protos = {f: F.normalize(torch.randn(fino_meta["n"][f], student_backbone.embed_dim, device=device), dim=-1) for f, _ in fino_disc} if fino_cfg else {}
    # FINO grad-equalisation EMA bank (one running grad-norm per factor); init 1.0 -> s_t~1 early. Not checkpointed
    # (mu=0.99 -> ~100-step memory, re-warms quickly on resume). Used only when fino.grad_equalize is set.
    grad_eq_ema = {f: torch.ones((), device=device) for f, _ in (fino_disc + fino_cont)} if fino_cfg else {}
    step = 0
    batch_size = int(train_cfg["batch_size"])
    max_train_samples = int(train_cfg["max_train_samples"])
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
    wandb_name = cfg["project"]["name"]
    if labless_autosubmit_file:
        wandb_name = json.loads(Path(labless_autosubmit_file).read_text()).get("run_name") or wandb_name
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    latest_checkpoint_path = output_dir / "latest.pt"
    # Fresh launches always start from scratch and wipe output_dir.
    resume_path = Path(train_cfg["resume"]) if train_cfg["resume"] else None
    if resume_path is None and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    wandb_meta = None
    if resume_path is not None:
        print(f"{console_prefix()} Resume  loading checkpoint: {resume_path}", flush=True)
        # Resume restores training progress, optimizer state, and wandb identity.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        student_backbone.load_state_dict(checkpoint["model"])
        teacher_backbone.load_state_dict(checkpoint["model_ema"])
        student_dino_head.load_state_dict(checkpoint["dino_head"])
        teacher_dino_head.load_state_dict(checkpoint["dino_head_ema"])
        student_predictor.load_state_dict(checkpoint["predictor"])
        if molcap_head:
            molcap_head.load_state_dict(checkpoint["molcap_head"])
        if centroid_cfg is None:
            opt.load_state_dict(checkpoint["opt"])
        if fino_cfg:
            protos = {k: v.to(device) for k, v in checkpoint["protos"].items()}
            for f, mdl in predictors.items():
                mdl.load_state_dict(checkpoint["predictors"][f])
        step = int(checkpoint["step"])
        examples_seen = int(checkpoint["examples_seen"])
        visible_patch_presentations = int(checkpoint["visible_patch_presentations"])
        train_flops = int(checkpoint["train_flops"])
        wandb_meta = dict(checkpoint["wandb"])
    wandb_init = {
        "project": "nanopath",
        "name": wandb_name,
        "dir": str(wandb_dir),
        "config": cfg,
        "settings": wandb.Settings(
            console="wrap",
            x_file_stream_transmit_interval=5,
        ),
    }
    if wandb_meta is not None:
        wandb_init["id"] = wandb_meta["id"]
        wandb_init["resume"] = "must"
    wandb_run = wandb.init(**wandb_init)
    for key in ("probe/target_flops", "probe/wall_seconds"):
        wandb_run.define_metric(key, hidden=True, overwrite=True)
    print(
        f"{console_prefix()} Run  start: {wandb_name}  "
        f"config: {cfg['config_path']}  batch_size: {batch_size}  max_train_samples: {max_train_samples}  "
        f"max_train_flops: {train_cfg['max_train_flops']}  "
        f"probe_count: {cfg['probe']['count']}  warmup_fraction: {dino_cfg['warmup_fraction']}  "
        f"lr: {dino_cfg['lr']}  adam_beta2: {dino_cfg['adam_beta2']}  kde_loss_weight: {dino_cfg['kde_loss_weight']}  "
        f"kde_concentration: {dino_cfg['kde_concentration']}  drop_path: {dino_cfg['drop_path_rate']}  "
        f"layerwise_decay: {dino_cfg['layerwise_decay']}",
        flush=True,
    )
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    git_remote = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=repo_dir, text=True, capture_output=True, check=False).stdout.strip()
    source_id = f"nanopath-source-{wandb_run.id}"
    artifact_ignore = [
        line.strip() for line in (repo_dir / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ] + [".git/", "baselines/", "slurm/", "AGENTS.md", "CLAUDE.md"]
    ignored_roots = [output_dir.resolve(), wandb_dir.resolve()]

    def artifact_ignored(path):
        if any(path.resolve().is_relative_to(root) for root in ignored_roots):
            return True
        rel_path = path.relative_to(repo_dir)
        if any(part.startswith(".") for part in rel_path.parts):
            return True
        rel, name = rel_path.as_posix(), path.name
        for pat in artifact_ignore:
            pat = pat.rstrip("/") if pat.endswith("/") else pat
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat) or rel == pat or rel.startswith(pat + "/"):
                return True
        return False

    source_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if not artifact_ignored(Path(root) / d))
        for name in sorted(files):
            path = Path(root) / name
            if artifact_ignored(path):
                continue
            rel = path.relative_to(repo_dir)
            source_files.append((path, rel))
    source_snapshot_dir = output_dir / "labless_source"
    if source_snapshot_dir.exists():
        shutil.rmtree(source_snapshot_dir)
    for path, rel in source_files:
        target = source_snapshot_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    wandb_meta = {"entity": wandb_run.entity, "project": "nanopath", "id": wandb_run.id, "name": wandb_name, "url": wandb_run.url,
                  "mode": getattr(wandb_run.settings, "mode", ""), "source_artifact": source_id,
                  "source_dir": str(source_snapshot_dir), "git": {"commit": git_commit, "remote": git_remote}}
    train_ds = TCGATileDataset(cfg, is_train=True)
    centroid_runtime = (
        build_centroid_runtime(centroid_cfg, molcap_cfg, train_ds, int(molcap_cfg["target_dim"]), device)
        if centroid_cfg is not None else None
    )
    if centroid_runtime is not None:
        opt.add_param_group({
            "params": list(centroid_runtime.head.parameters()),
            "lr_mult": 1.0,
            "wd_mult": 1.0,
            "last_layer": False,
        })
        if resume_path is not None:
            if checkpoint.get("centroid_target_digest") != centroid_runtime.target_digest:
                raise ValueError("centroid checkpoint target digest does not match the training dataset")
            if checkpoint.get("centroid_mapping_digest") != centroid_runtime.mapping_digest:
                raise ValueError("centroid checkpoint mapping digest does not match the training dataset")
            centroid_runtime.head.load_state_dict(checkpoint["centroid_head"])
            centroid_runtime.bank.restore_state(checkpoint["centroid_history"])
            centroid_runtime.gate_report = checkpoint["centroid_gate_report"]
            centroid_runtime.gate_checked = centroid_runtime.gate_report is not None
            opt.load_state_dict(checkpoint["opt"])
    val_ds = TCGATileDataset(cfg, is_train=False)
    probe_state = prepare_probe_state(cfg, output_dir) if probe_enabled(cfg) else None

    # Train shuffles + drops partials; the loop never starts a batch that would exceed
    # max_train_samples, so every optimizer step keeps the configured batch size.
    loader_kwargs = dict(batch_size=batch_size, drop_last=True, num_workers=train_cfg["num_workers"], pin_memory=True,
                         prefetch_factor=train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
                         persistent_workers=train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    global_grid = train_cfg["global_size"] // student_backbone.patch_size
    global_patches = global_grid ** 2
    local_patches = (train_cfg["local_size"] // student_backbone.patch_size) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (TILE_SIZE // student_backbone.patch_size) ** 2
    seen_ids = {"sample": set(), "slide": set(), "patient": set()}
    pending_ids = {key: set() for key in seen_ids}

    # cpu_state(m) materializes an on-CPU copy of a module's state_dict for torch.save.
    def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # Full checkpoint (latest.pt) covers explicit train.resume whereas probe checkpoint is a slim
    # weights-only ckpt, given probe.py does not need optimizer or projection heads.
    def checkpoint_payload(next_step, full):
        payload = {"model": cpu_state(student_backbone), "model_ema": cpu_state(teacher_backbone), "step": next_step, "config": cfg}
        if not full:
            return payload
        return attach_centroid_checkpoint_state(
            {**payload, "dino_head": cpu_state(student_dino_head), "dino_head_ema": cpu_state(teacher_dino_head),
             "predictor": cpu_state(student_predictor), "opt": opt.state_dict(),
             "examples_seen": examples_seen, "visible_patch_presentations": visible_patch_presentations,
             "train_flops": train_flops, "wandb": wandb_meta,
             **({"protos": {k: v.cpu() for k, v in protos.items()}, "predictors": {f: cpu_state(m) for f, m in predictors.items()}} if fino_cfg else {}),
             **({"molcap_head": cpu_state(molcap_head)} if molcap_head else {})},
            centroid_runtime,
        )

    def save_latest_checkpoint(checkpoint_step):
        nonlocal last_saved_step
        print(f"{console_prefix()} Checkpoint  [{checkpoint_step}]  save: latest.pt", flush=True)
        tmp_path = latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload(checkpoint_step, full=True), tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
        for stale_checkpoint_path in output_dir.glob("step_*.pt"):
            stale_checkpoint_path.unlink()
        last_saved_step = checkpoint_step

    # Count unique tiles/slides/patients for data-coverage diagnostics.
    def flush_unique_counts():
        for key in seen_ids:
            seen_ids[key].update(pending_ids[key])
            pending_ids[key].clear()
        unique_tiles_seen = len(seen_ids["sample"])
        return {
            "unique_slides_seen": len(seen_ids["slide"]),
            "unique_patients_seen": len(seen_ids["patient"]),
            "unique_tiles_seen": unique_tiles_seen,
            "unique_patches_seen": unique_tiles_seen * unique_tile_patch_count,
        }

    # Compute (dino_loss, jepa_loss, kde) for one batch of (gf, lf) crops with the given masks +
    # schedule values. Used by both the train step and evaluate() (no_grad).
    def compute_losses(gf, lf, b, masks, mask_idx, mask_w, t_temp, k_scale, ckpt=False, meta=None, cond=None,
                       molcap_target=None, molcap_present=None, molcap_scale=0.0, diagnose=False,
                       sfrac=None, centroid_batch=None):
        with torch.no_grad():
            if centroid_runtime is None:
                t = teacher_backbone(gf)
            else:
                t = teacher_backbone(gf, feature_blocks=centroid_cfg["feature_blocks"])
            t_cls = teacher_dino_head(t["x_norm_clstoken"]).chunk(train_cfg["global_views"])
            t_prob = sinkhorn(torch.cat((t_cls[1], t_cls[0])), t_temp).view(2, b, -1)
        if centroid_runtime is None:
            sg = student_backbone(gf, masks=masks, checkpoint=ckpt)
        else:
            sg = student_backbone(gf, masks=masks, checkpoint=ckpt, feature_blocks=centroid_cfg["feature_blocks"])
        sl = student_backbone(lf, checkpoint=ckpt)
        sg_cls, sl_cls = student_dino_head(sg["x_norm_clstoken"]), student_dino_head(sl["x_norm_clstoken"])
        L = train_cfg["local_views"]
        local_loss = sum(dino_ce(x, y) for x in sl_cls.chunk(L) for y in t_prob) / (2 * L + 2)
        global_loss = dino_ce(sg_cls, t_prob.flatten(0, 1)) * 2 / (2 * L + 2)
        target = F.layer_norm(t["x_norm_patchtokens"].flatten(0, 1), (student_backbone.embed_dim,))[mask_idx]
        pred = student_predictor(sg["x_norm_patchtokens"], cond).flatten(0, 1)[mask_idx]
        jepa_loss = F.smooth_l1_loss(pred, target, reduction="none").mean(-1).mul(mask_w).sum() / max(1, b * 2)
        kde = dino_cfg["kde_loss_weight"] * k_scale * sum(kde_loss(x, dino_cfg["kde_concentration"]) for x in sg["x_norm_clstoken"].chunk(train_cfg["global_views"]))
        # FINO metadata guidance on the CLS token (train-only; meta=None in eval), orthogonal to the JEPA patch
        # objective. lambda_meta=0.03/branch; GradScale gates the encoder gradient by the DANN ramp gamma with the
        # per-factor sign (+ M+ encourage / - M- suppress). fp32 island (1/tau=0.023 too sharp for bf16); missing
        # factors masked. Discrete: L2-normed student CLS vs EMA prototype bank (clone-rebind keeps the backward-saved
        # bank valid). Continuous: an MLP regresses the z-scored value.
        meta_loss = sg["x_norm_clstoken"].new_zeros(())
        if meta is not None:
            gamma, md, mc = meta  # md (B,n_disc) int64 (-1 missing); mc {factor: (B,dim) float, nan missing}
            phi_s = F.normalize(sg["x_norm_clstoken"].float(), dim=-1)
            phi_t = F.normalize(t["x_norm_clstoken"].float(), dim=-1)
            terms = []  # (factor, per-branch loss 0.03*L_t); combined below, optionally gradient-equalized
            with torch.autocast(device_type="cuda", enabled=False):
                for j, (f, sign) in enumerate(fino_disc):
                    lab = md[:, j].repeat(train_cfg["global_views"]); ok = lab >= 0  # repeat, NOT interleave
                    if ok.any():
                        logits = (GradScale.apply(phi_s[ok], sign * gamma) @ protos[f].t()) / 0.023
                        terms.append((f, 0.03 * F.cross_entropy(logits, lab[ok])))
                        with torch.no_grad():
                            pt, lt = phi_t[ok], lab[ok]
                            upd = torch.zeros_like(protos[f]).index_add_(0, lt, pt)
                            cnt = torch.zeros(protos[f].shape[0], 1, device=device).index_add_(0, lt, torch.ones_like(pt[:, :1]))
                            seen = cnt.squeeze(1) > 0; new = protos[f].clone()
                            new[seen] = F.normalize(0.99 * new[seen] + 0.01 * (upd[seen] / cnt[seen]), dim=-1); protos[f] = new
                # FINO Eq.3 regresses continuous factors from the RAW backbone CLS; phi_s is L2-normalized (needed only
                # for the cosine discrete branch and it strips the radial magnitude). raw_cls=True feeds the raw CLS.
                cls_cont = sg["x_norm_clstoken"].float() if fino_cfg.get("raw_cls") else phi_s
                for f, sign in fino_cont:
                    val = mc[f].repeat(train_cfg["global_views"], 1); ok = ~torch.isnan(val).any(dim=1)
                    if ok.any():
                        cpred = predictors[f](GradScale.apply(cls_cont[ok], sign * gamma))
                        terms.append((f, 0.03 * F.mse_loss(cpred, val[ok])))
                # FINO Alg A.3 per-branch gradient equalisation: rescale each branch by n_bar/EMA(||dL_t/dCLS||) so the
                # discrete-CE and continuous-MSE gradients reach the encoder at matched magnitudes (detached -> reweight
                # only; geometric-mean target; no-op for <2 branches). grad_eq_ema = per-factor EMA bank (mu=0.99).
                if fino_cfg.get("grad_equalize") and len(terms) > 1:
                    g = {f: torch.autograd.grad(L, sg["x_norm_clstoken"], retain_graph=True)[0].norm() for f, L in terms}
                    for f in g: grad_eq_ema[f] = 0.99 * grad_eq_ema[f] + 0.01 * g[f].detach().float()
                    nbar = torch.exp(torch.stack([grad_eq_ema[f].log() for f, _ in terms]).mean())
                    meta_loss = sum((nbar / grad_eq_ema[f]).detach() * L for f, L in terms)
                else:
                    for _, L in terms: meta_loss = meta_loss + L
        molcap = sg["x_norm_clstoken"].new_zeros(())
        grad_cosine = grad_norm_ratio = molcap
        if molcap_target is not None:
            molcap = float(molcap_cfg["weight"]) * molcap_scale * molcap_loss(
                molcap_head, sg["x_norm_patchtokens"].mean(1), molcap_target, molcap_present, train_cfg["global_views"]
            )
            if diagnose and molcap_scale > 0 and molcap_present.any():
                base = local_loss + global_loss + jepa_loss + kde + meta_loss
                grad_cosine, grad_norm_ratio = gradient_alignment(base, molcap, student_backbone.blocks[-1].attn.qkv.weight)
        if centroid_runtime is None:
            return local_loss + global_loss, jepa_loss, kde, meta_loss, molcap, grad_cosine, grad_norm_ratio
        centroid_step = CentroidStep.zero(sg["x_norm_clstoken"])
        if centroid_batch is not None:
            teacher_tiles = crop_major_tile_mean(
                t["x_norm_probe_features"], train_cfg["global_views"], b,
            )
            student_tiles = crop_major_tile_mean(
                sg["x_norm_probe_features"], train_cfg["global_views"], b,
            )
            teacher_hierarchy = hierarchical_means(
                teacher_tiles, centroid_batch["slide_idx"], centroid_runtime.bank.slide_to_patient,
            )
            student_hierarchy = hierarchical_means(
                student_tiles, centroid_batch["slide_idx"], centroid_runtime.bank.slide_to_patient,
            )
            patient_targets, patient_present = patient_caption_targets(
                molcap_target, molcap_present, centroid_batch["slide_idx"],
                centroid_runtime.bank.slide_to_patient, teacher_hierarchy.patient_ids,
            )
            centroid_step = compute_centroid_loss(
                centroid_runtime, centroid_cfg, teacher_hierarchy, student_hierarchy,
                patient_targets, patient_present, sfrac, output_dir / "centroid_gate.json",
            )
        return local_loss + global_loss, jepa_loss, kde, meta_loss, molcap, grad_cosine, grad_norm_ratio, centroid_step

    # Held-out validation pass: same DINO + JEPA + KDE losses on `val_batches` of the val split.
    # Schedule terms (teacher_temp, kde_scale) drift over training, so read val curves as same-step
    # diagnostics. RNG is snapshotted/restored so val masks don't perturb the next training step.
    def evaluate(eval_step, eval_teacher_temp, eval_kde_scale):
        for m in (student_backbone, student_dino_head, student_predictor):
            m.eval()
        py_rng, cpu_rng, cuda_rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        random.seed(train_cfg["seed"] + eval_step)
        torch.manual_seed(train_cfg["seed"] + eval_step)
        sums = torch.zeros(4, device=device)
        n_batches = 0
        for vb_idx, vbatch in enumerate(val_loader):
            if vb_idx >= int(train_cfg["val_batches"]):
                break
            vg, vl = vbatch["global_views"].to(device, non_blocking=True), vbatch["local_views"].to(device, non_blocking=True)
            b = vg.shape[0]
            with torch.no_grad(), autocast:
                gf, lf = vg.transpose(0, 1).flatten(0, 1), vl.transpose(0, 1).flatten(0, 1)
                masks, mask_idx, mask_w = make_block_mask(b * train_cfg["global_views"], global_grid, device, n_blocks=int(dino_cfg["jepa_blocks"]), block_scale=float(dino_cfg["jepa_block_scale"]))
                if centroid_runtime is None:
                    dino_l, jepa_l, kde_v, _, _, _, _ = compute_losses(gf, lf, b, masks, mask_idx, mask_w, eval_teacher_temp, eval_kde_scale)
                else:
                    dino_l, jepa_l, kde_v, _, _, _, _, _ = compute_losses(
                        gf, lf, b, masks, mask_idx, mask_w, eval_teacher_temp, eval_kde_scale,
                        sfrac=0.0, centroid_batch=None,
                    )
            sums += torch.tensor([float(dino_l), float(jepa_l), float(kde_v), float(dino_l + jepa_l + kde_v)], device=device)
            n_batches += 1
        random.setstate(py_rng)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        return dict(zip(("dino", "jepa", "kde", "total"), (sums / max(1, n_batches)).tolist()))

    # Ingest completed probe result JSONs into metrics.jsonl and wandb.
    def log_probe_results():
        if probe_state is not None:
            collect_probe_results(probe_state, wandb_run, metrics_path)

    # Queue a probe at `checkpoint_step` for the given sample target; no-op if already done.
    def run_probe_at(checkpoint_step, target_samples):
        if probe_state is None or (probe_state["paths"]["results_dir"] / f"step_{checkpoint_step:07d}.json").exists():
            log_probe_results()
            return
        queue_probe_job(probe_state, checkpoint_payload(checkpoint_step, full=False), checkpoint_step, train_flops, min(1.0, target_samples / max_train_samples))
        log_probe_results()

    # Queue the furthest crossed sample milestone so delayed probes do not run on stale checkpoints.
    def maybe_run_probe(checkpoint_step):
        nonlocal next_probe_idx
        if probe_state is None or next_probe_idx >= len(probe_targets) or examples_seen < probe_targets[next_probe_idx]:
            return
        while next_probe_idx + 1 < len(probe_targets) and examples_seen >= probe_targets[next_probe_idx + 1]:
            next_probe_idx += 1
        run_probe_at(checkpoint_step, probe_targets[next_probe_idx])
        next_probe_idx += 1

    log_probe_results()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_train_samples = math.ceil(max_train_samples * dino_cfg["warmup_fraction"])
    # Probe targets are sample milestones: one tile counts once even with many global/local crops.
    probe_count = int(cfg["probe"]["count"]) if probe_enabled(cfg) else 0
    probe_targets = [math.ceil(max_train_samples * (i + 1) / probe_count) for i in range(probe_count)]
    if len(set(probe_targets)) != len(probe_targets):
        raise ValueError(f"probe.count={probe_count} is too large for max_train_samples={max_train_samples}")
    next_probe_idx = 0
    if probe_state is not None:
        completed = [round(float(json.loads(p.read_text()).get("target_fraction", -1)) * max_train_samples) for p in probe_state["paths"]["results_dir"].glob("step_*.json")]
        if completed:
            next_probe_idx = sum(target <= max(completed) for target in probe_targets)
    train_loop_started_at = time.monotonic()
    last_saved_step = step
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if train_cfg["bf16"] else contextlib.nullcontext()
    # Per-step FLOPs are measured once via FlopCounterMode on the first wrapped step (forward +
    # backward + opt.step) and reused for every subsequent step since the shapes don't change.
    # Counts the EMA teacher forward + DINO/JEPA heads, not just the backbone, so the
    # 1e18 leaderboard cap reflects real GPU work.
    measured_flops_per_step = None

    while examples_seen + batch_size <= max_train_samples and train_flops < max_train_flops:
        for batch in train_loader:
            if examples_seen + batch_size > max_train_samples or train_flops >= max_train_flops:
                break
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            student_backbone.train()
            student_dino_head.train()
            student_predictor.train()
            if molcap_head: molcap_head.train()
            if centroid_runtime is not None: centroid_runtime.head.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            visible_now = batch_size * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            # LR warmup uses the 1M-tile sample cap; decay/WD/teacher/freeze/KDE default to the public FLOP budget.
            # But this run hits the sample cap at ~19% of the FLOP budget, so a FLOP-keyed cosine only traverses ~0.11
            # of its arc (LR never anneals, KDE peaks at 0.22, WD ~0.05). lr_key/reg_key="sample" re-key the decay/reg
            # schedules to SAMPLE progress so they complete over the actual 1M-tile run (same fix as the FINO gamma ramp).
            frac = min(1.0, train_flops / max_train_flops)
            sfrac = min(1.0, examples_seen / max_train_samples)
            lr_frac = sfrac if dino_cfg.get("lr_key") == "sample" else frac
            reg_frac = sfrac if dino_cfg.get("reg_key") == "sample" else frac
            warmup = min(1.0, examples_seen / max(1, warmup_train_samples))
            if warmup < 1.0:
                lr = dino_cfg["lr"] * warmup
            else:
                lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], (lr_frac - dino_cfg["warmup_fraction"]) / max(1e-9, 1 - dino_cfg["warmup_fraction"]))
            wd = cosine_schedule(0.04, 0.2, reg_frac)
            teacher_temp = 0.04 + min(1.0, reg_frac / 0.2727) * (0.07 - 0.04)
            last_layer_lr = 0.0 if frac < dino_cfg["freeze_last_layer_fraction"] else lr
            for group in opt.param_groups:
                base_lr = last_layer_lr if group["last_layer"] else lr
                group["lr"] = base_lr * group["lr_mult"]
                group["weight_decay"] = wd * group["wd_mult"]
            masks, mask_idx, mask_w = make_block_mask(batch_size * train_cfg["global_views"], global_grid, device, n_blocks=int(dino_cfg["jepa_blocks"]), block_scale=float(dino_cfg["jepa_block_scale"]))
            kde_scale = min(1.0, max(0.0, (reg_frac - 0.1) / 0.4))
            # Wrap forward + backward + opt.step in FlopCounterMode on the first step only;
            # subsequent steps reuse measured_flops_per_step (fixed shapes => fixed cost).
            flop_ctx = FlopCounterMode(display=False) if measured_flops_per_step is None else contextlib.nullcontext()
            with flop_ctx:
                with autocast:
                    # Crop-major flatten: collate shape is (B, V, 3, H, W) but DINO wants per-crop chunks
                    # so [crop0_img0, crop0_img1, ..., crop1_img0, ...] for clean teacher/student alignment.
                    gf = global_views.transpose(0, 1).flatten(0, 1)
                    lf = local_views.transpose(0, 1).flatten(0, 1)
                    # FINO DANN ramp keyed to nanopath's SAMPLE budget (NOT FLOPs — sample-capped at ~19% of the FLOP
                    # cap, so a flop-keyed ramp stalls gamma at ~0.75*gamma_max). Counted from the backbone-unfreeze
                    # point: gamma=0 through the frozen Phase 1 (banks warm), then ramps to full gamma_max by the cap.
                    ramp = max(0.0, (examples_seen / max_train_samples - freeze_backbone_frac) / max(1e-6, 1.0 - freeze_backbone_frac))
                    meta = ((fino_cfg["gamma_max"] * (2.0 / (1.0 + math.exp(-10.0 * ramp)) - 1.0),
                             batch["meta_disc"].to(device, non_blocking=True),
                             {f: batch["mc_" + f].to(device, non_blocking=True) for f, _ in fino_cont}) if fino_cfg else None)
                    cond = batch["meta_disc"][:, cond_col].repeat(train_cfg["global_views"]).to(device, non_blocking=True) if jepa_cond else None
                    molcap_target = batch["molcap_target"].to(device, non_blocking=True) if molcap_cfg else None
                    molcap_present = batch["molcap_present"].to(device, non_blocking=True) if molcap_cfg else None
                    molcap_scale = linear_ramp(sfrac, float(molcap_cfg["ramp_start"]), float(molcap_cfg["ramp_len"])) if molcap_cfg else 0.0
                    if centroid_runtime is None:
                        dino_loss_value, jepa_loss, kde, meta_loss, molcap, molcap_grad_cosine, molcap_grad_norm_ratio = compute_losses(
                            gf, lf, batch_size, masks, mask_idx, mask_w, teacher_temp, kde_scale,
                            ckpt=activation_checkpointing, meta=meta, cond=cond, molcap_target=molcap_target,
                            molcap_present=molcap_present, molcap_scale=molcap_scale,
                            diagnose=should_log and bool(molcap_cfg.get("diagnose", False)),
                        )
                    else:
                        dino_loss_value, jepa_loss, kde, meta_loss, molcap, molcap_grad_cosine, molcap_grad_norm_ratio, centroid_step = compute_losses(
                            gf, lf, batch_size, masks, mask_idx, mask_w, teacher_temp, kde_scale,
                            ckpt=activation_checkpointing, meta=meta, cond=cond, molcap_target=molcap_target,
                            molcap_present=molcap_present, molcap_scale=molcap_scale,
                            diagnose=should_log and bool(molcap_cfg.get("diagnose", False)),
                            sfrac=sfrac,
                            centroid_batch={"slide_idx": batch["molcap_slide_idx"].to(device, non_blocking=True)},
                        )
                    total_loss = dino_loss_value + jepa_loss + kde + meta_loss
                    if molcap_cfg: total_loss = total_loss + molcap
                    if centroid_runtime is not None: total_loss = total_loss + centroid_step.loss
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if examples_seen / max_train_samples < freeze_backbone_frac:  # Phase 1: backbone frozen (patch_embed + heads + metadata still train)
                    for n, p in student_backbone.named_parameters():
                        if not n.startswith("patch_embed"): p.grad = None
                clipped = [*student_backbone.parameters(), *student_dino_head.parameters(), *student_predictor.parameters()]
                if molcap_head: clipped += list(molcap_head.parameters())
                if centroid_runtime is not None: clipped += list(centroid_runtime.head.parameters())
                grad_norm = nn.utils.clip_grad_norm_(clipped, dino_cfg["clip_grad"])
                opt.step()
                if centroid_runtime is not None and centroid_step.proposal is not None:
                    centroid_runtime.bank.commit(
                        centroid_step.proposal,
                        observed_tiles=batch_size,
                        caption_present_tiles=int(molcap_present.sum().item()),
                    )
            if measured_flops_per_step is None:
                measured_flops_per_step = int(flop_ctx.get_total_flops())
                print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                m = cosine_schedule(0.994, 1.0, reg_frac)
                update_ema(student_backbone, teacher_backbone, m)
                update_ema(student_dino_head, teacher_dino_head, m)
            step_seconds = time.monotonic() - batch_started_at
            examples_seen += batch_size
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            if should_log:
                reduced = {
                    "dino": float(dino_loss_value.detach()),
                    "jepa": float(jepa_loss.detach()),
                    "kde": float(kde.detach()),
                    "meta": float(meta_loss.detach()),
                    "molcap": float(molcap.detach()),
                    "total": float(total_loss.detach()),
                }
                if centroid_runtime is not None:
                    reduced.update({"centroid_loss": float(centroid_step.loss.detach())})
                unique_counts = flush_unique_counts()
                now = time.time()
                elapsed = max(1e-6, now - last_time)
                items_per_sec = (examples_seen - last_examples) / elapsed
                visible_patches_per_sec = (visible_patch_presentations - last_visible_patch_presentations) / elapsed
                flops_per_sec = (train_flops - last_train_flops) / elapsed
                train_loop_wall_seconds = time.monotonic() - train_loop_started_at
                last_time = now
                last_examples = examples_seen
                last_visible_patch_presentations = visible_patch_presentations
                last_train_flops = train_flops
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                flop_steps_remaining = math.ceil(max(0, max_train_flops - train_flops) / max(1, step_train_flops))
                sample_steps_remaining = max(0, max_train_samples - examples_seen) // batch_size
                steps_remaining = min(flop_steps_remaining, sample_steps_remaining)
                total_steps_estimate = completed_step + steps_remaining
                eta_seconds = int(max(0.0, steps_remaining * console_gap_ms / 1000.0 / steps_since_console))
                eta_string = f"{eta_seconds // 3600}:{(eta_seconds % 3600) // 60:02d}:{eta_seconds % 60:02d}"
                current_lr = opt.param_groups[0]["lr"]
                train_log = {
                    "step": completed_step,
                    **reduced,
                    "items_per_sec": items_per_sec,
                    "visible_patches_per_sec": visible_patches_per_sec,
                    "flops_per_sec": flops_per_sec,
                    "wall_seconds": train_loop_wall_seconds,
                    "step_seconds": step_seconds,
                    "data_seconds": data_seconds,
                    "console_gap_ms": console_gap_ms,
                    "eta_seconds": eta_seconds,
                    "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
                    "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
                    "lr": current_lr,
                    "wd": wd,
                    "teacher_temp": teacher_temp,
                    "teacher_momentum": m,
                    "kde_scale": kde_scale,
                    "molcap_scale": molcap_scale,
                    "molcap_coverage": float(molcap_present.mean()) if molcap_present is not None else 0.0,
                    "molcap_grad_cosine": float(molcap_grad_cosine),
                    "molcap_grad_norm_ratio": float(molcap_grad_norm_ratio),
                    "batch_size": batch_size,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                    "grad_norm": float(grad_norm.detach()),
                }
                if centroid_runtime is not None:
                    train_log.update({
                        "centroid_scale": centroid_step.scale,
                        "centroid_gate_passed": bool(centroid_runtime.gate_checked and centroid_runtime.gate_report["passed"]),
                        "centroid_coverage": centroid_step.coverage,
                        "centroid_effective_rank": float(centroid_runtime.gate_report["effective_rank"]) if centroid_runtime.gate_report else 0.0,
                    })
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f}  total: {reduced['total']:.4f}  "
                    f"dino: {reduced['dino']:.4f}  jepa: {reduced['jepa']:.4f}  kde: {reduced['kde']:.4f}  "
                    f"meta: {reduced['meta']:.4f}  molcap: {reduced['molcap']:.4f}  "
                    f"grad_norm: {train_log['grad_norm']:.4f}  flops/s: {flops_per_sec:.3e}  "
                    f"time: {step_seconds:.6f}  data: {data_seconds:.6f}  "
                    f"max mem: {int(gpu_peak_mem_gb * 1024)}",
                    flush=True,
                )
                last_console_step = completed_step
                last_console_monotonic = console_now
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(train_log) + "\n")
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_log.items() if key != "step"},
                    step=completed_step,
                )
                log_probe_results()
                torch.cuda.reset_peak_memory_stats(device)
            if save_checkpoints and completed_step % save_every == 0:
                # Atomic rename keeps the previous good latest.pt intact if a
                # kill lands mid-save.
                save_latest_checkpoint(completed_step)
            # Probe at intermediate sample milestones (probe.count > 1); the final probe
            # always runs after the loop exits, regardless of milestones.
            maybe_run_probe(completed_step)
            if completed_step % int(train_cfg["eval_every"]) == 0 or train_flops >= max_train_flops or examples_seen + batch_size > max_train_samples:
                val = evaluate(completed_step, teacher_temp, kde_scale)
                val_log = {"step": completed_step, **{f"val_{k}": v for k, v in val.items()}}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(val_log) + "\n")
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=completed_step)
                print(f"{console_prefix()} Validation  [{completed_step}]  total: {val['total']:.4f}  dino: {val['dino']:.4f}  jepa: {val['jepa']:.4f}  kde: {val['kde']:.4f}", flush=True)
                # Reset rate clocks after validation so the next train log is train-rate only.
                last_console_step, last_console_monotonic = completed_step, time.monotonic()
                last_time, last_examples, last_visible_patch_presentations, last_train_flops = time.time(), examples_seen, visible_patch_presentations, train_flops
            step = completed_step
            data_wait_started_at = time.monotonic()
            if train_flops >= max_train_flops or examples_seen + batch_size > max_train_samples:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else "max_train_samples"
    final_unique_counts = flush_unique_counts()
    if step > 0:
        # Final probes have their own readers; close pretraining workers before they compete for CPU/IO.
        if train_cfg["num_workers"] > 0:
            if train_loader._iterator is not None:
                train_loader._iterator._shutdown_workers()
                train_loader._iterator = None
        # Probes get their own short-lived checkpoint via run_probe_at; only persist latest.pt
        # at end-of-run when periodic saving is on (save_every set) so smoke runs leave nothing.
        if save_checkpoints and step != last_saved_step:
            save_latest_checkpoint(step)
        run_probe_at(step, examples_seen)
    log_probe_results()
    # Summary is the small, stable artifact downstream scripts and humans compare across runs.
    summary = {
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "wandb": wandb_meta,
        "slurm_job_id": slurm_job_id,
        "backbone_activated_params": backbone_activated_params,
        "batch_size": batch_size,
        "max_train_samples": max_train_samples,
        "max_train_flops": max_train_flops,
        "train_loop_wall_seconds": train_loop_wall_seconds,
        "stop_reason": stop_reason,
        "steps_completed": step,
        "tile_presentations": examples_seen,
        "visible_patch_presentations": visible_patch_presentations,
        **final_unique_counts,
        "train_flops": train_flops,
        "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
        "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
        # Average throughput over the train loop; wall time is diagnostic, not an eligibility cap.
        "flops_per_sec": train_flops / max(1.0, train_loop_wall_seconds),
        "visible_patches_per_sec": visible_patch_presentations / max(1.0, train_loop_wall_seconds),
        "warmup_fraction": dino_cfg["warmup_fraction"],
        "warmup_train_samples": warmup_train_samples,
        "lr": dino_cfg["lr"],
        "adam_beta2": dino_cfg["adam_beta2"],
        "kde_loss_weight": dino_cfg["kde_loss_weight"],
        "kde_concentration": dino_cfg["kde_concentration"],
        "drop_path_rate": dino_cfg["drop_path_rate"],
        "layerwise_decay": dino_cfg["layerwise_decay"],
        "probe_target_samples": probe_targets,
        "probe_target_fractions": [None if max_train_samples == 0 else target / max_train_samples for target in probe_targets],
        **({} if probe_state is None else completed_probe_summary(output_dir)),
    }
    if probe_state is not None and "final_probe_score" not in summary:
        raise ValueError("probe.enabled is true but final_probe_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{console_prefix()} Summary  "
        f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
        f"final_probe_score: {summary.get('final_probe_score')}",
        flush=True,
    )
    for key in summary.keys():
        wandb_run.summary[key] = summary[key]
    wandb_run.finish()
    finish_labless_autosubmit(labless_autosubmit_file, output_dir, repo_dir)


if __name__ == "__main__":
    main()
