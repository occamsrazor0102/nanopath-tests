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
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
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
    raw_cap = os.environ.get("NANOPATH_RUNNER_STOP_AFTER_SAMPLES")
    runner_cap_active = (
        raw_cap is not None
        and int(raw_cap) < int(cfg["train"]["max_train_samples"])
    )
    if runner_cap_active:
        assert not token_path
        return ""
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


def molcap_route_enabled(molcap_cfg):
    return bool(molcap_cfg) and molcap_cfg.get("route") == "probe_cls_hierarchical"


def molcap_head_input_dim(molcap_cfg, embed_dim):
    assert type(embed_dim) is int and embed_dim > 0
    if not molcap_route_enabled(molcap_cfg):
        return embed_dim
    feature_blocks = tuple(molcap_cfg["feature_blocks"])
    assert feature_blocks and all(type(block) is int and block >= 0 for block in feature_blocks)
    assert len(set(feature_blocks)) == len(feature_blocks)
    assert int(molcap_cfg["head_hidden_dim"]) == 512
    input_dim = int(molcap_cfg["input_dim"])
    assert input_dim == len(feature_blocks) * embed_dim
    return input_dim


def training_preflight(cfg, environment=None):
    environment = os.environ if environment is None else environment
    assert int(environment.get("WORLD_SIZE", "1")) == 1
    max_train_samples = int(cfg["train"]["max_train_samples"])
    batch_size = int(cfg["train"]["batch_size"])
    assert batch_size > 0
    if probe_enabled(cfg):
        probe_count = cfg["probe"]["count"]
        assert type(probe_count) is int and probe_count >= 1
    raw_cap = environment.get("NANOPATH_RUNNER_STOP_AFTER_SAMPLES")
    if raw_cap is None:
        return max_train_samples, False
    runner_stop_after_samples = int(raw_cap)
    assert 0 < runner_stop_after_samples <= max_train_samples
    runner_cap_active = runner_stop_after_samples < max_train_samples
    assert not runner_cap_active or runner_stop_after_samples % batch_size == 0
    assert not runner_cap_active or not probe_enabled(cfg)
    assert not runner_cap_active or not environment.get("LABLESS_AUTOSUBMIT_FILE")
    return runner_stop_after_samples, runner_cap_active


def fold_peak_gpu_memory(prior_peak_gb, interval_peak_bytes):
    prior_peak_gb = float(prior_peak_gb)
    interval_peak_bytes = int(interval_peak_bytes)
    assert math.isfinite(prior_peak_gb) and prior_peak_gb >= 0
    assert interval_peak_bytes >= 0
    return max(prior_peak_gb, interval_peak_bytes / float(1024**3))


@contextlib.contextmanager
def isolated_torch_rng(seed, device):
    assert type(seed) is int
    device = torch.device(device)
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        torch.random.default_generator.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class Hierarchy(NamedTuple):
    slide_ids: torch.Tensor
    slide_means: torch.Tensor
    slide_tile_counts: torch.Tensor
    patient_ids: torch.Tensor
    patient_means: torch.Tensor


def crop_major_tile_mean(features, views, batch_size):
    assert isinstance(views, int) and views > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert features.ndim == 2 and features.shape[0] == views * batch_size
    return features.reshape(views, batch_size, -1).float().mean(0)


def deterministic_grouped_sum(values, group_ids, group_count, *, trusted_dense=False):
    assert values.ndim == 2 and values.shape[0] > 0
    assert values.dtype == torch.float32
    assert group_ids.shape == (len(values),)
    assert group_ids.dtype == torch.int64 and group_ids.device == values.device
    assert type(group_count) is int and group_count > 0
    assert type(trusted_dense) is bool
    if not trusted_dense:
        assert torch.isfinite(values).all()
        assert torch.all(group_ids >= 0) and torch.all(group_ids < group_count)
    order = torch.argsort(group_ids, stable=True)
    if trusted_dense and group_count == len(values):
        grouped = values[order]
    else:
        lengths = torch.bincount(group_ids, minlength=group_count)
        assert lengths.shape == (group_count,)
        if not trusted_dense:
            assert torch.all(lengths > 0)
        grouped = torch.segment_reduce(values[order], "sum", lengths=lengths, axis=0)
    assert grouped.shape == (group_count, values.shape[-1])
    assert grouped.dtype == values.dtype and grouped.device == values.device
    return grouped


# Pool tiles within slides before giving every present slide equal patient weight.
def hierarchical_means(features, slide_ids, slide_to_patient):
    assert features.ndim == 2 and features.shape[0] > 0 and torch.isfinite(features).all()
    assert slide_ids.ndim == 1 and slide_ids.shape[0] == features.shape[0]
    assert slide_ids.dtype == torch.int64 and slide_ids.device == features.device
    assert slide_to_patient.ndim == 1 and slide_to_patient.numel() > 0
    assert slide_to_patient.dtype == torch.int64 and slide_to_patient.device == features.device
    assert torch.all(slide_ids >= 0) and torch.all(slide_ids < len(slide_to_patient))
    assert torch.all(slide_to_patient >= 0)
    unique_slides, tile_inverse = torch.unique(slide_ids, sorted=True, return_inverse=True)
    tile_counts = torch.bincount(tile_inverse, minlength=len(unique_slides))
    slide_sums = deterministic_grouped_sum(
        features.float(), tile_inverse, len(unique_slides), trusted_dense=True
    )
    slide_means = slide_sums / tile_counts[:, None]
    slide_patients = slide_to_patient[unique_slides]
    unique_patients, slide_inverse = torch.unique(slide_patients, sorted=True, return_inverse=True)
    patient_sums = deterministic_grouped_sum(
        slide_means, slide_inverse, len(unique_patients), trusted_dense=True
    )
    patient_counts = torch.bincount(slide_inverse, minlength=len(unique_patients))
    return Hierarchy(
        unique_slides,
        slide_means,
        tile_counts,
        unique_patients,
        patient_sums / patient_counts[:, None],
    )


def patient_targets_from_tiles(targets, present, tile_patient_ids, patient_ids):
    assert targets.ndim == 2 and targets.shape[0] > 0
    assert present.ndim == tile_patient_ids.ndim == patient_ids.ndim == 1
    assert len(present) == len(tile_patient_ids) == len(targets)
    assert targets.dtype == present.dtype == torch.float32
    assert tile_patient_ids.dtype == patient_ids.dtype == torch.int64
    assert targets.device == present.device == tile_patient_ids.device == patient_ids.device
    assert torch.isfinite(present).all()
    assert torch.all((present == 0) | (present == 1))
    assert len(patient_ids) > 0
    assert len(patient_ids) == 1 or torch.all(patient_ids[1:] > patient_ids[:-1])
    inverse = torch.searchsorted(patient_ids, tile_patient_ids)
    assert torch.all(inverse < len(patient_ids))
    assert torch.equal(patient_ids[inverse], tile_patient_ids)
    counts = torch.bincount(inverse, minlength=len(patient_ids))
    assert torch.all(counts > 0)
    row_indices = torch.arange(len(targets), dtype=torch.int64, device=targets.device)
    representative_indices = torch.full(
        (len(patient_ids),), len(targets), dtype=torch.int64, device=targets.device
    )
    representative_indices.scatter_reduce_(
        0, inverse, row_indices, reduce="amin", include_self=True
    )
    assert torch.all(representative_indices < len(targets))
    grouped = targets[representative_indices]
    grouped_present = present[representative_indices]
    assert torch.equal(targets, grouped[inverse])
    assert torch.equal(present, grouped_present[inverse])
    return grouped, grouped_present


def teacher_value_student_gradient(student, teacher):
    assert student.shape == teacher.shape
    assert student.dtype == teacher.dtype and student.device == teacher.device
    return teacher.detach() + (student - student.detach())


class RoutedMolCapResult(NamedTuple):
    loss: torch.Tensor
    pending_history: object
    patient_features: torch.Tensor
    patient_targets: torch.Tensor
    patient_present: torch.Tensor
    student_hierarchy: Hierarchy
    teacher_hierarchy: Hierarchy


def paired_routed_molcap(
    molcap_head,
    student_probe_features,
    teacher_probe_features,
    tile_slide_ids,
    tile_patient_ids,
    slide_to_patient,
    tile_targets,
    tile_present,
    *,
    views,
    weight,
    scale,
    centroid_bank=None,
):
    batch_size = int(tile_slide_ids.numel())
    assert tile_slide_ids.shape == tile_patient_ids.shape == (batch_size,)
    assert tile_slide_ids.dtype == tile_patient_ids.dtype == torch.int64
    assert slide_to_patient.dtype == torch.int64
    assert (
        student_probe_features.device
        == teacher_probe_features.device
        == tile_slide_ids.device
        == tile_patient_ids.device
        == slide_to_patient.device
        == tile_targets.device
        == tile_present.device
    )
    assert torch.equal(slide_to_patient[tile_slide_ids], tile_patient_ids)
    student_tiles = crop_major_tile_mean(student_probe_features, views, batch_size)
    teacher_tiles = crop_major_tile_mean(teacher_probe_features, views, batch_size)
    student_hierarchy = hierarchical_means(student_tiles, tile_slide_ids, slide_to_patient)
    teacher_hierarchy = hierarchical_means(teacher_tiles, tile_slide_ids, slide_to_patient)
    assert torch.equal(student_hierarchy.slide_ids, teacher_hierarchy.slide_ids)
    assert torch.equal(student_hierarchy.patient_ids, teacher_hierarchy.patient_ids)
    pending_history = None
    teacher_value = teacher_hierarchy.patient_means
    if centroid_bank is not None:
        pending_history = centroid_bank.propose(teacher_hierarchy)
        assert torch.equal(pending_history.patient_ids, student_hierarchy.patient_ids)
        teacher_value = pending_history.patient_centroids
    patient_features = teacher_value_student_gradient(
        student_hierarchy.patient_means, teacher_value
    )
    patient_targets, patient_present = patient_targets_from_tiles(
        tile_targets,
        tile_present,
        tile_patient_ids,
        student_hierarchy.patient_ids,
    )
    assert torch.all(patient_present == 1)
    loss = float(weight) * float(scale) * molcap_loss(
        molcap_head,
        patient_features,
        patient_targets,
        patient_present,
        views=1,
    )
    return RoutedMolCapResult(
        loss,
        pending_history,
        patient_features,
        patient_targets,
        patient_present,
        student_hierarchy,
        teacher_hierarchy,
    )


def maybe_paired_routed_molcap(
    student_backbone,
    global_crops,
    teacher_probe_features,
    molcap_head,
    molcap_target,
    molcap_present,
    molcap_slide_idx,
    molcap_patient_idx,
    slide_to_patient,
    *,
    feature_blocks,
    seed,
    device,
    views,
    weight,
    scale,
    centroid_bank,
):
    if molcap_target is None:
        assert teacher_probe_features is None
        assert molcap_present is molcap_slide_idx is molcap_patient_idx is None
        return None
    assert teacher_probe_features is not None
    assert molcap_present is not None
    assert molcap_slide_idx is not None and molcap_patient_idx is not None
    assert type(seed) is int
    with isolated_torch_rng(seed, device):
        student_output = student_backbone(
            global_crops, feature_blocks=tuple(feature_blocks)
        )
    return paired_routed_molcap(
        molcap_head,
        student_output["x_norm_probe_features"],
        teacher_probe_features,
        molcap_slide_idx,
        molcap_patient_idx,
        slide_to_patient,
        molcap_target,
        molcap_present,
        views=views,
        weight=weight,
        scale=scale,
        centroid_bank=centroid_bank,
    )


def molcap_step_diagnostics(
    routed_result,
    molcap_head,
    *,
    centroid_bank=None,
    min_slide_updates=2,
    gate_report=None,
):
    assert isinstance(routed_result, RoutedMolCapResult)
    assert type(min_slide_updates) is int and min_slide_updates >= 1
    pending_history = routed_result.pending_history
    drift = (
        pending_history.drift_cosines.detach().float().cpu()
        if pending_history is not None
        else torch.empty(0)
    )
    drift_quantiles = (
        torch.quantile(drift, torch.tensor([0.1, 0.5, 0.9])).tolist()
        if drift.numel()
        else [None, None, None]
    )
    parameter_energy = sum(
        float(parameter.detach().float().square().sum().item())
        for parameter in molcap_head.parameters()
    )
    with torch.no_grad():
        caption_predictions = molcap_head(routed_result.patient_features.detach()).float()
    diagnostics = {
        "molcap_unique_patients": int(routed_result.student_hierarchy.patient_ids.numel()),
        "molcap_current_slides": int(routed_result.student_hierarchy.slide_ids.numel()),
        "molcap_target_coverage": float(routed_result.patient_present.mean().item()),
        "molcap_readout_norm_mean": float(
            routed_result.patient_features.detach().float().norm(dim=-1).mean().item()
        ),
        "molcap_head_parameter_norm": math.sqrt(parameter_energy),
        "molcap_centroid_caption_cosine": float(
            F.cosine_similarity(
                caption_predictions,
                routed_result.patient_targets.detach().float(),
                dim=-1,
            ).mean().item()
        ),
        "molcap_history_enabled": centroid_bank is not None,
        "molcap_history_state_step": (
            int(centroid_bank.centroid_state_step.item()) if centroid_bank is not None else 0
        ),
        "molcap_historical_tile_fraction": (
            float(pending_history.historical_tile_fraction.item())
            if pending_history is not None
            else 0.0
        ),
        "molcap_nonhistorical_tile_fraction": (
            1.0 - float(pending_history.historical_tile_fraction.item())
            if pending_history is not None
            else 1.0
        ),
        "molcap_teacher_drift_mean": float(drift.mean().item()) if drift.numel() else 1.0,
        "molcap_teacher_drift_q10": drift_quantiles[0],
        "molcap_teacher_drift_q50": drift_quantiles[1],
        "molcap_teacher_drift_q90": drift_quantiles[2],
        "molcap_observed_slides": 0,
        "molcap_mature_slides": 0,
        "molcap_observed_patients": 0,
        "molcap_mature_patients": 0,
        "molcap_observed_slides_per_patient_mean": 0.0,
        "molcap_observed_slides_per_patient_q0": 0.0,
        "molcap_observed_slides_per_patient_q50": 0.0,
        "molcap_observed_slides_per_patient_q100": 0.0,
        "molcap_sample_weighted_mature_coverage": 0.0,
        "molcap_update_count_q0": 0.0,
        "molcap_update_count_q25": 0.0,
        "molcap_update_count_q50": 0.0,
        "molcap_update_count_q75": 0.0,
        "molcap_update_count_q100": 0.0,
        "molcap_feature_bank_bytes": 0,
        "molcap_bank_bytes": 0,
        "molcap_bank_state_digest": None,
        "molcap_current_all_observed_geometry": None,
        "molcap_current_mature_geometry": None,
        "molcap_gate_geometry": (
            None
            if gate_report is None
            else {
                "all_observed": gate_report["all_observed"],
                "mature_only": gate_report["mature_only"],
            }
        ),
    }
    if centroid_bank is not None:
        counts = centroid_bank.slide_counts.detach().cpu()
        presentations = centroid_bank.slide_tile_presentations.detach().cpu()
        observed = counts > 0
        mature = counts >= min_slide_updates
        observed_counts = counts[observed].double()
        quantiles = (
            torch.quantile(observed_counts, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)).tolist()
            if observed_counts.numel()
            else [0.0] * 5
        )
        total_presentations = int(presentations.sum().item())
        mature_presentations = int(presentations[mature].sum().item())
        mapping = centroid_bank.slide_to_patient.detach().cpu()
        observed_slides_per_patient = torch.bincount(
            mapping[observed], minlength=len(centroid_bank.patient_slide_counts)
        )
        observed_slides_per_patient = observed_slides_per_patient[
            observed_slides_per_patient > 0
        ].double()
        slide_quantiles = (
            torch.quantile(
                observed_slides_per_patient,
                torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
            ).tolist()
            if observed_slides_per_patient.numel()
            else [0.0, 0.0, 0.0]
        )
        _, observed_centroids = centroid_bank.patient_centroids(1)
        _, mature_centroids = centroid_bank.patient_centroids(min_slide_updates)
        diagnostics.update(
            {
                "molcap_observed_slides": int(observed.sum().item()),
                "molcap_mature_slides": int(mature.sum().item()),
                "molcap_observed_patients": int(torch.unique(mapping[observed]).numel()),
                "molcap_mature_patients": int(torch.unique(mapping[mature]).numel()),
                "molcap_observed_slides_per_patient_mean": (
                    float(observed_slides_per_patient.mean().item())
                    if observed_slides_per_patient.numel()
                    else 0.0
                ),
                "molcap_observed_slides_per_patient_q0": slide_quantiles[0],
                "molcap_observed_slides_per_patient_q50": slide_quantiles[1],
                "molcap_observed_slides_per_patient_q100": slide_quantiles[2],
                "molcap_sample_weighted_mature_coverage": (
                    mature_presentations / total_presentations
                    if total_presentations
                    else 0.0
                ),
                "molcap_update_count_q0": quantiles[0],
                "molcap_update_count_q25": quantiles[1],
                "molcap_update_count_q50": quantiles[2],
                "molcap_update_count_q75": quantiles[3],
                "molcap_update_count_q100": quantiles[4],
                "molcap_feature_bank_bytes": int(
                    centroid_bank.slide_centroids.numel()
                    * centroid_bank.slide_centroids.element_size()
                ),
                "molcap_bank_bytes": int(
                    sum(
                        buffer.numel() * buffer.element_size()
                        for buffer in centroid_bank.buffers()
                    )
                ),
                "molcap_bank_state_digest": centroid_bank_state_digest(centroid_bank),
                "molcap_current_all_observed_geometry": _diagnostic_centroid_geometry(
                    observed_centroids
                ),
                "molcap_current_mature_geometry": _diagnostic_centroid_geometry(
                    mature_centroids
                ),
            }
        )
    for value in diagnostics.values():
        if type(value) is float:
            assert math.isfinite(value)
    return diagnostics


# Measure raw patient-centroid geometry deterministically on CPU in float64.
def centroid_geometry(patient_centroids):
    assert isinstance(patient_centroids, torch.Tensor)
    x = patient_centroids.detach().to(device="cpu", dtype=torch.float64)
    assert x.ndim == 2 and x.shape[0] >= 2 and torch.isfinite(x).all()
    norms = x.norm(dim=1)
    assert torch.all(norms > 0)
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (x.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    assert torch.isfinite(eigenvalues).all() and total > 0
    probabilities = eigenvalues[eigenvalues > 0] / total
    effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())
    participation_ratio = total.square() / eigenvalues.square().sum()
    unit = x / norms[:, None]
    mean_offdiag_cosine = (
        unit.sum(dim=0).square().sum() - x.shape[0]
    ) / (x.shape[0] * (x.shape[0] - 1))
    metrics = {
        "patient_count": int(x.shape[0]),
        "min_norm": float(norms.min().item()),
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation_ratio.item()),
        "mean_offdiag_cosine": float(mean_offdiag_cosine.item()),
    }
    assert all(math.isfinite(value) for value in metrics.values())
    return metrics


def _diagnostic_centroid_geometry(patient_centroids):
    x = patient_centroids.detach().to(device="cpu", dtype=torch.float64)
    assert x.ndim == 2 and torch.isfinite(x).all()
    norms = x.norm(dim=1)
    centered_energy = (x - x.mean(dim=0, keepdim=True)).square().sum() if len(x) else 0
    if len(x) >= 2 and torch.all(norms > 0) and centered_energy > 0:
        return centroid_geometry(x)
    return {
        "patient_count": int(x.shape[0]),
        "min_norm": float(norms.min().item()) if len(norms) else None,
        "effective_rank": None,
        "participation_ratio": None,
        "mean_offdiag_cosine": None,
    }


def _metadata_exactly_matches(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        if len(actual) != len(expected):
            return False
        for actual_key, actual_value in actual.items():
            matching_keys = [
                expected_key
                for expected_key in expected
                if type(actual_key) is type(expected_key) and actual_key == expected_key
            ]
            if len(matching_keys) != 1:
                return False
            if not _metadata_exactly_matches(actual_value, expected[matching_keys[0]]):
                return False
        return True
    if type(actual) in (tuple, list):
        return len(actual) == len(expected) and all(
            _metadata_exactly_matches(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    try:
        return bool(actual == expected)
    except (TypeError, ValueError):
        return False


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
        assert slide_to_patient.ndim == 1 and slide_to_patient.numel() > 0
        assert slide_to_patient.dtype == torch.int64 and torch.all(slide_to_patient >= 0)
        assert isinstance(feature_dim, int) and feature_dim > 0
        assert float(momentum) == 0.9
        mapping = slide_to_patient.detach().clone()
        patient_count = int(mapping.max().item()) + 1
        assert torch.equal(torch.unique(mapping), torch.arange(patient_count, device=mapping.device))
        self.momentum = float(momentum)
        self.register_buffer("slide_to_patient", mapping, persistent=False)
        self.register_buffer(
            "slide_centroids",
            torch.zeros(len(mapping), feature_dim, dtype=torch.float32, device=mapping.device),
        )
        self.register_buffer(
            "slide_counts", torch.zeros(len(mapping), dtype=torch.int64, device=mapping.device)
        )
        self.register_buffer(
            "slide_tile_presentations",
            torch.zeros(len(mapping), dtype=torch.int64, device=mapping.device),
        )
        self.register_buffer(
            "centroid_state_step", torch.zeros((), dtype=torch.int64, device=mapping.device)
        )
        self.register_buffer(
            "patient_sums",
            torch.zeros(patient_count, feature_dim, dtype=torch.float32, device=mapping.device),
            persistent=False,
        )
        self.register_buffer(
            "patient_slide_counts",
            torch.zeros(patient_count, dtype=torch.int64, device=mapping.device),
            persistent=False,
        )

    # Return the exact eligible equal-slide patient population used by geometry.
    # The reduction is deliberately CPU float64 and traverses ascending slide ids.
    def patient_centroids(self, min_slide_updates=1):
        assert type(min_slide_updates) is int and min_slide_updates >= 1
        slide_counts = self.slide_counts.detach().cpu()
        slide_ids = (slide_counts >= min_slide_updates).nonzero().flatten()
        centroids = self.slide_centroids.detach().to(device="cpu", dtype=torch.float64)
        mapping = self.slide_to_patient.detach().cpu()
        patients = mapping[slide_ids]
        patient_ids, inverse = torch.unique(patients, sorted=True, return_inverse=True)
        sums = centroids.new_zeros((len(patient_ids), centroids.shape[-1]))
        if len(slide_ids):
            sums.index_add_(0, inverse, centroids[slide_ids])
        counts = torch.bincount(inverse, minlength=len(patient_ids))
        return patient_ids, sums / counts[:, None] if len(patient_ids) else sums

    def sample_weighted_mature_coverage(self, min_slide_updates=2):
        assert type(min_slide_updates) is int and min_slide_updates >= 1
        presentations = self.slide_tile_presentations.detach().cpu()
        counts = self.slide_counts.detach().cpu()
        total = presentations.sum()
        assert total > 0
        mature = presentations[counts >= min_slide_updates].sum()
        return float((mature.double() / total.double()).item())

    def _assert_canonical_state_dtypes(self):
        assert self.slide_to_patient.dtype == torch.int64
        assert self.slide_centroids.dtype == self.patient_sums.dtype == torch.float32
        assert self.slide_counts.dtype == torch.int64
        assert self.slide_tile_presentations.dtype == torch.int64
        assert self.centroid_state_step.dtype == torch.int64
        assert self.patient_slide_counts.dtype == torch.int64

    def _rebuild_patient_caches_cpu(self, slide_centroids, slide_counts):
        assert slide_centroids.shape == self.slide_centroids.shape
        assert slide_centroids.dtype == torch.float32
        assert slide_counts.shape == self.slide_counts.shape
        assert slide_counts.dtype == torch.int64
        centroids = slide_centroids.detach().to(device="cpu", dtype=torch.float32)
        counts = slide_counts.detach().to(device="cpu", dtype=torch.int64)
        mapping = self.slide_to_patient.detach().to(device="cpu", dtype=torch.int64)
        observed = counts > 0
        observed_patients = mapping[observed]
        patient_sums = torch.zeros(self.patient_sums.shape, dtype=torch.float32)
        if len(observed_patients):
            patient_sums.index_add_(0, observed_patients, centroids[observed])
        patient_slide_counts = torch.bincount(
            observed_patients, minlength=len(self.patient_slide_counts)
        )
        assert patient_slide_counts.shape == self.patient_slide_counts.shape
        return patient_sums, patient_slide_counts

    # Build the exact next state without mutating any committed buffer.
    def propose(self, teacher):
        assert isinstance(teacher, Hierarchy)
        slide_ids = teacher.slide_ids
        assert slide_ids.ndim == 1 and slide_ids.numel() > 0
        assert slide_ids.dtype == torch.int64 and slide_ids.device == self.slide_centroids.device
        assert len(slide_ids) == 1 or torch.all(slide_ids[1:] > slide_ids[:-1])
        assert torch.all(slide_ids >= 0) and torch.all(slide_ids < len(self.slide_to_patient))
        assert teacher.slide_means.shape == (len(slide_ids), self.slide_centroids.shape[-1])
        assert teacher.slide_means.dtype == self.slide_centroids.dtype
        assert teacher.slide_means.device == self.slide_centroids.device
        assert torch.isfinite(teacher.slide_means).all()
        assert teacher.slide_tile_counts.shape == slide_ids.shape
        assert teacher.slide_tile_counts.dtype == torch.int64
        assert teacher.slide_tile_counts.device == self.slide_centroids.device
        assert torch.all(teacher.slide_tile_counts > 0)
        slide_patients = self.slide_to_patient[slide_ids]
        patient_ids, inverse = torch.unique(slide_patients, sorted=True, return_inverse=True)
        assert teacher.patient_ids.shape == patient_ids.shape
        assert teacher.patient_ids.dtype == torch.int64
        assert teacher.patient_ids.device == self.slide_centroids.device
        assert torch.equal(teacher.patient_ids, patient_ids)
        assert teacher.patient_means.shape == (len(patient_ids), self.slide_centroids.shape[-1])
        assert teacher.patient_means.dtype == self.slide_centroids.dtype
        assert teacher.patient_means.device == self.slide_centroids.device
        current_sums = deterministic_grouped_sum(
            teacher.slide_means, inverse, len(patient_ids), trusted_dense=True
        )
        current_counts = torch.bincount(inverse, minlength=len(patient_ids))
        assert torch.allclose(
            teacher.patient_means, current_sums / current_counts[:, None], atol=1e-6, rtol=0
        )
        old = self.slide_centroids[slide_ids]
        seen = self.slide_counts[slide_ids] > 0
        teacher_means = teacher.slide_means.detach()
        next_values = torch.where(
            seen[:, None],
            self.momentum * old + (1.0 - self.momentum) * teacher_means,
            teacher_means,
        )
        deltas = next_values - torch.where(seen[:, None], old, torch.zeros_like(old))
        sums = self.patient_sums[patient_ids] + deterministic_grouped_sum(
            deltas, inverse, len(patient_ids), trusted_dense=True
        )
        counts = self.patient_slide_counts[patient_ids] + torch.bincount(
            inverse[~seen], minlength=len(patient_ids)
        )
        assert torch.all(counts > 0)
        historical_tile_fraction = (
            teacher.slide_tile_counts[seen].sum().float()
            / teacher.slide_tile_counts.sum().float()
        )
        return CentroidProposal(
            int(self.centroid_state_step.item()),
            slide_ids.detach().clone(),
            next_values.detach().clone(),
            teacher.slide_tile_counts.detach().clone(),
            patient_ids.detach().clone(),
            (sums / counts[:, None]).detach().clone(),
            F.cosine_similarity(old[seen], next_values[seen], dim=-1).detach().clone(),
            historical_tile_fraction.detach().clone(),
        )

    @torch.no_grad()
    def commit(self, proposal, step):
        # Validate the whole proposal before the first write, making rejection atomic.
        assert isinstance(proposal, CentroidProposal)
        state_step = int(self.centroid_state_step.item())
        assert type(proposal.base_state_step) is int and proposal.base_state_step == state_step
        assert type(step) is int and step == state_step + 1
        maximum_int64 = torch.iinfo(torch.int64).max
        assert step <= maximum_int64
        slide_ids = proposal.slide_ids
        assert slide_ids.ndim == 1 and slide_ids.numel() > 0
        assert slide_ids.dtype == torch.int64 and slide_ids.device == self.slide_centroids.device
        assert len(slide_ids) == 1 or torch.all(slide_ids[1:] > slide_ids[:-1])
        assert torch.all(slide_ids >= 0) and torch.all(slide_ids < len(self.slide_to_patient))
        assert proposal.next_slide_centroids.shape == (len(slide_ids), self.slide_centroids.shape[-1])
        assert proposal.next_slide_centroids.dtype == self.slide_centroids.dtype
        assert proposal.next_slide_centroids.device == self.slide_centroids.device
        assert not proposal.next_slide_centroids.requires_grad
        assert torch.isfinite(proposal.next_slide_centroids).all()
        assert proposal.slide_tile_counts.shape == slide_ids.shape
        assert proposal.slide_tile_counts.dtype == torch.int64
        assert proposal.slide_tile_counts.device == self.slide_centroids.device
        assert torch.all(proposal.slide_tile_counts > 0)
        patients = self.slide_to_patient[slide_ids]
        patient_ids, inverse = torch.unique(patients, sorted=True, return_inverse=True)
        assert proposal.patient_ids.shape == patient_ids.shape
        assert proposal.patient_ids.dtype == torch.int64
        assert proposal.patient_ids.device == self.slide_centroids.device
        assert torch.equal(proposal.patient_ids, patient_ids)
        assert proposal.patient_centroids.shape == (len(patient_ids), self.slide_centroids.shape[-1])
        assert proposal.patient_centroids.dtype == self.slide_centroids.dtype
        assert proposal.patient_centroids.device == self.slide_centroids.device
        assert not proposal.patient_centroids.requires_grad
        assert torch.isfinite(proposal.patient_centroids).all()
        old = self.slide_centroids[slide_ids]
        seen = self.slide_counts[slide_ids] > 0
        assert torch.all(self.slide_counts[slide_ids] < maximum_int64)
        assert torch.all(
            self.slide_tile_presentations[slide_ids]
            <= maximum_int64 - proposal.slide_tile_counts
        )
        patient_increments = torch.bincount(
            inverse[~seen], minlength=len(patient_ids)
        )
        assert torch.all(
            self.patient_slide_counts[patient_ids] <= maximum_int64 - patient_increments
        )
        expected_sums = self.patient_sums[patient_ids] + deterministic_grouped_sum(
            proposal.next_slide_centroids
            - torch.where(seen[:, None], old, torch.zeros_like(old)),
            inverse,
            len(patient_ids),
            trusted_dense=True,
        )
        expected_counts = self.patient_slide_counts[patient_ids] + patient_increments
        assert torch.all(expected_counts > 0)
        assert torch.allclose(
            proposal.patient_centroids,
            expected_sums / expected_counts[:, None],
            atol=1e-6,
            rtol=0,
        )
        expected_drift = F.cosine_similarity(old[seen], proposal.next_slide_centroids[seen], dim=-1)
        assert proposal.drift_cosines.shape == expected_drift.shape
        assert proposal.drift_cosines.dtype == self.slide_centroids.dtype
        assert proposal.drift_cosines.device == self.slide_centroids.device
        assert not proposal.drift_cosines.requires_grad and torch.isfinite(proposal.drift_cosines).all()
        assert torch.allclose(proposal.drift_cosines, expected_drift, atol=1e-6, rtol=0)
        expected_fraction = (
            proposal.slide_tile_counts[seen].sum().float()
            / proposal.slide_tile_counts.sum().float()
        )
        assert proposal.historical_tile_fraction.shape == expected_fraction.shape
        assert proposal.historical_tile_fraction.dtype == self.slide_centroids.dtype
        assert proposal.historical_tile_fraction.device == self.slide_centroids.device
        assert not proposal.historical_tile_fraction.requires_grad
        assert torch.isfinite(proposal.historical_tile_fraction)
        assert torch.allclose(proposal.historical_tile_fraction, expected_fraction, atol=1e-6, rtol=0)
        old = old.clone()
        self.slide_centroids[slide_ids] = proposal.next_slide_centroids
        self.slide_counts[slide_ids] += 1
        self.slide_tile_presentations[slide_ids] += proposal.slide_tile_counts
        self.patient_sums[patient_ids] = expected_sums
        self.patient_slide_counts[patient_ids] = expected_counts
        self.centroid_state_step.fill_(step)

    @torch.no_grad()
    def export_state(self, metadata):
        assert type(metadata) is dict
        self._assert_canonical_state_dtypes()
        rebuilt_sums, rebuilt_counts = self._rebuild_patient_caches_cpu(
            self.slide_centroids, self.slide_counts
        )
        staged_sums = rebuilt_sums.to(device=self.patient_sums.device)
        staged_counts = rebuilt_counts.to(device=self.patient_slide_counts.device)
        self.patient_sums.copy_(staged_sums)
        self.patient_slide_counts.copy_(staged_counts)
        return {
            "metadata": dict(metadata),
            "slide_centroids": self.slide_centroids.detach().cpu().clone(),
            "slide_counts": self.slide_counts.detach().cpu().clone(),
            "slide_tile_presentations": self.slide_tile_presentations.detach().cpu().clone(),
            "centroid_state_step": self.centroid_state_step.detach().cpu().clone(),
        }

    @torch.no_grad()
    def restore_state(self, payload, expected_metadata, expected_step):
        # Validate and stage every authoritative and derived value before the first write.
        state_names = (
            "slide_centroids",
            "slide_counts",
            "slide_tile_presentations",
            "centroid_state_step",
        )
        self._assert_canonical_state_dtypes()
        assert type(payload) is dict
        assert len(payload) == len(state_names) + 1
        assert all(type(name) is str for name in payload)
        assert set(payload) == {"metadata", *state_names}
        assert type(payload["metadata"]) is dict and type(expected_metadata) is dict
        assert _metadata_exactly_matches(payload["metadata"], expected_metadata)
        assert type(expected_step) is int and expected_step >= 0

        staged = {}
        for name in state_names:
            source = payload[name]
            target = getattr(self, name)
            assert isinstance(source, torch.Tensor) and source.layout == torch.strided
            assert not source.requires_grad
            assert source.shape == target.shape and source.dtype == target.dtype
            staged[name] = source.detach().to(device=target.device).clone()

        centroids = staged["slide_centroids"]
        counts = staged["slide_counts"]
        presentations = staged["slide_tile_presentations"]
        state_step = int(staged["centroid_state_step"].item())
        assert state_step == expected_step
        assert torch.isfinite(centroids).all()
        assert torch.all(counts >= 0) and torch.all(counts <= state_step)
        assert sum(int(value) for value in counts.detach().cpu().tolist()) >= state_step
        assert torch.all(presentations >= 0) and torch.all(presentations >= counts)
        assert torch.equal(counts == 0, presentations == 0)
        observed = counts > 0
        assert state_step == 0 or torch.any(observed)
        if torch.any(~observed):
            assert torch.count_nonzero(centroids[~observed]) == 0

        rebuilt_sums_cpu, rebuilt_counts_cpu = self._rebuild_patient_caches_cpu(
            centroids, counts
        )
        rebuilt_sums = rebuilt_sums_cpu.to(device=self.patient_sums.device)
        rebuilt_counts = rebuilt_counts_cpu.to(device=self.patient_slide_counts.device)
        assert torch.isfinite(rebuilt_sums).all()

        for name in state_names:
            getattr(self, name).copy_(staged[name])
        self.patient_sums.copy_(rebuilt_sums)
        self.patient_slide_counts.copy_(rebuilt_counts)


def centroid_bank_state_digest(bank):
    assert isinstance(bank, HierarchicalCentroidBank)
    hasher = hashlib.sha256()
    formats = {
        "slide_to_patient": "<i8",
        "slide_centroids": "<f4",
        "slide_counts": "<i8",
        "slide_tile_presentations": "<i8",
        "centroid_state_step": "<i8",
    }
    for name, dtype in formats.items():
        tensor = getattr(bank, name)
        array = np.asarray(tensor.detach().cpu().contiguous().numpy(), dtype=dtype)
        hasher.update(name.encode("ascii") + b"\0")
        hasher.update(dtype.encode("ascii") + b"\0")
        hasher.update(len(array.shape).to_bytes(1, byteorder="little", signed=False))
        for dimension in array.shape:
            hasher.update(int(dimension).to_bytes(8, byteorder="little", signed=False))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


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
        assert isinstance(boundary_proposal, CentroidProposal)
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


def centroid_audit(bank, min_slide_updates=2, *, boundary_proposal=None):
    assert isinstance(bank, HierarchicalCentroidBank)
    assert type(min_slide_updates) is int and min_slide_updates >= 1
    observed_patient_ids, observed = bank.patient_centroids(1)
    mature_patient_ids, mature = bank.patient_centroids(min_slide_updates)
    slide_counts = bank.slide_counts.detach().cpu()
    mapping = bank.slide_to_patient.detach().cpu()
    observed_slides = slide_counts > 0
    mature_slides = slide_counts >= min_slide_updates
    observed_slides_per_patient = torch.bincount(
        mapping[observed_slides], minlength=len(bank.patient_slide_counts)
    )
    observed_slides_per_patient = observed_slides_per_patient[
        observed_slides_per_patient > 0
    ]
    assert len(observed_patient_ids) == len(observed_slides_per_patient)
    return {
        "sample_weighted_mature_coverage": bank.sample_weighted_mature_coverage(
            min_slide_updates
        ),
        "all_observed": centroid_geometry(observed),
        "mature_only": _diagnostic_centroid_geometry(mature),
        "population_sizes": {
            "mature_min_slide_updates": min_slide_updates,
            "observed_slides": int(observed_slides.sum().item()),
            "mature_slides": int(mature_slides.sum().item()),
            "observed_patients": int(len(observed_patient_ids)),
            "mature_patients": int(len(mature_patient_ids)),
        },
        "slide_update_count_distribution": {
            "population": "observed_slides",
            **_fixed_distribution(
                slide_counts[observed_slides],
                (
                    ("q0", 0.0),
                    ("q25", 0.25),
                    ("q50", 0.5),
                    ("q75", 0.75),
                    ("q100", 1.0),
                ),
            ),
        },
        "observed_slides_per_patient_distribution": {
            "population": "observed_patients",
            **_fixed_distribution(
                observed_slides_per_patient,
                (
                    ("q0", 0.0),
                    ("q25", 0.25),
                    ("q50", 0.5),
                    ("q75", 0.75),
                    ("q100", 1.0),
                ),
            ),
        },
        "boundary_teacher_centroid_drift": _boundary_teacher_drift(
            bank, boundary_proposal
        ),
    }


def require_centroid_gate(audit, history_cfg):
    hard = audit["all_observed"]
    coverage = audit["sample_weighted_mature_coverage"]
    patient_count = hard["patient_count"]
    effective_rank = hard["effective_rank"]
    participation_ratio = hard["participation_ratio"]
    mean_offdiag_cosine = hard["mean_offdiag_cosine"]
    min_norm = hard["min_norm"]
    assert type(patient_count) is int
    assert all(
        math.isfinite(float(value))
        for value in (
            coverage,
            effective_rank,
            participation_ratio,
            mean_offdiag_cosine,
            min_norm,
        )
    )
    thresholds = {
        "min_sample_weighted_coverage": float(history_cfg["min_sample_weighted_coverage"]),
        "min_effective_rank": float(history_cfg["min_effective_rank"]),
        "min_participation_ratio": float(history_cfg["min_participation_ratio"]),
        "max_mean_offdiag_cosine": float(history_cfg["max_mean_offdiag_cosine"]),
        "min_centroid_norm": float(history_cfg["min_centroid_norm"]),
    }
    assert type(history_cfg["min_geometry_patients"]) is int
    assert all(math.isfinite(value) for value in thresholds.values())
    assert coverage >= thresholds["min_sample_weighted_coverage"]
    assert patient_count >= history_cfg["min_geometry_patients"]
    assert effective_rank >= thresholds["min_effective_rank"]
    assert participation_ratio >= thresholds["min_participation_ratio"]
    assert mean_offdiag_cosine < thresholds["max_mean_offdiag_cosine"]
    assert min_norm > thresholds["min_centroid_norm"]


def build_molcap_history_metadata(molcap_cfg, train_ds):
    return {
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


def sample_order_prefix_digest(sample_ids, limit=8192):
    assert type(limit) is int and limit > 0
    hasher = hashlib.sha256()
    count = 0
    for raw_value in sample_ids:
        if count == limit:
            break
        value = int(raw_value)
        assert -(1 << 63) <= value < (1 << 63)
        hasher.update(value.to_bytes(8, byteorder="little", signed=True))
        count += 1
    return hasher.hexdigest(), count


def restore_sample_order_prefix(checkpoint, *, routed):
    if not routed:
        return [], False
    if checkpoint is None:
        return [], True
    available = checkpoint.get("molcap_sample_order_available")
    prefix = checkpoint.get("molcap_sample_order_prefix")
    if available is None and prefix is None:
        return [], False
    assert type(available) is bool
    assert isinstance(prefix, torch.Tensor)
    assert prefix.ndim == 1 and prefix.dtype == torch.int64 and len(prefix) <= 8192
    values = [int(value) for value in prefix.detach().cpu().tolist()]
    return values, available and len(values) == 8192


def new_molcap_gradient_diagnostics():
    return {
        "count": 0,
        "last_step": None,
        "cosine_last": None,
        "cosine_sum": 0.0,
        "norm_ratio_last": None,
        "norm_ratio_sum": 0.0,
    }


def record_molcap_gradient_diagnostic(diagnostics, *, step, cosine, norm_ratio):
    assert type(step) is int and step >= 1
    cosine = float(cosine)
    norm_ratio = float(norm_ratio)
    assert math.isfinite(cosine) and math.isfinite(norm_ratio)
    last_step = diagnostics["last_step"]
    assert last_step is None or step > last_step
    diagnostics["count"] += 1
    diagnostics["last_step"] = step
    diagnostics["cosine_last"] = cosine
    diagnostics["cosine_sum"] += cosine
    diagnostics["norm_ratio_last"] = norm_ratio
    diagnostics["norm_ratio_sum"] += norm_ratio


def molcap_gradient_diagnostic_summary(diagnostics):
    count = int(diagnostics["count"])
    assert count >= 0
    if count == 0:
        assert diagnostics["last_step"] is None
        cosine_last = cosine_mean = norm_ratio_last = norm_ratio_mean = None
    else:
        assert type(diagnostics["last_step"]) is int
        cosine_last = float(diagnostics["cosine_last"])
        norm_ratio_last = float(diagnostics["norm_ratio_last"])
        cosine_mean = float(diagnostics["cosine_sum"]) / count
        norm_ratio_mean = float(diagnostics["norm_ratio_sum"]) / count
        assert all(
            math.isfinite(value)
            for value in (cosine_last, cosine_mean, norm_ratio_last, norm_ratio_mean)
        )
    return {
        "molcap_grad_diagnostic_count": count,
        "molcap_grad_diagnostic_last_step": diagnostics["last_step"],
        "molcap_grad_cosine_last": cosine_last,
        "molcap_grad_cosine_mean": cosine_mean,
        "molcap_grad_norm_ratio_last": norm_ratio_last,
        "molcap_grad_norm_ratio_mean": norm_ratio_mean,
        # Legacy scalar names continue to mean the latest active observation.
        "molcap_grad_cosine": cosine_last,
        "molcap_grad_norm_ratio": norm_ratio_last,
    }


def build_molcap_summary(
    *,
    routed_result,
    molcap_head,
    centroid_bank,
    molcap_cfg,
    train_ds,
    config_sha256,
    git_commit,
    sample_order_prefix,
    sample_order_available,
    centroid_gate_report,
    centroid_gate_passed,
    molcap_grad_diagnostics,
):
    summary = {}
    if routed_result is not None:
        summary.update(
            molcap_step_diagnostics(
                routed_result,
                molcap_head,
                centroid_bank=centroid_bank,
                min_slide_updates=int(molcap_cfg["history"]["min_slide_updates"]),
                gate_report=centroid_gate_report,
            )
        )
    if sample_order_available:
        sample_order_digest, sample_order_count = sample_order_prefix_digest(
            sample_order_prefix
        )
    else:
        sample_order_digest, sample_order_count = None, 0
    summary.update(
        {
            "molcap_mapping_digest": train_ds.molcap_mapping_digest,
            "molcap_target_sha256": train_ds.molcap_target_sha256,
            "molcap_config_sha256": config_sha256,
            "molcap_source_commit": git_commit,
            "molcap_train_patients": len(train_ds.molcap_patient_ids),
            "molcap_train_slides": len(train_ds.molcap_slide_ids),
            "molcap_sample_order_available": sample_order_available,
            "molcap_sample_order_digest": sample_order_digest,
            "molcap_sample_order_count": sample_order_count,
            "molcap_centroid_gate_passed": centroid_gate_passed,
            **molcap_gradient_diagnostic_summary(molcap_grad_diagnostics),
        }
    )
    return summary


def checkpoint_molcap_state(
    payload,
    *,
    full,
    checkpoint_step,
    molcap_head,
    centroid_bank,
    history_metadata,
    sample_order_prefix=None,
    sample_order_available=None,
):
    payload = dict(payload)
    if not full:
        return payload
    if molcap_head is not None:
        payload["molcap_head"] = {
            name: value.detach().cpu().clone()
            for name, value in molcap_head.state_dict().items()
        }
    if centroid_bank is not None:
        assert type(checkpoint_step) is int
        assert int(centroid_bank.centroid_state_step.item()) == checkpoint_step
        assert type(history_metadata) is dict
        payload["molcap_history"] = centroid_bank.export_state(history_metadata)
    if sample_order_available is not None:
        assert type(sample_order_available) is bool
        prefix = [] if sample_order_prefix is None else list(sample_order_prefix)
        assert len(prefix) <= 8192
        payload["molcap_sample_order_available"] = sample_order_available
        payload["molcap_sample_order_prefix"] = torch.tensor(prefix, dtype=torch.int64)
    return payload


def restore_molcap_history(
    checkpoint,
    *,
    routed,
    centroid_bank,
    history_metadata,
    checkpoint_step,
):
    if not routed:
        return
    if centroid_bank is None:
        assert "molcap_history" not in checkpoint
        return
    assert "molcap_history" in checkpoint
    assert type(history_metadata) is dict
    centroid_bank.restore_state(
        checkpoint["molcap_history"], history_metadata, checkpoint_step
    )
    assert int(centroid_bank.centroid_state_step.item()) == checkpoint_step


def run_centroid_ramp_gate(
    bank, history_cfg, report_path, *, boundary_proposal=None
):
    report_path = Path(report_path)
    audit = None
    try:
        audit = centroid_audit(
            bank,
            int(history_cfg["min_slide_updates"]),
            boundary_proposal=boundary_proposal,
        )
        require_centroid_gate(audit, history_cfg)
        report = {**audit, "passed": True}
        report_path.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        return report
    except Exception as error:
        report = {
            **(audit if audit is not None else {}),
            "passed": False,
            "failure": f"{type(error).__name__}: {error}",
        }
        report_path.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        raise


def transactional_optimizer_step(
    total_loss,
    optimizer,
    clipped_parameters,
    *,
    clip_grad,
    centroid_bank=None,
    pending_history=None,
    completed_step=None,
    post_backward=None,
):
    assert isinstance(total_loss, torch.Tensor) and total_loss.numel() == 1
    assert torch.isfinite(total_loss)
    total_loss.backward()
    if post_backward is not None:
        post_backward()
    optimized = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    assert all(torch.isfinite(parameter.grad).all() for parameter in optimized)
    grad_norm = nn.utils.clip_grad_norm_(clipped_parameters, clip_grad)
    assert torch.isfinite(grad_norm)
    optimizer.step()
    if pending_history is not None:
        assert centroid_bank is not None and type(completed_step) is int
        centroid_bank.commit(pending_history, completed_step)
        assert int(centroid_bank.centroid_state_step.item()) == completed_step
    else:
        assert centroid_bank is None
    return grad_norm


# Orchestrates one pretraining run: setup, train+probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    runner_stop_after_samples, runner_cap_active = training_preflight(cfg)
    repo_dir = Path(__file__).resolve().parent
    labless_autosubmit_file = maybe_arm_labless_autosubmit(cfg, repo_dir)
    train_cfg = cfg["train"]
    dino_cfg = cfg["dino"]
    # FINO metadata-guidance: select factors + signs (float; + encourage M+ / - suppress M-). fino_meta (built or
    # copied beside the dataset by prepare.py) holds per-factor barcode maps + cardinalities (n) / vector dims.
    fino_cfg = cfg["fino"] if (cfg.get("fino") or {}).get("enabled") else None
    molcap_cfg = cfg["molcap"] if (cfg.get("molcap") or {}).get("enabled") else None
    molcap_routed = molcap_route_enabled(molcap_cfg)
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
    molcap_head = (
        seed_neutral_molcap_head(
            molcap_head_input_dim(molcap_cfg, student_backbone.embed_dim),
            int(molcap_cfg["target_dim"]),
            device,
        )
        if molcap_cfg
        else None
    )
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
    checkpoint = None
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
    config_sha256 = hashlib.sha256(Path(cfg["config_path"]).read_bytes()).hexdigest()
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
    val_ds = TCGATileDataset(cfg, is_train=False)
    molcap_slide_to_patient = None
    centroid_bank = None
    history_metadata = None
    if molcap_routed:
        molcap_slide_to_patient = torch.as_tensor(
            train_ds.molcap_slide_to_patient, dtype=torch.int64, device=device
        )
        if bool(molcap_cfg["history"]["enabled"]):
            centroid_bank = HierarchicalCentroidBank(
                molcap_slide_to_patient,
                feature_dim=int(molcap_cfg["input_dim"]),
                momentum=float(molcap_cfg["history"]["momentum"]),
            ).to(device)
            history_metadata = build_molcap_history_metadata(molcap_cfg, train_ds)
    if checkpoint is not None:
        restore_molcap_history(
            checkpoint,
            routed=molcap_routed,
            centroid_bank=centroid_bank,
            history_metadata=history_metadata,
            checkpoint_step=step,
        )
    sample_order_prefix, sample_order_available = restore_sample_order_prefix(
        checkpoint, routed=molcap_routed
    )
    probe_state = prepare_probe_state(cfg, output_dir) if probe_enabled(cfg) else None

    # Train shuffles + drops partials; the loop never starts a batch that would exceed
    # max_train_samples, so every optimizer step keeps the configured batch size.
    loader_kwargs = dict(batch_size=batch_size, drop_last=True, num_workers=train_cfg["num_workers"], pin_memory=True,
                         prefetch_factor=train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
                         persistent_workers=train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    centroid_gate_report = None
    centroid_gate_passed = False
    centroid_gate_path = output_dir / "molcap_centroid_ramp_gate.json"
    last_routed_result = None
    centroid_gate_boundary_proposal = None
    molcap_grad_diagnostics = new_molcap_gradient_diagnostics()
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
        payload.update(
            {
                "dino_head": cpu_state(student_dino_head),
                "dino_head_ema": cpu_state(teacher_dino_head),
                "predictor": cpu_state(student_predictor),
                "opt": opt.state_dict(),
                "examples_seen": examples_seen,
                "visible_patch_presentations": visible_patch_presentations,
                "train_flops": train_flops,
                "wandb": wandb_meta,
                **(
                    {
                        "protos": {k: v.cpu() for k, v in protos.items()},
                        "predictors": {f: cpu_state(m) for f, m in predictors.items()},
                    }
                    if fino_cfg
                    else {}
                ),
            }
        )
        payload = checkpoint_molcap_state(
            payload,
            full=True,
            checkpoint_step=next_step,
            molcap_head=molcap_head,
            centroid_bank=centroid_bank,
            history_metadata=history_metadata,
            sample_order_prefix=sample_order_prefix,
            sample_order_available=sample_order_available if molcap_routed else None,
        )
        return payload

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
                       molcap_target=None, molcap_present=None, molcap_slide_idx=None,
                       molcap_patient_idx=None, molcap_scale=0.0, molcap_completed_step=None,
                       diagnose=False):
        routed_step = molcap_routed and molcap_target is not None
        with torch.no_grad():
            if routed_step:
                t = teacher_backbone(
                    gf, feature_blocks=tuple(molcap_cfg["feature_blocks"])
                )
            else:
                t = teacher_backbone(gf)
            t_cls = teacher_dino_head(t["x_norm_clstoken"]).chunk(train_cfg["global_views"])
            t_prob = sinkhorn(torch.cat((t_cls[1], t_cls[0])), t_temp).view(2, b, -1)
        sg = student_backbone(gf, masks=masks, checkpoint=ckpt)
        sl = student_backbone(lf, checkpoint=ckpt)
        routed_result = None
        if routed_step:
            assert type(molcap_completed_step) is int and molcap_completed_step >= 1
        if molcap_routed:
            routed_result = maybe_paired_routed_molcap(
                student_backbone,
                gf,
                t.get("x_norm_probe_features"),
                molcap_head,
                molcap_target,
                molcap_present,
                molcap_slide_idx,
                molcap_patient_idx,
                molcap_slide_to_patient,
                feature_blocks=tuple(molcap_cfg["feature_blocks"]),
                seed=(
                    int(train_cfg["seed"]) + 1_000_003 * molcap_completed_step
                    if routed_step
                    else None
                ),
                device=device,
                views=train_cfg["global_views"],
                weight=float(molcap_cfg["weight"]),
                scale=molcap_scale,
                centroid_bank=centroid_bank,
            )
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
        grad_diagnostic_active = False
        if molcap_target is not None:
            if routed_step:
                assert routed_result is not None
                molcap = routed_result.loss
            else:
                molcap = float(molcap_cfg["weight"]) * molcap_scale * molcap_loss(
                    molcap_head, sg["x_norm_patchtokens"].mean(1), molcap_target, molcap_present, train_cfg["global_views"]
                )
            if diagnose and molcap_scale > 0 and molcap_present.any():
                base = local_loss + global_loss + jepa_loss + kde + meta_loss
                grad_cosine, grad_norm_ratio = gradient_alignment(base, molcap, student_backbone.blocks[-1].attn.qkv.weight)
                grad_diagnostic_active = True
        return local_loss + global_loss, jepa_loss, kde, meta_loss, molcap, grad_cosine, grad_norm_ratio, grad_diagnostic_active, routed_result

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
                dino_l, jepa_l, kde_v, _, _, _, _, _, routed_result = compute_losses(
                    gf,
                    lf,
                    b,
                    masks,
                    mask_idx,
                    mask_w,
                    eval_teacher_temp,
                    eval_kde_scale,
                )
                assert routed_result is None
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
    peak_gpu_mem_gb = 0.0

    while examples_seen + batch_size <= runner_stop_after_samples and train_flops < max_train_flops:
        for batch in train_loader:
            if examples_seen + batch_size > runner_stop_after_samples or train_flops >= max_train_flops:
                break
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            student_backbone.train()
            student_dino_head.train()
            student_predictor.train()
            if molcap_head: molcap_head.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            if sample_order_available and len(sample_order_prefix) < 8192:
                remaining = 8192 - len(sample_order_prefix)
                sample_order_prefix.extend(
                    int(value) for value in batch["sample_idx"].tolist()[:remaining]
                )
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
            molcap_scale = linear_ramp(sfrac, float(molcap_cfg["ramp_start"]), float(molcap_cfg["ramp_len"])) if molcap_cfg else 0.0
            if centroid_bank is not None and molcap_scale > 0 and not centroid_gate_passed:
                centroid_gate_report = run_centroid_ramp_gate(
                    centroid_bank,
                    molcap_cfg["history"],
                    centroid_gate_path,
                    boundary_proposal=centroid_gate_boundary_proposal,
                )
                centroid_gate_passed = True
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
                    molcap_slide_idx = batch["molcap_slide_idx"].to(device, non_blocking=True) if molcap_routed else None
                    molcap_patient_idx = batch["molcap_patient_idx"].to(device, non_blocking=True) if molcap_routed else None
                    dino_loss_value, jepa_loss, kde, meta_loss, molcap, molcap_grad_cosine, molcap_grad_norm_ratio, molcap_grad_diagnostic_active, routed_result = compute_losses(
                        gf, lf, batch_size, masks, mask_idx, mask_w, teacher_temp, kde_scale,
                        ckpt=activation_checkpointing, meta=meta, cond=cond, molcap_target=molcap_target,
                        molcap_present=molcap_present, molcap_slide_idx=molcap_slide_idx,
                        molcap_patient_idx=molcap_patient_idx, molcap_scale=molcap_scale,
                        molcap_completed_step=completed_step,
                        diagnose=should_log and bool(molcap_cfg.get("diagnose", False)),
                    )
                    total_loss = dino_loss_value + jepa_loss + kde + meta_loss
                    if molcap_cfg: total_loss = total_loss + molcap
                opt.zero_grad(set_to_none=True)
                post_backward = None
                if examples_seen / max_train_samples < freeze_backbone_frac:
                    # Phase 1: backbone frozen (patch_embed + heads + metadata still train).
                    def post_backward():
                        for name, parameter in student_backbone.named_parameters():
                            if not name.startswith("patch_embed"):
                                parameter.grad = None
                clipped = [*student_backbone.parameters(), *student_dino_head.parameters(), *student_predictor.parameters()]
                if molcap_head: clipped += list(molcap_head.parameters())
                pending_history = (
                    routed_result.pending_history if routed_result is not None else None
                )
                grad_norm = transactional_optimizer_step(
                    total_loss,
                    opt,
                    clipped,
                    clip_grad=dino_cfg["clip_grad"],
                    centroid_bank=centroid_bank if pending_history is not None else None,
                    pending_history=pending_history,
                    completed_step=completed_step,
                    post_backward=post_backward,
                )
                if pending_history is not None and molcap_scale == 0.0:
                    assert int(centroid_bank.centroid_state_step.item()) == completed_step
                    centroid_gate_boundary_proposal = pending_history
            if measured_flops_per_step is None:
                measured_flops_per_step = int(flop_ctx.get_total_flops())
                print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                m = cosine_schedule(0.994, 1.0, reg_frac)
                update_ema(student_backbone, teacher_backbone, m)
                update_ema(student_dino_head, teacher_dino_head, m)
            if routed_result is not None:
                last_routed_result = routed_result
            if molcap_grad_diagnostic_active:
                record_molcap_gradient_diagnostic(
                    molcap_grad_diagnostics,
                    step=completed_step,
                    cosine=molcap_grad_cosine,
                    norm_ratio=molcap_grad_norm_ratio,
                )
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
                peak_gpu_mem_gb = fold_peak_gpu_memory(
                    peak_gpu_mem_gb, torch.cuda.max_memory_allocated(device)
                )
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                flop_steps_remaining = math.ceil(max(0, max_train_flops - train_flops) / max(1, step_train_flops))
                sample_steps_remaining = max(0, runner_stop_after_samples - examples_seen) // batch_size
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
                if routed_result is not None:
                    train_log.update(
                        build_molcap_summary(
                            routed_result=routed_result,
                            molcap_head=molcap_head,
                            centroid_bank=centroid_bank,
                            molcap_cfg=molcap_cfg,
                            train_ds=train_ds,
                            config_sha256=config_sha256,
                            git_commit=git_commit,
                            sample_order_prefix=sample_order_prefix,
                            sample_order_available=sample_order_available,
                            centroid_gate_report=centroid_gate_report,
                            centroid_gate_passed=centroid_gate_passed,
                            molcap_grad_diagnostics=molcap_grad_diagnostics,
                        )
                    )
                    train_log["runner_stop_after_samples"] = runner_stop_after_samples
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
            if completed_step % int(train_cfg["eval_every"]) == 0 or train_flops >= max_train_flops or examples_seen + batch_size > runner_stop_after_samples:
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
            if train_flops >= max_train_flops or examples_seen + batch_size > runner_stop_after_samples:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    peak_gpu_mem_gb = fold_peak_gpu_memory(
        peak_gpu_mem_gb, torch.cuda.max_memory_allocated(device)
    )
    if train_flops >= max_train_flops:
        stop_reason = "max_train_flops"
    elif runner_cap_active:
        stop_reason = "runner_stop_after_samples"
    else:
        stop_reason = "max_train_samples"
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
    molcap_summary = {}
    if molcap_routed:
        molcap_summary = build_molcap_summary(
            routed_result=last_routed_result,
            molcap_head=molcap_head,
            centroid_bank=centroid_bank,
            molcap_cfg=molcap_cfg,
            train_ds=train_ds,
            config_sha256=config_sha256,
            git_commit=git_commit,
            sample_order_prefix=sample_order_prefix,
            sample_order_available=sample_order_available,
            centroid_gate_report=centroid_gate_report,
            centroid_gate_passed=centroid_gate_passed,
            molcap_grad_diagnostics=molcap_grad_diagnostics,
        )
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
        "runner_stop_after_samples": runner_stop_after_samples,
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
        "gpu_peak_mem_gb": peak_gpu_mem_gb,
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
        **molcap_summary,
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
