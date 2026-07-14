# MolCap Matched-Control EMA Centroid Gate Design

Status: preregistered before any latest-observation shadow metrics exist.

## Purpose

The original seed-7777 EMA-centroid arm stopped correctly at its frozen
absolute geometry gate. It had not applied a nonzero MolCap loss. The failure
therefore establishes that the original gate construct was incompatible with
the raw 1,536-dimensional probe readout; it does not test whether historical
EMA patient centroids improve caption supervision.

This is a new postmortem-informed experiment. It preserves the failed run and
its thresholds unchanged, uses the completed route control already submitted
as Labless `run_sub_91ae661e33`, and replaces the absolute isotropy demand with
a matched latest-observation control. No metric from that control has been
observed at preregistration time.

## Frozen references

- Implementation reference: `213a74796e68641a852756e3afd76803ab11a367`.
- Canonical MiniLM target SHA-256:
  `2f6648a4155b96757a136335a253e3faeb6029a92a7e6356380ce80805011577`.
- Mapping digest:
  `8cf4e2e46ba593231ae68ea390e05365b75ce408ff80471a79007b77422d4922`.
- Locked probe/config reference:
  `01c1cdf8017a0481636a28ab58a0ddc67d6e0a06`.
- Route score: `0.6637154886140434`, seed `7777`, B200.
- Route sample-order digest:
  `8b75561dc6862ca9dd0655355b0f983982bdb3d1acf401a68694241719d01bef`.
- Failed absolute gate: effective rank `29.047819056871564`, participation
  ratio `13.545192115431405`, raw mean off-diagonal cosine
  `0.9783163985279353`; coverage and population integrity passed.

The original failed arm remains an archived preregistration. Its config,
report, and thresholds are not edited or reinterpreted as a pass.

## Hypothesis and intervention

The hypothesis remains exactly the historical-forward intervention:

```text
route:    z_R = s_p + stop_gradient(t_p - s_p)
centroid: z_C = s_p + stop_gradient(e_p - s_p)
```

`s_p` is the current student hierarchy, `t_p` is the current detached teacher
hierarchy, and `e_p` is the existing momentum-0.9 equal-slide patient
centroid. The MolCap head, caption target, target weight `0.03`, ramp `0.50`
through `0.75`, readout blocks `[4, 6, 8, 11]`, FINO/DINO/JEPA/KDE recipe,
batch, crops, seed, and one-million-sample budget remain unchanged.

The new machinery is audit-only. It must not change `e_p`, `z_C`, any loss,
gradient, optimizer group, schedule, stochastic forward, or probe.

## Matched latest-observation shadow

During the zero-caption-loss warmup, maintain a second slide bank beside the
EMA bank:

```text
EMA:    e_s <- 0.9 e_s + 0.1 t_s  (first observation copies)
latest: l_s <- t_s                 (every observation copies)
```

Both banks:

- receive the same detached teacher slide means on the same successful
  optimizer steps;
- use the same dense identities, tile-to-slide pooling, equal-slide patient
  pooling, counts, and tile-presentation counts;
- require no gradient and consume no RNG;
- commit only after the optimizer step succeeds;
- have exact matched observed and mature populations at audit time.

The latest bank never enters the MolCap forward or loss. It is serialized in
pre-boundary full checkpoints so calibration can audit it, and is discarded
immediately after the production boundary gate passes. The scored run starts
from scratch; an infrastructure interruption restarts it from scratch.

## Relative geometry audit

For matched all-observed patient matrices `E` (EMA) and `L` (latest), in
canonical patient order, compute in CPU float64:

```text
E0 = E - mean_rows(E)
L0 = L - mean_rows(L)
T_E = ||E0||_F^2 / (n - 1)
T_L = ||L0||_F^2 / (n - 1)
trace_ratio = T_E / T_L
erank_ratio = erank(E0) / erank(L0)
pr_ratio = participation(E0) / participation(L0)
alignment = <E0, L0>_F / (||E0||_F ||L0||_F)
```

Effective rank and participation ratio use the nonnegative eigenvalues of
sample covariance exactly as in the archived gate. Both complete descending
spectra are written to the boundary report.

Generate exactly 256 patient-row permutations of `L0`. Seed a CPU
`torch.Generator` from the first eight digest bytes, interpreted as one
unsigned big-endian integer, of:

```text
SHA256(bytes.fromhex(target_sha256)
       || bytes.fromhex(mapping_digest)
       || b"molcap-matched-latest-v1")
```

For every permutation, compute the same coordinate-matched centered
alignment. The one-sided permutation value is:

```text
p = (1 + count(permuted_alignment >= observed_alignment)) / 257
```

The report also includes, diagnostically only:

- raw mean off-diagonal cosine for both banks and EMA minus latest;
- covariance trace and the full spectra;
- centered linear CKA;
- norms, population sizes, update distributions, and teacher drift;
- the 256 permutation alignments and their deterministic seed.

## Frozen hard gates

Immediately before the first nonzero MolCap scale, all of the following must
hold:

1. Existing target SHA, mapping, finite-state, one-GPU, and identity checks.
2. Sample-weighted mature coverage at least `0.95` with
   `min_slide_updates = 2`.
3. At least `512` exactly matched patients.
4. Exact equality of slide mapping, slide counts, tile-presentation counts,
   state step, observed slides, mature slides, patient ids, and matrix shapes
   between EMA and latest banks.
5. Every EMA and latest patient centroid norm is greater than `1e-6`.
6. `T_E > 0`, `T_L > 0`, and every reported scalar is finite.
7. `trace_ratio >= 1/19`, stored as
   `0.05263157894736842`. This is the stationary variance ratio
   `(1 - momentum) / (1 + momentum)` for momentum `0.9`.
8. `erank_ratio >= 0.5`.
9. `pr_ratio >= 0.5`.
10. `alignment > 0.0`.
11. permutation `p <= 0.01`.

Raw cosine and absolute effective-rank/participation thresholds are not hard
gates in this new experiment. The old absolute values remain visible as
historical diagnostics; they are not silently applied or changed.

The JSON report names every failed condition. Any failure stops before a
nonzero caption loss and is not submitted as a completed run.

## Replay certificate before the full launch

Before launching Python training, copy the complete `111.605 GiB` parquet
training dataset from persistent storage to container-local ephemeral storage.
Validate source/destination file counts, total bytes, and a deterministic
manifest before exposing the destination at the unchanged configured path
`/data/nanopath_parquet`. Keep targets, probe datasets, checkpoints, logs, and
outputs linked to persistent storage. This copy is intensive preprocessing and
is outside training time under the published rules; `train.py` starts only
after staging succeeds. It may change I/O throughput but not config leaves,
data bytes, sample order, or model behavior.

Run the new source for 32,768 samples on B200 and exact H100 with the locked
one-million-sample schedule denominator and probes disabled. Before reusing
the completed route control, certify against the archived `213a747`
calibrations:

- exact first-8,192 sample digest;
- exact values for every persisted core student, teacher, DINO-head,
  predictor, FINO-module, MolCap-head, optimizer, and counter tensor after
  excluding only source/config/output metadata and the new shadow payload;
- exact EMA history payload digest for the centroid calibration;
- zero RNG-state change and zero gradient/loss change when the shadow is
  enabled on a synthetic and real B200 step;
- exact replay under the staged dataset layout, thereby certifying that local
  staging changes only data latency;
- paired peak-memory delta no greater than `0.5 GiB`;
- a control-relative gate preview from the calibration checkpoint. If 32,768
  samples do not reach `0.95` maturity, run one preregistered 40,960-sample
  preview; do not change the gate.

Archived checkpoint references:

| Arm/hardware | Checkpoint SHA-256 | EMA history SHA-256 |
|---|---|---|
| route B200 | `9b8791ff5679b4ce3a40d1cddd20530965ac73acadfa31518d24d24e1bce9aac` | N/A |
| centroid B200 | `c0d1919f8cdc3acd7ddae924c7b3454db517de17573d2b60b5717d50e05325a8` | `c4590c2b2fb75eca518c631e6ea55f3e1461ec180deab03a46ebbfae314dc2a0` |
| route H100 | `e0f201f7d00e4028bc61e3a4b8a37d254b13094f2fbd3d3ed81485884c2e13b0` | N/A |
| centroid H100 | `f63887f8553371f9ee0a6e0bb564d191db986c1c3c3fc6a2b1f81db9cde90f94` | `d8e510a5f17f3d359188aecfca4e934a9f064e679183d6932a4cea85e0e1a533` |

If replay differs, do not compare a new seed-7777 centroid result to the old
route as a paired causal test. Instead preregister and run both arms at unseen
seed `7778` under one source commit.

## Execution and decision rule

If replay and the calibration preview pass, run one fresh B200 seed-7777
centroid arm from sample zero. The production boundary gate runs again on the
full prefix. If it passes, continue through one million samples and all locked
probes. Calculate the H100-equivalent training time from the new B200/H100
calibration ratio; a projection above two hours is reported but does not
suppress the public submission.

The centroid mechanism is supported versus the submitted route control only
if all are true:

- progression, mutation, and survival each improve;
- their unweighted mean improves by at least `0.003`;
- linear, kNN, and few-shot each decline by less than `0.003`.

Overall promotion remains `>= 0.6719107210` with the existing linear/kNN
guards. A completed full run is submitted to Labless regardless of outcome.

No momentum, target, weight, ramp, pooling, head capacity, transform,
relative threshold, or permutation sweep is permitted after observing this
arm. A relative-gate failure means EMA lost too much structure relative to its
matched latest-teacher control. A passed gate followed by endpoint failure is
a negative test of this EMA-centroid mechanism.
