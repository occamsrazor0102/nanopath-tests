# FINO metadata

`fino_meta.json` — the **final processed metadata** for FINO, committed so the repo is self-contained. A fresh
clone needs nothing else: `prepare.py` copies it next to the tile dataset, and `dataloader.py` / `train.py` read
it from there. The raw TCGA CSVs and the build code that produced it are intentionally not shipped.

## Format
```
{
  "discrete":   {factor: {barcode: int_id}},      # 22 factors (cancer, subtype, morphology, scanner, grade, stage, …)
  "continuous": {factor: {barcode: float | [floats]}},  # 11 factors (fga, til, necrosis, age, expr512, …) — z-scored
  "n":          {factor: cardinality},            # number of classes per discrete factor
  "cont_dim":   {factor: dim}                      # vector length per continuous factor (1 for scalars, 512 for expr512)
}
```
Patients absent from a factor's map are masked out of that branch at train time (`dataloader.py` emits `-1`/`nan`).

## Which factors are present
Discrete: cancer, tss, msi, year, subtype, grade, site, organ, resection, tstage, nstage, stage, sampletype,
section, stageedition, gender, priortx, scanner, appmag, morphology, diseasetype, classif.
Continuous: necrosis, fga, mutcount, til, age, stromal, mpp, expr, expr512, expr_pca, expr_path.

A config selects which to use as M+/M− via `fino.discrete` / `fino.continuous` (see configs/abl_*.yaml).

## Evaluating *any* metadata selection (generic builder)
The shipped `fino_meta.json` is one curated artifact, but FINO is general: `prepare.py:build_fino_meta` turns **any
column** of the TCGA tables (`tcga_master_dataset.csv` + `tcga_master_cancer_genomics.csv`, joined on the 12-char
barcode) into a factor. Point a config at the CSVs with `fino.csv_dir`; the factor name **is** the column name and
the encoding is chosen by which list it sits in — `discrete` (categorical → dense-id prototype target) or
`continuous` (numeric → z-scored regression target). No per-factor code, no edits to `dataloader.py`/`train.py`
(both are already generic over the factor set). `prepare.py` builds `fino_meta.json` on first run; train + the probe
suite then evaluate it.

```yaml
fino:
  enabled: true
  gamma_max: 1.0
  csv_dir: /path/to/tcga-clinical-data/clinical/tcga_clinical_data   # triggers the generic build
  discrete:   [[project_id, 1], [gender, -0.3]]      # any categorical column; sign>0 = M+, <0 = M−
  continuous: [[cbio_fraction_genome_altered, 1], [age_at_index, 1]]  # any numeric column
```
Unknown column names fail loudly. The curated artifact's special encodings (msi threshold, year bins, subtype
collapse, `expr*` vectors) are not reproduced by the generic path — use raw column names there, or the shipped
artifact (omit `csv_dir`).
