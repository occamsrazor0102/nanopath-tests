import hashlib

import numpy as np
import pandas as pd

from build_molcap_targets import (
    aggregate_patients,
    encode_text,
    render_captions,
    save_target_bank,
    structured_targets,
)


def metadata_csv(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "submitter_id": "TCGA-BB-0002",
                "cancer_type": "COAD",
                "disease_type": "Adenomas and Adenocarcinomas",
                "primary_site": "Colon",
                "primary_diagnosis": "Adenocarcinoma",
                "cbio_subtype": np.nan,
                "ajcc_pathologic_stage": np.nan,
                "tumor_grade": np.nan,
                "cbio_msi_score": 0.9,
                "cbio_fraction_genome_altered": 0.1,
                "cbio_mutation_count": 120,
                "mol_kras_": "Positive",
                "mol_braf_": np.nan,
                "mol_tp53_": np.nan,
                "mol_egfr_": np.nan,
            },
            {
                "submitter_id": "TCGA-AA-0001",
                "cancer_type": "BRCA",
                "disease_type": "Ductal and Lobular Neoplasms",
                "primary_site": "Breast",
                "primary_diagnosis": "Infiltrating duct carcinoma",
                "cbio_subtype": "BRCA_LumA",
                "ajcc_pathologic_stage": "Stage II",
                "tumor_grade": "G2",
                "cbio_msi_score": 0.2,
                "cbio_fraction_genome_altered": 0.6,
                "cbio_mutation_count": 20,
                "mol_kras_": "Negative",
                "mol_braf_": np.nan,
                "mol_tp53_": np.nan,
                "mol_egfr_": np.nan,
            },
            {
                "submitter_id": "TCGA-AA-0001",
                "cancer_type": "BRCA",
                "disease_type": "Ductal and Lobular Neoplasms",
                "primary_site": "Breast",
                "primary_diagnosis": "Infiltrating duct carcinoma",
                "cbio_subtype": "BRCA_LumA",
                "ajcc_pathologic_stage": "Stage II",
                "tumor_grade": "G2",
                "cbio_msi_score": 0.2,
                "cbio_fraction_genome_altered": 0.6,
                "cbio_mutation_count": 20,
                "mol_kras_": "Negative",
                "mol_braf_": np.nan,
                "mol_tp53_": np.nan,
                "mol_egfr_": np.nan,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_real_schema_aggregates_patients_and_omits_missing_values(tmp_path):
    patients = aggregate_patients(metadata_csv(tmp_path))
    captions = render_captions(patients)

    assert patients["submitter_id"].tolist() == ["TCGA-AA-0001", "TCGA-BB-0002"]
    assert "Breast" in captions[0]
    assert "BRCA_LumA" in captions[0]
    assert "KRAS positive" in captions[1]
    assert "nan" not in " ".join(captions).lower()
    assert "KRAS negative" not in captions[0]


def test_conflicting_patient_rows_fail_loudly(tmp_path):
    path = metadata_csv(tmp_path)
    frame = pd.read_csv(path)
    conflict = frame.iloc[[1]].copy()
    conflict["primary_site"] = "Lung"
    pd.concat([frame, conflict], ignore_index=True).to_csv(path, index=False)

    try:
        aggregate_patients(path)
    except AssertionError as exc:
        assert "primary_site" in str(exc)
    else:
        raise AssertionError("conflicting patient metadata was accepted")


def test_structured_and_text_targets_are_deterministic_unit_vectors(tmp_path):
    patients = aggregate_patients(metadata_csv(tmp_path))
    structured_a = structured_targets(patients, dim=384, seed=7777)
    structured_b = structured_targets(patients, dim=384, seed=7777)

    class FakeEncoder:
        def encode(self, captions, normalize_embeddings):
            assert normalize_embeddings
            return np.arange(len(captions) * 6, dtype=np.float32).reshape(len(captions), 6) + 1

    text = encode_text(render_captions(patients), encoder=FakeEncoder())
    np.testing.assert_allclose(structured_a, structured_b)
    np.testing.assert_allclose(np.linalg.norm(structured_a, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-6)


def test_npz_is_non_pickled_and_byte_deterministic(tmp_path):
    patients = aggregate_patients(metadata_csv(tmp_path))
    captions = render_captions(patients)
    targets = structured_targets(patients, dim=8, seed=7777)
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"

    save_target_bank(first, patients["submitter_id"], targets, captions, "structured")
    save_target_bank(second, patients["submitter_id"], targets, captions, "structured")

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    with np.load(first, allow_pickle=False) as bank:
        assert set(bank.files) == {"patient_ids", "targets", "captions", "mode"}
        assert bank["targets"].dtype == np.float32
        assert bank["mode"].item() == "structured"
