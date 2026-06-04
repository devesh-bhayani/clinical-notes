"""Unit tests for the data layer: pipeline extraction, rejection generation,
and bias audit.

Run with:
    python -m pytest data/test_data.py -v
"""

import json

import pandas as pd
import pytest

from data import bias_audit, pipeline, rejection_gen


def _chosen():
    return {
        "diagnoses": ["Type 2 diabetes mellitus", "Hypertension"],
        "medications": [
            {"name": "Aspirin", "dose": "81 mg", "freq": "once daily", "route": "oral"},
            {"name": "Metformin", "dose": "500 mg", "freq": "twice daily", "route": "oral"},
        ],
        "procedures": ["Colonoscopy", "Echocardiogram"],
        "discharge_instructions": "Do not drive for 24 hours. Follow up in two weeks.",
        "confidence_flags": [],
    }


# --- Rejection generation ---------------------------------------------------

def test_rejection_classes_change_output():
    chosen = _chosen()
    for cls in rejection_gen.FAILURE_CLASSES:
        rejected = rejection_gen.generate_rejected(chosen, cls)
        assert rejected != chosen, f"class {cls} produced identical output"


def test_rejection_is_deterministic():
    chosen = _chosen()
    a = rejection_gen.generate_rejected(chosen, "A")
    b = rejection_gen.generate_rejected(chosen, "A")
    assert a == b


def test_rejection_does_not_mutate_input():
    chosen = _chosen()
    snapshot = json.dumps(chosen, sort_keys=True)
    rejection_gen.generate_rejected(chosen, "C")
    assert json.dumps(chosen, sort_keys=True) == snapshot


def test_medication_hallucination_adds_meds():
    rej = rejection_gen.generate_medication_hallucination(_chosen())
    assert len(rej["medications"]) > len(_chosen()["medications"])


def test_diagnosis_omission_drops_diagnosis():
    rej = rejection_gen.generate_diagnosis_omission(_chosen())
    assert len(rej["diagnoses"]) < len(_chosen()["diagnoses"])


def test_contraindication_reversal_changes_advice():
    rej = rejection_gen.generate_contraindication_reversal(_chosen())
    assert rej["discharge_instructions"] != _chosen()["discharge_instructions"]


def test_build_preference_pairs(tmp_path):
    src = tmp_path / "chosen.jsonl"
    src.write_text(
        json.dumps({"note": "Patient note.", "chosen": _chosen()}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "pairs.jsonl"
    count = rejection_gen.build_preference_pairs(str(src), str(out), batch_size=10)
    assert count == 1
    pair = json.loads(out.read_text(encoding="utf-8").strip())
    assert set(pair) >= {"prompt", "chosen", "rejected", "failure_class"}
    assert pair["chosen"] != pair["rejected"]


# --- Pipeline extraction ----------------------------------------------------

_NOTE = """\
Discharge Diagnosis:
Type 2 diabetes mellitus
Acute kidney injury

Discharge Medications:
1. Aspirin 81 mg PO daily
2. Metformin 500 mg PO twice daily
3. Cardizopine 10 mg PO daily

Major Procedures:
Colonoscopy

Discharge Instructions:
Do not lift heavy objects. Follow up with your doctor in one week.
"""


def test_extract_sections():
    sections = pipeline.extract_sections(_NOTE)
    assert "Type 2 diabetes mellitus" in sections["diagnoses"]
    assert "Colonoscopy" in sections["procedures"]
    assert "Aspirin" in sections["medications_raw"]
    assert "follow up" in sections["discharge_instructions"].lower()


def test_parse_medications():
    sections = pipeline.extract_sections(_NOTE)
    meds = pipeline.extract_medications_spacy(sections["medications_raw"])
    names = {m["name"].lower() for m in meds}
    assert "aspirin" in names and "metformin" in names


def test_build_chosen_summary_drops_unrecognized(drugbank_vocab):
    summary = pipeline.build_chosen_summary(_NOTE, drugbank_vocab)
    names = {m["name"].lower() for m in summary["medications"]}
    assert "aspirin" in names
    # Cardizopine is fabricated and not in the sample vocab → dropped + flagged.
    assert "cardizopine" not in names
    assert any("cardizopine" in f.lower() for f in summary["confidence_flags"])


# --- Bias audit -------------------------------------------------------------

def _bias_df():
    return pd.DataFrame(
        {
            "age": [25, 70, 80, 30, 65, 19, 90, 45],
            "ethnicity": ["White", "White", "White", "Black",
                          "White", "Asian", "White", "White"],
            "admission_type": ["EMERGENCY", "ELECTIVE", "EMERGENCY", "EMERGENCY",
                               "ELECTIVE", "EMERGENCY", "EMERGENCY", "ELECTIVE"],
        }
    )


def test_age_distribution_sums_to_one():
    dist = bias_audit.compute_age_distribution(_bias_df())
    assert pytest.approx(sum(v["pct"] for v in dist.values()), rel=1e-6) == 1.0


def test_detect_imbalance_flags_minorities():
    dist = bias_audit.compute_ethnicity_distribution(_bias_df())
    flagged = bias_audit.detect_imbalance(dist, threshold=0.15)
    assert "Asian" in flagged and "Black" in flagged
    assert "White" not in flagged


def test_oversampling_increases_minority_share():
    df = _bias_df()
    balanced = bias_audit.apply_oversampling(df, "ethnicity", target_ratio=0.25)
    new_dist = bias_audit.compute_ethnicity_distribution(balanced)
    assert new_dist["Asian"]["pct"] >= 0.20


def test_run_full_audit_structure():
    report = bias_audit.run_full_audit(_bias_df())
    assert set(report["distributions"]) == {"age", "ethnicity", "admission_type"}
    assert report["n_records"] == 8
