"""Unit tests for deterministic evaluation metrics and gating logic.

Text-model metrics (BERTScore, HHEM model path, GPT-4o) are not exercised here
because they require heavy/hosted models; their deterministic fallbacks and the
gating logic are covered instead.

Run with:
    python -m pytest eval/test_metrics.py -v
"""

import pytest

from eval.metrics import (
    compute_drug_entity_error_rate,
    compute_factscore,
    compute_hhem_score,
    compute_rouge_l,
    summary_to_text,
)
from eval.run_eval_suite import check_gates


def _summary(diagnoses=None, meds=None, procedures=None, instructions=""):
    return {
        "diagnoses": diagnoses or [],
        "medications": meds or [],
        "procedures": procedures or [],
        "discharge_instructions": instructions,
        "confidence_flags": [],
    }


def _med(name):
    return {"name": name, "dose": "", "freq": "", "route": ""}


def test_summary_to_text_includes_sections():
    text = summary_to_text(_summary(diagnoses=["Diabetes"], meds=[_med("Aspirin")]))
    assert "Diabetes" in text and "Aspirin" in text


def test_deer_all_valid(drugbank_vocab):
    preds = [_summary(meds=[_med("Aspirin"), _med("Metformin")])]
    assert compute_drug_entity_error_rate(preds, drugbank_vocab) == 0.0


def test_deer_half_invalid(drugbank_vocab):
    preds = [_summary(meds=[_med("Aspirin"), _med("Zzzdrug")])]
    assert compute_drug_entity_error_rate(preds, drugbank_vocab) == 0.5


def test_deer_no_meds_is_zero(drugbank_vocab):
    assert compute_drug_entity_error_rate([_summary()], drugbank_vocab) == 0.0


def test_factscore_perfect_match():
    ref = _summary(diagnoses=["Diabetes"], meds=[_med("Aspirin")])
    pred = _summary(diagnoses=["Diabetes"], meds=[_med("Aspirin")])
    assert compute_factscore([pred], [ref]) == 1.0


def test_factscore_partial():
    ref = _summary(diagnoses=["Diabetes"], meds=[_med("Aspirin")])
    pred = _summary(diagnoses=["Hypertension"], meds=[_med("Aspirin")])
    # 1 of 2 predicted facts (aspirin) supported.
    assert compute_factscore([pred], [ref]) == 0.5


def test_hhem_proxy_identical_is_high():
    s = _summary(diagnoses=["Diabetes"], instructions="Follow up in two weeks")
    score = compute_hhem_score([s], [s], use_model=False)
    assert score == pytest.approx(1.0)


def test_hhem_proxy_unsupported_is_low():
    pred = _summary(instructions="patient prescribed unicorn extract daily")
    ref = _summary(instructions="follow up with cardiology")
    assert compute_hhem_score([pred], [ref], use_model=False) < 0.5


def test_rouge_l_identical_is_one():
    s = _summary(instructions="take aspirin once daily and rest at home")
    assert compute_rouge_l([s], [s]) == pytest.approx(1.0)


def test_check_gates_pass():
    results = {
        "drug_entity_error_rate": 0.0,
        "hhem": 0.9,
        "bertscore": 0.9,
        "rouge_l": 0.5,
        "factscore": 0.8,
        "gpt4o_preference": 0.75,
    }
    gates = check_gates(results)
    assert gates["overall"]["status"] == "pass"
    assert gates["drug_entity_error_rate"]["status"] == "pass"


def test_check_gates_fail_on_deer():
    results = {
        "drug_entity_error_rate": 0.10,  # exceeds 2% max
        "hhem": 0.9, "bertscore": 0.9, "rouge_l": 0.5,
        "factscore": 0.8, "gpt4o_preference": 0.75,
    }
    gates = check_gates(results)
    assert gates["drug_entity_error_rate"]["status"] == "fail"
    assert gates["overall"]["status"] == "fail"


def test_check_gates_skipped_does_not_fail():
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.9, "bertscore": 0.9,
        "rouge_l": 0.5, "factscore": 0.8,
        "gpt4o_preference": None,  # unconfigured judge
    }
    gates = check_gates(results)
    assert gates["gpt4o_preference"]["status"] == "skipped"
    assert gates["overall"]["status"] == "pass"
