"""Unit tests for deterministic evaluation metrics and gating logic.

Text-model metrics (BERTScore, HHEM model path, GPT-4o) are not exercised here
because they require heavy/hosted models; their deterministic fallbacks and the
gating logic are covered instead.

Run with:
    python -m pytest eval/test_metrics.py -v
"""

import pytest

from eval import metrics
from eval.metrics import (
    compute_drug_entity_error_rate,
    compute_factscore,
    compute_hhem_score,
    compute_rouge_l,
    summary_to_text,
)
from eval.run_eval_suite import (
    PREDICTIONS_MODEL,
    PREDICTIONS_STUB,
    check_gates,
    detect_prediction_provenance,
)


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


def test_check_gates_hhem_proxy_is_degraded_not_pass():
    """A proxy-derived HHEM score must never satisfy the gate.

    0.9 clears the 0.80 target, but lexical overlap does not measure
    hallucination - the run is unverified, not green.
    """
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.9, "bertscore": 0.9,
        "rouge_l": 0.5, "factscore": 0.8, "gpt4o_preference": 0.8,
        "_methods": {"hhem": metrics.HHEM_PROXY},
    }
    gates = check_gates(results)
    assert gates["hhem"]["status"] == "degraded"
    assert gates["overall"]["status"] == "degraded"


def test_check_gates_hhem_real_model_passes():
    """The same score from the real model is a genuine pass."""
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.9, "bertscore": 0.9,
        "rouge_l": 0.5, "factscore": 0.8, "gpt4o_preference": 0.8,
        "_methods": {"hhem": metrics.HHEM_MODEL},
    }
    gates = check_gates(results)
    assert gates["hhem"]["status"] == "pass"
    assert gates["overall"]["status"] == "pass"


def test_hhem_detailed_reports_proxy_when_model_disabled():
    """use_model=False must self-identify as the proxy, not stay silent."""
    s = {"diagnoses": ["pneumonia"], "medications": [], "procedures": [],
         "discharge_instructions": "rest", "confidence_flags": []}
    score, method = metrics.compute_hhem_detailed([s], [s], use_model=False)
    assert method == metrics.HHEM_PROXY
    assert 0.0 <= score <= 1.0


# --- prediction provenance ------------------------------------------------
# A metric is only as meaningful as the output it scored. These pin the rule
# that stub predictions invalidate a run instead of failing it.


def _stub_summary():
    """A stub-flagged prediction, as api.inference.StubSummarizer emits."""
    from api.inference import STUB_INFERENCE_FLAG

    s = _summary(diagnoses=["pneumonia"], instructions="rest")
    s["confidence_flags"] = [STUB_INFERENCE_FLAG]
    return s


def test_stub_flag_constant_is_shared_with_inference():
    """Eval must key off the same literal the stub actually writes."""
    from api.inference import STUB_INFERENCE_FLAG, StubSummarizer

    out = StubSummarizer().summarize("Diagnosis: pneumonia\nIbuprofen 400mg PO")
    assert STUB_INFERENCE_FLAG in out["confidence_flags"]


def test_detect_provenance_flags_stub_predictions():
    provenance, count = detect_prediction_provenance([_stub_summary()] * 3)
    assert provenance == PREDICTIONS_STUB
    assert count == 3


def test_detect_provenance_model_when_unflagged():
    provenance, count = detect_prediction_provenance([_summary(), _summary()])
    assert provenance == PREDICTIONS_MODEL
    assert count == 0


def test_detect_provenance_one_stub_record_taints_the_run():
    """Mixed output is not partially trustworthy — the file is not a model's."""
    provenance, count = detect_prediction_provenance(
        [_summary(), _stub_summary(), _summary()]
    )
    assert provenance == PREDICTIONS_STUB
    assert count == 1


def test_detect_provenance_tolerates_missing_flags_key():
    provenance, _ = detect_prediction_provenance([{"medications": []}])
    assert provenance == PREDICTIONS_MODEL


def test_stub_provenance_invalidates_every_gate():
    """Numbers that clear their thresholds still cannot read as pass."""
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.9, "bertscore": 0.9,
        "rouge_l": 0.5, "factscore": 0.8, "gpt4o_preference": 0.8,
        "_methods": {"hhem": metrics.HHEM_MODEL},
    }
    gates = check_gates(results, provenance=PREDICTIONS_STUB)
    assert gates["overall"]["status"] == "invalid"
    for name in results:
        if name.startswith("_"):
            continue
        assert gates[name]["status"] == "invalid", name
        assert "stub" in gates[name]["note"].lower()


def test_stub_provenance_outranks_fail():
    """The real regression: a stub run must not read as a weak model."""
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.23, "bertscore": 0.92,
        "rouge_l": 0.39, "factscore": 0.70, "gpt4o_preference": None,
        "_methods": {"hhem": metrics.HHEM_MODEL},
    }
    assert check_gates(results)["overall"]["status"] == "fail"
    gates = check_gates(results, provenance=PREDICTIONS_STUB)
    assert gates["overall"]["status"] == "invalid"
    assert gates["overall"]["provenance"] == PREDICTIONS_STUB


def test_stub_provenance_leaves_skipped_metrics_skipped():
    """A metric that never ran did not score the stub; don't claim it did."""
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.9, "bertscore": 0.9,
        "rouge_l": 0.5, "factscore": 0.8, "gpt4o_preference": None,
        "_methods": {"hhem": metrics.HHEM_MODEL},
    }
    gates = check_gates(results, provenance=PREDICTIONS_STUB)
    assert gates["gpt4o_preference"]["status"] == "skipped"
    assert "note" not in gates["gpt4o_preference"]
    assert gates["drug_entity_error_rate"]["status"] == "invalid"
    assert gates["overall"]["status"] == "invalid"


def test_model_provenance_leaves_gates_untouched():
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.9, "bertscore": 0.9,
        "rouge_l": 0.5, "factscore": 0.8, "gpt4o_preference": 0.8,
        "_methods": {"hhem": metrics.HHEM_MODEL},
    }
    gates = check_gates(results, provenance=PREDICTIONS_MODEL)
    assert gates["overall"]["status"] == "pass"
    assert gates["drug_entity_error_rate"]["status"] == "pass"


def test_provenance_defaults_to_not_invalidating():
    """Omitting provenance must preserve the pre-existing contract."""
    results = {
        "drug_entity_error_rate": 0.0, "hhem": 0.9, "bertscore": 0.9,
        "rouge_l": 0.5, "factscore": 0.8, "gpt4o_preference": 0.8,
        "_methods": {"hhem": metrics.HHEM_MODEL},
    }
    assert check_gates(results)["overall"]["status"] == "pass"
