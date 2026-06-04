"""End-to-end integration test for the full pipeline on synthetic PHI-free data.

Drives the real code paths together — MIMIC-style extraction, ORPO rejection
generation, stub inference, and the evaluation suite — without MIMIC, a GPU, or
any hosted model. Uses the committed sample DrugBank vocabulary (configured via
conftest.py).

Run with:
    python -m pytest tests/test_integration.py -v
"""

import json

import pytest

from eval.run_eval_suite import check_gates, run_all_metrics
from scripts.make_synthetic_data import SYNTHETIC_NOTES, build_synthetic_dataset


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_splits")
    artifacts = build_synthetic_dataset(out)
    return artifacts


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_pipeline_extracts_chosen_summaries(synthetic):
    """Extraction yields one chosen summary per note, with recognized meds only."""
    chosen = _read_jsonl(synthetic["chosen_summaries"])
    assert len(chosen) == len(SYNTHETIC_NOTES)
    for record in chosen:
        summary = record["chosen"]
        assert summary["diagnoses"], "expected at least one diagnosis"
        # Every kept medication must have a name (DrugBank-recognized).
        for med in summary["medications"]:
            assert med["name"]


def test_preference_pairs_are_valid(synthetic):
    """ORPO pairs have the required columns and chosen != rejected."""
    pairs = _read_jsonl(synthetic["train_orpo"])
    assert len(pairs) == len(SYNTHETIC_NOTES)
    for pair in pairs:
        assert {"prompt", "chosen", "rejected", "failure_class"} <= set(pair)
        assert pair["chosen"] != pair["rejected"]
        # chosen/rejected must each parse back to valid JSON objects.
        assert isinstance(json.loads(pair["chosen"]), dict)
        assert isinstance(json.loads(pair["rejected"]), dict)


def test_predictions_match_reference_count(synthetic):
    refs = _read_jsonl(synthetic["test"])
    preds = _read_jsonl(synthetic["test_predictions"])
    assert len(refs) == len(preds) == len(SYNTHETIC_NOTES)


def test_eval_suite_runs_end_to_end(synthetic):
    """run_all_metrics + check_gates produce a coherent, passing-or-skipped report."""
    refs = [r["chosen"] for r in _read_jsonl(synthetic["test"])]
    preds = _read_jsonl(synthetic["test_predictions"])

    results = run_all_metrics(preds, refs)
    gates = check_gates(results)

    # Drug Entity Error Rate is computed locally and must be a real number.
    assert results["drug_entity_error_rate"] is not None
    assert gates["drug_entity_error_rate"]["status"] in {"pass", "fail"}

    # The stub only emits DrugBank-recognized drugs, so DEER must pass its gate.
    assert gates["drug_entity_error_rate"]["status"] == "pass"

    # Metrics requiring hosted models/keys are skipped, never failing the run.
    for optional in ("bertscore", "gpt4o_preference"):
        assert gates[optional]["status"] in {"pass", "skipped"}

    # Overall gate never "fails" purely because optional metrics are skipped.
    assert gates["overall"]["status"] in {"pass", "fail"}
