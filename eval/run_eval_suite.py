"""Main evaluation entry point for Clinical Note Summarizer.

Runs all gating metrics against a specified data split and outputs a pass/fail
report to eval/results/latest.json.

Usage:
    python eval/run_eval_suite.py --split test
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running as a script (`python eval/run_eval_suite.py`) by ensuring the
# repo root is importable for the `eval`/`api` packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from api.inference import STUB_INFERENCE_FLAG
from eval import metrics as metrics_mod
from eval.metrics import (
    compute_bertscore,
    compute_drug_entity_error_rate,
    compute_factscore,
    compute_gpt4o_preference,
    compute_hhem_detailed,
    compute_rouge_l,
    load_drugbank_vocab,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinical_notes.eval")

# (target, direction) — direction "max" means the metric must be <= target;
# "min" means it must be >= target.
EVAL_TARGETS = {
    "drug_entity_error_rate": (0.02, "max"),
    "hhem": (0.80, "min"),
    "bertscore": (0.88, "min"),
    "rouge_l": (0.42, "min"),
    "factscore": (0.75, "min"),
    "gpt4o_preference": (0.70, "min"),
}


# Where the predictions came from. A metric computed over stub output measures
# the stub, not a model: "drug_entity_error_rate: 0.0" is what a backend that
# names no drugs always scores (see test_deer_no_meds_is_zero). So provenance
# invalidates the whole run rather than any single gate.
PREDICTIONS_MODEL = "model"
PREDICTIONS_STUB = "stub"


def _splits_dir() -> Path:
    return Path(os.getenv("SPLITS_DIR", "data/splits"))


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_predictions(split: str) -> list[dict]:
    """Load model output predictions from data/splits/{split}_predictions.jsonl."""
    path = _splits_dir() / f"{split}_predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Predictions not found at {path}. Generate them before evaluating."
        )
    return _read_jsonl(path)


def detect_prediction_provenance(predictions: list[dict]) -> tuple[str, int]:
    """Classify who produced ``predictions``.

    Returns ``(provenance, stub_record_count)``. Any record carrying
    :data:`api.inference.STUB_INFERENCE_FLAG` makes the whole run
    :data:`PREDICTIONS_STUB`: the file is not a model's output, so none of the
    metrics computed over it describe a model.
    """
    stub_records = sum(
        1
        for p in predictions
        if any(
            STUB_INFERENCE_FLAG in str(flag)
            for flag in (p.get("confidence_flags") or [])
        )
    )
    provenance = PREDICTIONS_STUB if stub_records else PREDICTIONS_MODEL
    return provenance, stub_records


def load_ground_truth(split: str) -> list[dict]:
    """Load ground-truth references from data/splits/{split}.jsonl."""
    path = _splits_dir() / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth not found at {path}.")
    records = _read_jsonl(path)
    # Records may wrap the summary under a "chosen"/"summary" key; unwrap if so.
    unwrapped = []
    for r in records:
        if isinstance(r, dict) and "medications" not in r:
            for key in ("chosen", "summary", "reference"):
                if isinstance(r.get(key), dict):
                    unwrapped.append(r[key])
                    break
            else:
                unwrapped.append(r)
        else:
            unwrapped.append(r)
    return unwrapped


def run_all_metrics(predictions: list[dict], references: list[dict]) -> dict:
    """Compute all gating metrics. Metrics that cannot run are recorded as None."""
    if len(predictions) != len(references):
        raise ValueError(
            f"Prediction/reference count mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )

    vocab = load_drugbank_vocab()
    results: dict[str, object] = {}
    methods: dict[str, str] = {}

    def _safe(name, fn):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Metric %s failed: %s", name, exc)
            results[name] = None

    def _hhem():
        # Record which implementation answered: the lexical proxy clears the
        # threshold easily and must not be mistaken for a hallucination score.
        score, method = compute_hhem_detailed(predictions, references)
        methods["hhem"] = method
        return score

    _safe("drug_entity_error_rate",
          lambda: compute_drug_entity_error_rate(predictions, vocab))
    _safe("hhem", _hhem)
    _safe("bertscore", lambda: compute_bertscore(predictions, references))
    _safe("rouge_l", lambda: compute_rouge_l(predictions, references))
    _safe("factscore", lambda: compute_factscore(predictions, references))
    _safe("gpt4o_preference",
          lambda: compute_gpt4o_preference(predictions, references))
    # Carried alongside the values so check_gates sees provenance without the
    # caller having to thread a second argument through.
    results["_methods"] = methods
    return results


def check_gates(
    results: dict,
    methods: dict | None = None,
    provenance: str | None = None,
) -> dict:
    """Compare metric results against EVAL_TARGETS. Returns per-metric status.

    ``methods`` maps a metric to the implementation that produced it (see
    ``run_all_metrics``). A metric computed by a stand-in rather than its real
    implementation is marked ``degraded`` and can never read as ``pass``:
    the number may clear the threshold, but it did not measure the property the
    gate exists to protect.

    ``provenance`` says who produced the predictions (see
    ``detect_prediction_provenance``). :data:`PREDICTIONS_STUB` marks every gate
    ``invalid``, because the run scored a placeholder: its numbers are not weak
    model results, they are not model results at all. ``invalid`` outranks
    ``fail`` for exactly that reason — reporting ``fail`` here would invite the
    reader to tune a model that never ran.

    ``overall`` is ``pass`` only when every gate genuinely passed.
    """
    if methods is None:
        raw = results.get("_methods")
        methods = raw if isinstance(raw, dict) else {}
    gates = {}
    for metric, (target, direction) in EVAL_TARGETS.items():
        value = results.get(metric)
        if value is None:
            gates[metric] = {"target": target, "value": None, "status": "skipped"}
            continue
        passed = value <= target if direction == "max" else value >= target
        gate = {
            "target": target,
            "value": value,
            "direction": direction,
            "status": "pass" if passed else "fail",
        }
        if metric == "hhem" and methods.get("hhem") == metrics_mod.HHEM_PROXY:
            gate["status"] = "degraded"
            gate["method"] = metrics_mod.HHEM_PROXY
            gate["note"] = (
                "Lexical proxy, not HHEM — this value does not measure "
                "hallucination and cannot satisfy the gate."
            )
        elif metric in methods:
            gate["method"] = methods[metric]
        gates[metric] = gate

    if provenance == PREDICTIONS_STUB:
        for name, gate in gates.items():
            # A metric that never produced a value stays "skipped": nothing
            # scored the stub, so claiming it describes the stub would be false.
            if gate.get("value") is None:
                continue
            gate["status"] = "invalid"
            gate["provenance"] = PREDICTIONS_STUB
            gate["note"] = (
                "Scored against stub output, not a trained model. This value "
                "describes the stub and cannot satisfy or fail the gate."
            )

    statuses = {g["status"] for g in gates.values()}
    if "invalid" in statuses:
        overall = "invalid"
    elif "fail" in statuses:
        overall = "fail"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "pass"
    gates["overall"] = {"status": overall}
    if provenance is not None:
        gates["overall"]["provenance"] = provenance
    return gates


def save_results(results: dict, output_path: str) -> None:
    """Write evaluation results to disk as JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote evaluation results to %s", out)


def main() -> None:
    """Parse --split and run the full evaluation suite."""
    parser = argparse.ArgumentParser(
        description="Evaluation suite for Clinical Note Summarizer"
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "val", "test"],
        help="Data split to evaluate",
    )
    parser.add_argument(
        "--output", type=str, default="eval/results/latest.json",
        help="Path to write the results JSON",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Exit 0 if the suite ran and wrote a report, regardless of gate "
             "pass/fail. For CI plumbing checks on non-production data.",
    )
    args = parser.parse_args()

    predictions = load_predictions(args.split)
    references = load_ground_truth(args.split)
    provenance, stub_records = detect_prediction_provenance(predictions)
    results = run_all_metrics(predictions, references)
    gate_results = check_gates(results, provenance=provenance)
    report = {
        "split": args.split,
        # Recorded in the report itself: a reader opening latest.json weeks
        # later must not have to cross-check the predictions file to learn
        # whether a model produced these numbers.
        "provenance": {
            "predictions": provenance,
            "stub_records": stub_records,
            "total_records": len(predictions),
        },
        "metrics": results,
        "gates": gate_results,
    }
    save_results(report, args.output)

    print(json.dumps(gate_results, indent=2))

    overall = gate_results["overall"]["status"]
    if overall == "invalid":
        logger.error(
            "INVALID: %d/%d predictions carry the stub flag, so this run scored "
            "a placeholder rather than a model. Every number above describes the "
            "stub. Generate predictions with a trained checkpoint "
            "(CHECKPOINT_DIR) before reading these as results.",
            stub_records,
            len(predictions),
        )
    if overall == "degraded":
        degraded = [
            name for name, g in gate_results.items()
            if name != "overall"
            and isinstance(g, dict) and g.get("status") == "degraded"
        ]
        logger.error(
            "DEGRADED: %s did not run its real implementation. The thresholds "
            "they clear are meaningless — treat this run as unverified, not green.",
            ", ".join(degraded),
        )

    if args.smoke:
        raise SystemExit(0)
    raise SystemExit(0 if overall == "pass" else 1)


if __name__ == "__main__":
    main()
