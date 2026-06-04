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

from eval.metrics import (
    compute_bertscore,
    compute_drug_entity_error_rate,
    compute_factscore,
    compute_gpt4o_preference,
    compute_hhem_score,
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
    results: dict[str, float | None] = {}

    def _safe(name, fn):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Metric %s failed: %s", name, exc)
            results[name] = None

    _safe("drug_entity_error_rate",
          lambda: compute_drug_entity_error_rate(predictions, vocab))
    _safe("hhem", lambda: compute_hhem_score(predictions, references))
    _safe("bertscore", lambda: compute_bertscore(predictions, references))
    _safe("rouge_l", lambda: compute_rouge_l(predictions, references))
    _safe("factscore", lambda: compute_factscore(predictions, references))
    _safe("gpt4o_preference",
          lambda: compute_gpt4o_preference(predictions, references))
    return results


def check_gates(results: dict) -> dict:
    """Compare metric results against EVAL_TARGETS. Returns per-metric status."""
    gates = {}
    for metric, (target, direction) in EVAL_TARGETS.items():
        value = results.get(metric)
        if value is None:
            gates[metric] = {"target": target, "value": None, "status": "skipped"}
            continue
        passed = value <= target if direction == "max" else value >= target
        gates[metric] = {
            "target": target,
            "value": value,
            "direction": direction,
            "status": "pass" if passed else "fail",
        }
    gates["overall"] = {
        "status": "pass"
        if all(g["status"] != "fail" for g in gates.values())
        else "fail"
    }
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
    args = parser.parse_args()

    predictions = load_predictions(args.split)
    references = load_ground_truth(args.split)
    results = run_all_metrics(predictions, references)
    gate_results = check_gates(results)
    report = {"split": args.split, "metrics": results, "gates": gate_results}
    save_results(report, args.output)

    print(json.dumps(gate_results, indent=2))
    raise SystemExit(0 if gate_results["overall"]["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
