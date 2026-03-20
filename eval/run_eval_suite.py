"""Main evaluation entry point for Clinical Note Summarizer.

Runs all gating metrics against a specified data split and outputs
a pass/fail report.

Usage:
    python eval/run_eval_suite.py --split test
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from eval.metrics import (
    compute_bertscore,
    compute_drug_entity_error_rate,
    compute_hhem_score,
    load_drugbank_vocab,
)

load_dotenv()

EVAL_TARGETS = {
    "drug_entity_error_rate": 0.02,  # ≤ 2%
    "hhem": 0.80,                     # ≥ 0.80
    "bertscore": 0.88,                # ≥ 0.88
}


def load_predictions(split: str) -> list[dict]:
    """Load model output predictions from JSONL."""
    raise NotImplementedError


def load_ground_truth(split: str) -> list[dict]:
    """Load ground-truth references from data/splits/{split}.jsonl."""
    raise NotImplementedError


def run_all_metrics(predictions: list[dict], references: list[dict]) -> dict:
    """Compute all gating metrics and return results dict."""
    raise NotImplementedError


def check_gates(results: dict) -> dict:
    """Compare metric results against EVAL_TARGETS. Return per-metric pass/fail."""
    raise NotImplementedError


def save_results(results: dict, output_path: str) -> None:
    """Write evaluation results to eval/results/latest.json."""
    raise NotImplementedError


def main() -> None:
    """Parse --split argument and run the full evaluation suite."""
    parser = argparse.ArgumentParser(description="Evaluation suite for Clinical Note Summarizer")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"],
                        help="Data split to evaluate")
    args = parser.parse_args()

    predictions = load_predictions(args.split)
    references = load_ground_truth(args.split)
    results = run_all_metrics(predictions, references)
    gate_results = check_gates(results)
    save_results({**results, "gates": gate_results}, "eval/results/latest.json")


if __name__ == "__main__":
    main()
