"""Adversarial rejection generation for ORPO preference pairs.

Implements 5 failure classes from the PRD:
    A: Medication hallucination — invent medications not in the note
    B: Diagnosis omission — drop critical diagnoses
    C: Dosage mutation — alter dose, frequency, or route
    D: Timeline inversion — swap chronological ordering
    E: Contraindication reversal — reverse contraindication advice

Usage:
    python -m data.rejection_gen --input data/chosen_summaries.jsonl --output data/preference_dataset.jsonl --batch_size 500
"""

import argparse
import copy
import json
import os
import random

from dotenv import load_dotenv

load_dotenv()

FAILURE_CLASSES = ["A", "B", "C", "D", "E"]
FAILURE_WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]  # Weighted sampling for "random" mode


def generate_medication_hallucination(chosen: dict) -> dict:
    """Class A: Generate rejection by inventing medications not present in the note.

    Args:
        chosen: Valid chosen summary conforming to output schema.

    Returns:
        Rejected summary with hallucinated medication entries.
    """
    raise NotImplementedError


def generate_diagnosis_omission(chosen: dict) -> dict:
    """Class B: Generate rejection by dropping one or more critical diagnoses.

    Args:
        chosen: Valid chosen summary conforming to output schema.

    Returns:
        Rejected summary with missing diagnoses.
    """
    raise NotImplementedError


def generate_dosage_mutation(chosen: dict) -> dict:
    """Class C: Generate rejection by altering medication dose, frequency, or route.

    Args:
        chosen: Valid chosen summary conforming to output schema.

    Returns:
        Rejected summary with mutated dosage information.
    """
    raise NotImplementedError


def generate_timeline_inversion(chosen: dict) -> dict:
    """Class D: Generate rejection by swapping chronological order of events.

    Args:
        chosen: Valid chosen summary conforming to output schema.

    Returns:
        Rejected summary with inverted timeline in discharge instructions.
    """
    raise NotImplementedError


def generate_contraindication_reversal(chosen: dict) -> dict:
    """Class E: Generate rejection by reversing contraindication advice.

    Args:
        chosen: Valid chosen summary conforming to output schema.

    Returns:
        Rejected summary with dangerous contraindication reversal.
    """
    raise NotImplementedError


def generate_rejected(chosen: dict, failure_class: str = "random") -> dict:
    """Dispatch to the appropriate failure class generator.

    Args:
        chosen: Valid chosen summary.
        failure_class: One of A/B/C/D/E or "random" for weighted sampling.

    Returns:
        Rejected summary with the specified failure mode applied.
    """
    raise NotImplementedError


def build_preference_pairs(chosen_path: str, output_path: str, batch_size: int = 500) -> None:
    """Generate ORPO preference pairs from chosen summaries.

    Args:
        chosen_path: Path to chosen summaries JSONL.
        output_path: Path to write preference pairs JSONL.
        batch_size: Number of pairs to generate.
    """
    raise NotImplementedError


def main() -> None:
    """Entry point: parse arguments and generate rejection pairs."""
    parser = argparse.ArgumentParser(description="Adversarial rejection generator")
    parser.add_argument("--input", type=str, required=True, help="Path to chosen summaries JSONL")
    parser.add_argument("--output", type=str, default="data/preference_dataset.jsonl",
                        help="Output path for preference pairs")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of pairs to generate")
    parser.add_argument("--failure_class", type=str, default="random",
                        choices=["A", "B", "C", "D", "E", "random"],
                        help="Failure class to generate (default: random)")
    args = parser.parse_args()
    build_preference_pairs(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
