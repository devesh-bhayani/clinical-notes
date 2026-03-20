"""Metric computation for Clinical Note Summarizer evaluation.

Gating metrics (all must pass for release):
    - Drug Entity Error Rate: ≤ 2%
    - HHEM (hallucination score): ≥ 0.80
    - BERTScore F1: ≥ 0.88
"""

import csv
import os

from dotenv import load_dotenv

load_dotenv()


def load_drugbank_vocab(vocab_path: str | None = None) -> set[str]:
    """Load and normalize DrugBank vocabulary from CSV.

    Args:
        vocab_path: Path to drugbank_vocabulary.csv. Defaults to DRUGBANK_VOCAB_PATH env var.
    """
    raise NotImplementedError


def compute_drug_entity_error_rate(predictions: list[dict], vocab: set[str]) -> float:
    """Compute percentage of medication names in predictions not found in DrugBank.

    Args:
        predictions: Model outputs conforming to the output schema.
        vocab: Normalized set of known drug names from DrugBank.

    Returns:
        Error rate as a float between 0.0 and 1.0.
    """
    raise NotImplementedError


def compute_hhem_score(predictions: list[dict], references: list[dict]) -> float:
    """Compute HHEM hallucination score between predictions and references.

    Args:
        predictions: Model-generated summaries.
        references: Ground-truth summaries.

    Returns:
        HHEM score (higher is better, target ≥ 0.80).
    """
    raise NotImplementedError


def compute_bertscore(predictions: list[dict], references: list[dict]) -> float:
    """Compute BERTScore F1 between predictions and references.

    Args:
        predictions: Model-generated summaries.
        references: Ground-truth summaries.

    Returns:
        BERTScore F1 (target ≥ 0.88).
    """
    raise NotImplementedError
