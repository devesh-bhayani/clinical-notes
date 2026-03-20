"""Demographic bias audit for MIMIC-IV training data.

Analyzes age, race/ethnicity, and admission type distributions.
Flags underrepresented groups and applies oversampling if needed.

Usage:
    python -m data.bias_audit --input data/splits/train.jsonl
"""

import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def compute_age_distribution(df: pd.DataFrame) -> dict:
    """Compute age group breakdowns across the dataset.

    Returns:
        Dict mapping age groups to counts and percentages.
    """
    raise NotImplementedError


def compute_ethnicity_distribution(df: pd.DataFrame) -> dict:
    """Compute race/ethnicity breakdowns across the dataset.

    Returns:
        Dict mapping ethnicity groups to counts and percentages.
    """
    raise NotImplementedError


def compute_admission_type_distribution(df: pd.DataFrame) -> dict:
    """Compute admission type breakdowns (emergency, elective, etc.).

    Returns:
        Dict mapping admission types to counts and percentages.
    """
    raise NotImplementedError


def detect_imbalance(distribution: dict, threshold: float = 0.1) -> list[str]:
    """Flag groups that fall below the representation threshold.

    Args:
        distribution: Group-to-percentage mapping.
        threshold: Minimum acceptable representation ratio.

    Returns:
        List of underrepresented group names.
    """
    raise NotImplementedError


def apply_oversampling(df: pd.DataFrame, column: str, target_ratio: float) -> pd.DataFrame:
    """Oversample minority groups to reach target representation ratio.

    Args:
        df: Input DataFrame.
        column: Column to balance on.
        target_ratio: Target minimum ratio for each group.

    Returns:
        Rebalanced DataFrame.
    """
    raise NotImplementedError


def run_full_audit(df: pd.DataFrame) -> dict:
    """Run complete bias audit: all distributions + imbalance flags.

    Returns:
        Dict with age, ethnicity, admission_type distributions and flags.
    """
    raise NotImplementedError


def main() -> None:
    """Entry point: parse arguments and run bias audit."""
    parser = argparse.ArgumentParser(description="Bias audit for MIMIC-IV training data")
    parser.add_argument("--input", type=str, required=True, help="Path to training data JSONL")
    parser.add_argument("--output", type=str, default="logs/bias_audit.json",
                        help="Output path for audit report")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
