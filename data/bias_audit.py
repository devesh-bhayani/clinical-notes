"""Demographic bias audit for MIMIC-IV training data.

Analyzes age, race/ethnicity, and admission type distributions. Flags
underrepresented groups and can oversample to a target ratio. Operates on a
DataFrame of metadata rows (one per record) — never on the raw notes — so it is
safe to run on de-identified demographic fields only.

Usage:
    python -m data.bias_audit --input data/splits/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Standard age buckets for clinical cohorts.
_AGE_BINS = [0, 18, 30, 45, 60, 75, 200]
_AGE_LABELS = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]


def _distribution(series: pd.Series) -> dict:
    """Return {group: {"count": int, "pct": float}} for a categorical series."""
    series = series.dropna()
    total = int(series.shape[0])
    counts = series.value_counts()
    out = {}
    for group, count in counts.items():
        out[str(group)] = {
            "count": int(count),
            "pct": (int(count) / total) if total else 0.0,
        }
    return out


def compute_age_distribution(df: pd.DataFrame) -> dict:
    """Compute age-group breakdowns across the dataset."""
    if "age" not in df.columns:
        return {}
    ages = pd.to_numeric(df["age"], errors="coerce")
    groups = pd.cut(ages, bins=_AGE_BINS, labels=_AGE_LABELS, right=False)
    return _distribution(groups.astype("object"))


def compute_ethnicity_distribution(df: pd.DataFrame) -> dict:
    """Compute race/ethnicity breakdowns across the dataset."""
    col = next((c for c in ("ethnicity", "race") if c in df.columns), None)
    if col is None:
        return {}
    return _distribution(df[col].astype("object"))


def compute_admission_type_distribution(df: pd.DataFrame) -> dict:
    """Compute admission-type breakdowns (emergency, elective, etc.)."""
    col = next(
        (c for c in ("admission_type", "admission", "adm_type") if c in df.columns),
        None,
    )
    if col is None:
        return {}
    return _distribution(df[col].astype("object"))


def detect_imbalance(distribution: dict, threshold: float = 0.1) -> list[str]:
    """Flag groups whose representation falls below ``threshold``."""
    return [
        group
        for group, stats in distribution.items()
        if stats.get("pct", 0.0) < threshold
    ]


def apply_oversampling(
    df: pd.DataFrame, column: str, target_ratio: float
) -> pd.DataFrame:
    """Oversample minority groups in ``column`` up to ``target_ratio``.

    Replicates rows (with replacement) from any group below the target share
    until it reaches that share. Returns a new, shuffled DataFrame.
    """
    if column not in df.columns or df.empty:
        return df.copy()

    frames = [df]
    total = len(df)
    counts = df[column].value_counts()
    for group, count in counts.items():
        target_count = int(target_ratio * total)
        if count < target_count:
            deficit = target_count - count
            pool = df[df[column] == group]
            sampled = pool.sample(n=deficit, replace=True, random_state=42)
            frames.append(sampled)
    result = pd.concat(frames, ignore_index=True)
    return result.sample(frac=1.0, random_state=42).reset_index(drop=True)


def run_full_audit(df: pd.DataFrame, threshold: float = 0.1) -> dict:
    """Run the complete bias audit: all distributions plus imbalance flags."""
    age = compute_age_distribution(df)
    ethnicity = compute_ethnicity_distribution(df)
    admission = compute_admission_type_distribution(df)
    return {
        "n_records": int(len(df)),
        "distributions": {
            "age": age,
            "ethnicity": ethnicity,
            "admission_type": admission,
        },
        "underrepresented": {
            "age": detect_imbalance(age, threshold),
            "ethnicity": detect_imbalance(ethnicity, threshold),
            "admission_type": detect_imbalance(admission, threshold),
        },
        "threshold": threshold,
    }


def _load_records(input_path: str) -> pd.DataFrame:
    rows = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", rec) if isinstance(rec, dict) else {}
            rows.append(meta)
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point: parse arguments and run the bias audit."""
    parser = argparse.ArgumentParser(description="Bias audit for MIMIC-IV training data")
    parser.add_argument("--input", type=str, required=True, help="Path to training data JSONL")
    parser.add_argument("--output", type=str, default="logs/bias_audit.json",
                        help="Output path for audit report")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Minimum representation ratio before flagging")
    args = parser.parse_args()

    df = _load_records(args.input)
    report = run_full_audit(df, args.threshold)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
