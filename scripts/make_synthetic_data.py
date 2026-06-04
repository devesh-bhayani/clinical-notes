"""Generate a synthetic, PHI-free dataset to smoke-test the full pipeline.

Exercises the real code paths — MIMIC extraction, ORPO rejection generation,
stub inference, and evaluation — using fabricated notes and the committed sample
DrugBank vocabulary. No MIMIC data and no real PHI are involved, so this is safe
to run anywhere (CI, laptops) without the HIPAA-regulated corpus.

Outputs (under --output, default data/splits):
    chosen_summaries.jsonl   pipeline extraction output  ({note, chosen})
    train_orpo.jsonl         ORPO preference pairs       ({prompt, chosen, rejected})
    test.jsonl               eval references             ({chosen})
    test_predictions.jsonl   stub-model predictions      (output schema)

Usage:
    python scripts/make_synthetic_data.py --output data/splits
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Default to the committed sample vocab so the script is self-contained.
os.environ.setdefault(
    "DRUGBANK_VOCAB_PATH",
    str(_REPO_ROOT / "data" / "drugbank_vocabulary.sample.csv"),
)
os.environ.setdefault("ALLOW_STUB_INFERENCE", "1")

from api.guardrail import get_vocabulary  # noqa: E402
from api.inference import StubSummarizer  # noqa: E402
from data.pipeline import process_dataset  # noqa: E402
from data.rejection_gen import build_preference_pairs  # noqa: E402

# Fabricated discharge notes. Deliberately avoid PHI header patterns
# (Admission Date / Service / Attending / Chief Complaint) so the compliance
# checker stays green; medications are real DrugBank entries from the sample.
SYNTHETIC_NOTES: list[str] = [
    """\
Discharge Diagnosis:
Type 2 diabetes mellitus
Hypertension

Discharge Medications:
1. Aspirin 81 mg PO daily
2. Metformin 500 mg PO twice daily

Major Procedures:
Colonoscopy

Discharge Instructions:
Do not skip doses. Follow up with your primary care provider in two weeks.
""",
    """\
Discharge Diagnosis:
Atrial fibrillation
Hyperlipidemia

Discharge Medications:
1. Warfarin 5 mg PO daily
2. Atorvastatin 40 mg PO daily

Major Procedures:
Echocardiogram

Discharge Instructions:
Avoid contact sports while on anticoagulation. Return for INR check in one week.
""",
    """\
Discharge Diagnosis:
Community acquired pneumonia

Discharge Medications:
1. Azithromycin 250 mg PO daily
2. Ibuprofen 400 mg PO every 6 hours as needed

Major Procedures:
Chest radiograph

Discharge Instructions:
Complete the full antibiotic course. Rest and stay hydrated at home.
""",
    """\
Discharge Diagnosis:
Gastroesophageal reflux disease

Discharge Medications:
1. Omeprazole 20 mg PO daily

Major Procedures:
Upper endoscopy

Discharge Instructions:
Take medication before breakfast. Avoid late meals and follow up as needed.
""",
]


def write_synthetic_csv(path: Path) -> None:
    """Write the synthetic notes as a MIMIC-style notes CSV."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["category", "text"])
        for note in SYNTHETIC_NOTES:
            writer.writerow(["Discharge summary", note])


def build_synthetic_dataset(output_dir: str | Path) -> dict[str, str]:
    """Build the full synthetic dataset and return a map of artifact paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Raw notes CSV -> pipeline extraction (chosen_summaries.jsonl).
    csv_path = out / "synthetic_notes.csv"
    write_synthetic_csv(csv_path)
    n_chosen = process_dataset(str(csv_path), str(out), sample_size=0)
    chosen_path = out / "chosen_summaries.jsonl"

    # 2) Chosen summaries -> ORPO preference pairs (train_orpo.jsonl).
    train_path = out / "train_orpo.jsonl"
    n_pairs = build_preference_pairs(
        str(chosen_path), str(train_path), batch_size=len(SYNTHETIC_NOTES)
    )

    # 3) Eval references (test.jsonl) + stub predictions (test_predictions.jsonl).
    references_path = out / "test.jsonl"
    predictions_path = out / "test_predictions.jsonl"
    summarizer = StubSummarizer()
    with open(chosen_path, encoding="utf-8") as src, \
            open(references_path, "w", encoding="utf-8") as ref_fh, \
            open(predictions_path, "w", encoding="utf-8") as pred_fh:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ref_fh.write(json.dumps({"chosen": record["chosen"]}) + "\n")
            prediction = summarizer.summarize(record["note"])
            pred_fh.write(json.dumps(prediction) + "\n")

    return {
        "notes_csv": str(csv_path),
        "chosen_summaries": str(chosen_path),
        "train_orpo": str(train_path),
        "test": str(references_path),
        "test_predictions": str(predictions_path),
        "n_chosen": str(n_chosen),
        "n_pairs": str(n_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PHI-free dataset")
    parser.add_argument(
        "--output", type=str, default=os.getenv("SPLITS_DIR", "data/splits"),
        help="Output directory for the synthetic artifacts",
    )
    args = parser.parse_args()

    # Sanity check the vocab is reachable before doing work.
    get_vocabulary()
    artifacts = build_synthetic_dataset(args.output)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
