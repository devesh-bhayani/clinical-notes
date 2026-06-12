"""Asclepius synthetic-clinical-notes adapter.

Turns the public, DUA-free Asclepius dataset
(``starmpcc/Asclepius-Synthetic-Clinical-Notes``, 157k synthetic discharge
summaries generated from PMC case reports) into the project's training inputs:

    Asclepius notes ──▶ chosen_summaries.jsonl  (deterministic extraction)
                   └──▶ train_orpo.jsonl / val.jsonl / test.jsonl

Because the notes are fully synthetic there is no HIPAA exposure, but they carry
PHI-shaped section headers (admission/discharge dates, record numbers), so the
generated artifacts are gitignored and must never be committed.

Note on coverage: Asclepius notes are narrative. Diagnoses/instructions extract
well; medications are frequently "None"/"N/A" in the source, so the medications
field is often empty (faithful to the note). For a medication-dense corpus, use
MIMIC-IV discharge meds via data/pipeline.py instead.

Usage:
    python -m data.asclepius --output data/splits --limit 5000 --make-splits
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

from api.guardrail import get_vocabulary
from api.inference import build_prompt
from data.pipeline import build_chosen_summary
from data.rejection_gen import generate_rejected

load_dotenv()

ASCLEPIUS_DATASET = "starmpcc/Asclepius-Synthetic-Clinical-Notes"


def iter_asclepius_notes(
    limit: int | None = None,
    split: str = "train",
    streaming: bool = True,
) -> Iterator[str]:
    """Yield unique clinical notes from the Asclepius dataset.

    The dataset has multiple task rows per note; this dedups by patient_id (or
    the note text) so each clinical note is yielded once. Requires the
    ``datasets`` library and network/cache access.
    """
    from datasets import load_dataset

    dataset = load_dataset(ASCLEPIUS_DATASET, split=split, streaming=streaming)
    seen: set = set()
    for row in dataset:
        note = row.get("note")
        if not note:
            continue
        key = row.get("patient_id", note)
        if key in seen:
            continue
        seen.add(key)
        yield note
        if limit and len(seen) >= limit:
            break


def build_chosen_summaries(
    output_dir: str | Path,
    limit: int | None = 5000,
    split: str = "train",
    min_diagnoses: int = 1,
    notes: Iterator[str] | None = None,
) -> dict:
    """Extract chosen summaries from Asclepius notes into chosen_summaries.jsonl.

    Args:
        output_dir: Directory to write chosen_summaries.jsonl.
        limit: Max unique notes to process (None = all ~157k).
        split: Dataset split.
        min_diagnoses: Drop notes that yield fewer than this many diagnoses, so
            the chosen targets are substantive (a good ORPO "chosen" must be good).
        notes: Optional pre-built iterable of note strings (used by tests to
            avoid the network); defaults to streaming Asclepius.

    Returns:
        Stats dict: {processed, kept, output}.
    """
    vocab = get_vocabulary()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "chosen_summaries.jsonl"

    source = notes if notes is not None else iter_asclepius_notes(limit=limit, split=split)
    processed = kept = 0
    with open(path, "w", encoding="utf-8") as fh:
        for note in source:
            processed += 1
            summary = build_chosen_summary(note, vocab)
            if len(summary["diagnoses"]) < min_diagnoses:
                continue
            fh.write(json.dumps({"note": note, "chosen": summary}, ensure_ascii=False) + "\n")
            kept += 1
    return {"processed": processed, "kept": kept, "output": str(path)}


def prepare_orpo_splits(
    chosen_path: str | Path,
    output_dir: str | Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    failure_class: str = "random",
    seed: int = 42,
) -> dict:
    """Split chosen summaries into ORPO train pairs + val/test references.

    Writes:
        train_orpo.jsonl  {prompt, chosen, rejected, failure_class}
        val.jsonl         {chosen}
        test.jsonl        {chosen}

    Returns counts per split.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    with open(chosen_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test, val, train = records[:n_test], records[n_test:n_test + n_val], records[n_test + n_val:]

    def _write_refs(path, rows):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"chosen": r["chosen"]}, ensure_ascii=False) + "\n")

    _write_refs(out / "val.jsonl", val)
    _write_refs(out / "test.jsonl", test)

    train_path = out / "train_orpo.jsonl"
    with open(train_path, "w", encoding="utf-8") as fh:
        for r in train:
            chosen = r["chosen"]
            rejected = generate_rejected(chosen, failure_class)
            pair = {
                "prompt": build_prompt(r.get("note", "")),
                "chosen": json.dumps(chosen, ensure_ascii=False),
                "rejected": json.dumps(rejected, ensure_ascii=False),
                "failure_class": failure_class if failure_class != "random" else "mixed",
            }
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return {"train": len(train), "val": len(val), "test": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Asclepius -> ORPO dataset adapter")
    parser.add_argument("--output", type=str, default=os.getenv("SPLITS_DIR", "data/splits"),
                        help="Output directory for JSONL artifacts")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max unique notes to process (0 = all)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--min-diagnoses", type=int, default=1,
                        help="Drop notes yielding fewer than N diagnoses")
    parser.add_argument("--make-splits", action="store_true",
                        help="Also build train_orpo/val/test from the chosen summaries")
    args = parser.parse_args()

    limit = args.limit or None
    stats = build_chosen_summaries(
        args.output, limit=limit, split=args.split, min_diagnoses=args.min_diagnoses
    )
    print(json.dumps(stats, indent=2))

    if args.make_splits:
        split_stats = prepare_orpo_splits(stats["output"], args.output)
        print(json.dumps(split_stats, indent=2))


if __name__ == "__main__":
    main()
