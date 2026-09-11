"""Generate model predictions for an eval split.

Bridges the gap between a trained adapter and eval/run_eval_suite.py, which
loads predictions from ``{SPLITS_DIR}/{split}_predictions.jsonl`` but does not
produce them. Without this step the only predictions in the repo come from
StubSummarizer via scripts/make_synthetic_data.py, so no real checkpoint can
ever be scored.

Output is one JSON object per line, in the same order as the split's references
— run_eval_suite.py zips the two files positionally and rejects a length
mismatch, so order and count must match exactly.

Usage:
    # real adapter (auto-detects CUDA 4-bit vs Apple Silicon bf16)
    python scripts/generate_predictions.py --split test --checkpoint models/mac

    # plumbing check with no GPU and no model
    ALLOW_STUB_INFERENCE=1 python scripts/generate_predictions.py --split test --stub
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from api.inference import build_prompt  # noqa: F401  (re-exported for callers)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinical_notes.predict")


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


def load_notes(split: str) -> list[str]:
    """Return the source note for every row of ``{split}.jsonl``, in order.

    Splits written after the note was added to _write_refs carry it inline.
    Older splits stored only ``{"chosen": ...}``, so fall back to recovering the
    note from chosen_summaries.jsonl by matching the reference summary. The
    fallback refuses to guess: if any row is missing or ambiguous it raises,
    because silently mis-pairing notes with references would produce eval
    numbers that look plausible and mean nothing.
    """
    split_path = _splits_dir() / f"{split}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split not found at {split_path}.")
    rows = _read_jsonl(split_path)

    if all(isinstance(r.get("note"), str) and r["note"] for r in rows):
        return [r["note"] for r in rows]

    source = _splits_dir() / "chosen_summaries.jsonl"
    if not source.is_file():
        raise ValueError(
            f"{split_path} has no 'note' field and {source} is absent, so the "
            f"model input cannot be recovered. Rebuild the splits "
            f"(python -m data.asclepius --output {_splits_dir()} --make-splits) "
            f"— they now carry the note inline."
        )

    logger.warning(
        "%s predates self-contained splits; recovering notes from %s by "
        "matching reference summaries.", split_path.name, source.name,
    )
    by_chosen: dict[str, list[str]] = {}
    for rec in _read_jsonl(source):
        key = json.dumps(rec.get("chosen"), sort_keys=True)
        by_chosen.setdefault(key, []).append(rec.get("note", ""))

    notes, unmatched, ambiguous = [], 0, 0
    for row in rows:
        key = json.dumps(row.get("chosen"), sort_keys=True)
        candidates = by_chosen.get(key, [])
        if not candidates:
            unmatched += 1
            notes.append(None)
        elif len({c for c in candidates}) > 1:
            ambiguous += 1
            notes.append(None)
        else:
            notes.append(candidates[0])

    if unmatched or ambiguous:
        raise ValueError(
            f"Could not recover notes for {unmatched} unmatched and "
            f"{ambiguous} ambiguous row(s) of {split_path.name}. Rebuild the "
            f"splits so they carry the note inline rather than evaluating "
            f"against mis-paired inputs."
        )
    return notes


def build_summarizer(checkpoint: str | None, use_stub: bool):
    """Pick an inference backend, refusing to fake it by accident."""
    from api.inference import BioMistralSummarizer, StubSummarizer

    if use_stub:
        if os.getenv("ALLOW_STUB_INFERENCE") != "1":
            raise SystemExit(
                "--stub requires ALLOW_STUB_INFERENCE=1. Stub output is not "
                "model-generated and must never be mistaken for an eval result."
            )
        logger.warning(
            "StubSummarizer: output is NOT model-generated. Any metric computed "
            "from these predictions measures the stub, not a trained model."
        )
        return StubSummarizer()

    if not checkpoint:
        raise SystemExit(
            "Pass --checkpoint <adapter dir> (or set CHECKPOINT_DIR), or --stub "
            "for a plumbing check."
        )
    if not Path(checkpoint).is_dir():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint}")

    logger.info("Loading adapter from %s", checkpoint)
    return BioMistralSummarizer(checkpoint)


def generate(split: str, summarizer, limit: int | None = None) -> Path:
    """Summarize every note in ``split`` and write ``{split}_predictions.jsonl``."""
    notes = load_notes(split)
    if limit:
        notes = notes[:limit]

    out_path = _splits_dir() / f"{split}_predictions.jsonl"
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, note in enumerate(notes, 1):
            try:
                prediction = summarizer.summarize(note)
            except Exception as exc:  # noqa: BLE001
                # An empty schema-shaped record keeps the file aligned with the
                # references; dropping the row would silently shift every later
                # pairing and corrupt the whole eval.
                logger.error("Note %d failed to summarize: %s", i, exc)
                prediction = {
                    "diagnoses": [], "medications": [], "procedures": [],
                    "discharge_instructions": "",
                    "confidence_flags": ["generation_failed"],
                }
            fh.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            written += 1
            if i % 10 == 0 or i == len(notes):
                logger.info("  %d/%d", i, len(notes))

    logger.info("Wrote %d predictions to %s", written, out_path)
    if limit:
        logger.warning(
            "--limit %d was used, so this file has fewer rows than %s.jsonl. "
            "run_eval_suite.py will reject the count mismatch; use it for "
            "smoke checks only.", limit, split,
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions for an eval split"
    )
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--checkpoint", default=os.getenv("CHECKPOINT_DIR") or None,
        help="Trained adapter directory (defaults to CHECKPOINT_DIR)",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="Use the stub backend; requires ALLOW_STUB_INFERENCE=1",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only summarize the first N notes (smoke checks; breaks eval alignment)",
    )
    args = parser.parse_args()

    summarizer = build_summarizer(args.checkpoint, args.stub)
    generate(args.split, summarizer, args.limit)


if __name__ == "__main__":
    main()
