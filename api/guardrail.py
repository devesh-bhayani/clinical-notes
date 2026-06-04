"""DrugBank NER validation layer for Clinical Note Summarizer.

Validates medication entity names in model outputs against the DrugBank
vocabulary. Appends unrecognized names to confidence_flags without mutating
the medications array. Must not raise exceptions on malformed input.

Contract:
    Input:  parsed model output dict
    Output: same dict with confidence_flags populated
    - Must not mutate medications[] entries — only append to confidence_flags
    - Must not raise exceptions on malformed input — catch and flag instead
    - Importable as a pure function: validate_medications does no file I/O when
      a pre-loaded vocabulary is supplied.
"""

import copy
import csv
import os
import re
import string
from functools import lru_cache

from dotenv import load_dotenv
from rapidfuzz import fuzz, process

load_dotenv()

# Columns in the official DrugBank vocabulary export that contain drug names.
# "Synonyms" is a "| "-delimited list. We are permissive about exact header
# casing/whitespace and fall back to the first column for simple single-column
# vocab files used in tests.
_NAME_COLUMNS = ("common name", "name", "drug", "drug name")
_SYNONYM_COLUMNS = ("synonyms", "synonym")
_SYNONYM_SPLIT = re.compile(r"\s*\|\s*")
# Strip surrounding punctuation but keep internal hyphens (e.g. "co-trimoxazole").
_PUNCT_STRIP = string.punctuation


def normalize_name(name: str) -> str:
    """Normalize a drug name for vocabulary lookup.

    Lowercases, collapses internal whitespace, and strips surrounding
    punctuation. Returns an empty string for non-string or blank input.
    """
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Strip leading/trailing punctuation tokens without touching internal hyphens.
    cleaned = cleaned.strip(_PUNCT_STRIP + " ")
    return cleaned


def load_drugbank_vocabulary(vocab_path: str | None = None) -> set[str]:
    """Load and normalize drug names from a DrugBank CSV.

    Args:
        vocab_path: Path to CSV. Defaults to os.getenv("DRUGBANK_VOCAB_PATH").

    Returns:
        Set of lowercase, normalized drug names (common names + synonyms).

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ValueError: If no path is configured.
    """
    path = vocab_path or os.getenv("DRUGBANK_VOCAB_PATH")
    if not path:
        raise ValueError(
            "No DrugBank vocabulary path provided and DRUGBANK_VOCAB_PATH is unset."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DrugBank vocabulary not found at: {path}")

    vocab: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return vocab

        lowered = [h.strip().lower() for h in header]
        name_idx = next((i for i, h in enumerate(lowered) if h in _NAME_COLUMNS), None)
        syn_idx = next((i for i, h in enumerate(lowered) if h in _SYNONYM_COLUMNS), None)

        # Single-column / headerless vocab: treat every row's first cell as a name
        # and the header row itself as data.
        if name_idx is None:
            name_idx = 0
            for cell in (header[0:1] if header else []):
                norm = normalize_name(cell)
                if norm:
                    vocab.add(norm)

        for row in reader:
            if not row:
                continue
            if name_idx < len(row):
                norm = normalize_name(row[name_idx])
                if norm:
                    vocab.add(norm)
            if syn_idx is not None and syn_idx < len(row) and row[syn_idx]:
                for syn in _SYNONYM_SPLIT.split(row[syn_idx]):
                    norm = normalize_name(syn)
                    if norm:
                        vocab.add(norm)
    return vocab


@lru_cache(maxsize=4)
def _cached_vocabulary(vocab_path: str | None) -> frozenset[str]:
    """Process-level cache so the API does not re-read the CSV per request."""
    return frozenset(load_drugbank_vocabulary(vocab_path))


def get_vocabulary(vocab_path: str | None = None) -> set[str]:
    """Return the DrugBank vocabulary, cached by resolved path.

    Convenience for long-lived processes (the API). Pure callers should load
    once at startup and pass the set into validate_medications directly.
    """
    path = vocab_path or os.getenv("DRUGBANK_VOCAB_PATH")
    return set(_cached_vocabulary(path))


def _is_recognized(name: str, vocab: set[str], fuzzy_threshold: int) -> bool:
    """Return True if a normalized name matches the vocabulary."""
    if name in vocab:
        return True
    if fuzzy_threshold <= 0 or not vocab:
        return False
    match = process.extractOne(
        name, vocab, scorer=fuzz.WRatio, score_cutoff=fuzzy_threshold
    )
    return match is not None


def validate_medications(
    output: dict,
    vocab: set[str] | None = None,
    fuzzy_threshold: int = 88,
) -> dict:
    """Validate medication names against the DrugBank vocabulary.

    For each entry in output["medications"], checks whether the name matches a
    vocabulary entry (exact-normalized first, then fuzzy match above the
    threshold). Unrecognized or malformed names are appended to
    confidence_flags. The medications array is never mutated.

    Args:
        output: Model output dict conforming to the output schema.
        vocab: Pre-loaded vocabulary set. If None, loads (and caches) from the
            DRUGBANK_VOCAB_PATH env var.
        fuzzy_threshold: Minimum WRatio score (0-100) to accept a fuzzy match.

    Returns:
        A copy of the output dict with confidence_flags updated. Never raises.
    """
    if vocab is None:
        try:
            vocab = get_vocabulary()
        except (FileNotFoundError, ValueError):
            # Fail safe: with no vocabulary we cannot verify anything, so flag
            # rather than silently passing unrecognized drugs.
            vocab = set()

    # Never mutate the caller's object.
    if not isinstance(output, dict):
        return {
            "diagnoses": [],
            "medications": [],
            "procedures": [],
            "discharge_instructions": "",
            "confidence_flags": ["guardrail: malformed model output (expected object)"],
        }

    result = copy.deepcopy(output)
    flags = result.get("confidence_flags")
    if not isinstance(flags, list):
        flags = []
    result["confidence_flags"] = flags

    medications = result.get("medications")
    if not isinstance(medications, list):
        if medications is not None:
            flags.append("guardrail: medications field is not a list")
        return result

    for med in medications:
        try:
            raw_name = med.get("name") if isinstance(med, dict) else med
            norm = normalize_name(raw_name)
            if not norm:
                flags.append("guardrail: medication entry missing a name")
                continue
            if not _is_recognized(norm, vocab, fuzzy_threshold):
                display = raw_name if isinstance(raw_name, str) else repr(raw_name)
                flags.append(
                    f"guardrail: unrecognized medication '{display}' "
                    f"not found in DrugBank vocabulary"
                )
        except Exception:  # noqa: BLE001 - guardrail must never raise
            flags.append("guardrail: could not validate a medication entry")

    return result
