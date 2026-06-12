"""Deterministic MIMIC-IV extraction pipeline.

Filters discharge summaries, extracts structured sections via regex (with an
optional spaCy NER pass for medications), validates medications against
DrugBank, and produces *chosen* summaries in the output schema format.

The pipeline is deterministic so dataset builds are reproducible. spaCy is used
when available; otherwise a regex-based medication parser is used so the
pipeline runs without the model download.

Usage:
    python -m data.pipeline --input /path/to/mimic --output data/splits/ --sample 10000
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv

from api.guardrail import get_vocabulary, normalize_name
from api.guardrail import _is_recognized  # internal, reused for cross-validation

load_dotenv()

# Section headers found in MIMIC discharge summaries and in narrative synthetic
# notes (e.g. Asclepius). Patterns are ordered general-to-specific via alternation
# and matched case-insensitively.
_SECTION_PATTERNS = {
    "diagnoses": re.compile(
        r"(?:discharge diagnos[ei]s|final diagnos[ei]s|principal diagnos[ei]s"
        r"|admission diagnos[ei]s|diagnos[ei]s)\s*:?\s*\n?(.*?)"
        r"(?=\n\s*[A-Z][A-Za-z /]+:|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    "medications": re.compile(
        r"(?:discharge medications?|medications? on discharge|medications?)\s*:?\s*\n?(.*?)"
        r"(?=\n\s*[A-Z][A-Za-z /]+:|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    "procedures": re.compile(
        r"(?:major (?:surgical or invasive )?procedures?|procedures?)\s*:?\s*\n?(.*?)"
        r"(?=\n\s*[A-Z][A-Za-z /]+:|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    "discharge_instructions": re.compile(
        r"(?:discharge instructions?|followup instructions?)\s*:?\s*\n?(.*?)"
        r"(?=\n\s*[A-Z][A-Za-z /]+:|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
}

# Parse a medication line like "Aspirin 81 mg PO daily".
_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?|tablet?s?|puffs?))", re.IGNORECASE)
_ROUTE_RE = re.compile(r"\b(PO|IV|IM|SC|SQ|topical|oral|inhaled|nasal|rectal)\b", re.IGNORECASE)
_FREQ_RE = re.compile(
    r"\b(once daily|twice daily|q\.?d\.?|b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|"
    r"q\d+h|every \d+ hours?|daily|weekly|prn|as needed)\b",
    re.IGNORECASE,
)
_LIST_SPLIT = re.compile(r"\n|;|(?:^|\n)\s*\d+[.)]\s*", re.MULTILINE)

# Prose-section helpers (narrative notes where sections are sentences, not lists).
# A "negative" section ("None", "N/A", "not listed") yields no entries.
_NEG_SECTION = re.compile(
    r"^\s*(none|n/?a|not (?:specified|listed|applicable|provided|available|reported)"
    r"|no (?:\w+\s+)?medications?|unknown)\b",
    re.IGNORECASE,
)
# Lead-ins to strip from prose diagnosis blocks before extracting terms.
_DX_LEADIN = re.compile(
    r"^(?:the patient (?:was|is|has been) (?:discharged|diagnosed)(?: with)?(?: a)?"
    r"(?: diagnosis of)?|diagnos(?:is|ed)(?: with| of)?|discharged with(?: a)?"
    r"(?: diagnosis of)?)\s*",
    re.IGNORECASE,
)
_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+")


def filter_discharge_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a MIMIC-IV notes DataFrame to discharge summaries only."""
    if "category" in df.columns:
        mask = df["category"].astype(str).str.contains("discharge", case=False, na=False)
        return df[mask].copy()
    if "note_type" in df.columns:
        mask = df["note_type"].astype(str).str.upper().str.startswith("DS")
        return df[mask].copy()
    return df.copy()


def _clean_items(block: str) -> list[str]:
    items = []
    for raw in _LIST_SPLIT.split(block or ""):
        item = re.sub(r"\s+", " ", (raw or "").strip(" \t-*.")).strip()
        if item and not item.isdigit():
            items.append(item)
    return items


def extract_diagnoses(block: str) -> list[str]:
    """Extract diagnosis terms from a section block (list-style or prose).

    List-style blocks (newlines or numbered/bulleted items) are split on those
    markers. Prose blocks ("...diagnosis of A, B, and C") have their lead-in
    stripped and are list-split only when the result is clearly a short-term
    enumeration; otherwise the leading clause is kept as a single diagnosis.
    Negative sections ("None", "N/A") yield no diagnoses.
    """
    block = (block or "").strip()
    if not block or _NEG_SECTION.match(block):
        return []
    block = _DX_LEADIN.sub("", block).strip()

    if _LIST_MARKER.search(block) or "\n" in block:
        parts = _LIST_SPLIT.split(block)
    else:
        first_sentence = re.split(r"(?<=[.])\s", block, maxsplit=1)[0]
        segments = re.split(r",|\band\b", first_sentence)
        if len(segments) >= 2 and all(len(s.split()) <= 5 for s in segments):
            parts = segments
        else:
            parts = [first_sentence]

    out: list[str] = []
    seen: set[str] = set()
    for raw in parts:
        item = re.sub(r"\s+", " ", (raw or "")).strip(" .,-*\t")
        if not (2 < len(item) <= 90):
            continue
        if item.lower().startswith(("the ", "he ", "she ", "patient", "based on")):
            continue
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out[:8]


def extract_sections(note: str) -> dict:
    """Extract diagnoses, medications, procedures, and instructions via regex."""
    sections = {
        "diagnoses": [],
        "medications_raw": "",
        "procedures": [],
        "discharge_instructions": "",
    }
    if not isinstance(note, str):
        return sections

    for key, pattern in _SECTION_PATTERNS.items():
        match = pattern.search(note)
        block = match.group(1).strip() if match else ""
        if key == "medications":
            # A negative section ("None", "N/A") carries no medications.
            sections["medications_raw"] = "" if _NEG_SECTION.match(block) else block
        elif key == "discharge_instructions":
            text = "" if _NEG_SECTION.match(block) else block
            sections["discharge_instructions"] = re.sub(r"\s+", " ", text).strip()
        elif key == "diagnoses":
            sections[key] = extract_diagnoses(block)
        else:
            sections[key] = _clean_items(block)
    return sections


def _parse_med_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    dose = _DOSE_RE.search(line)
    route = _ROUTE_RE.search(line)
    freq = _FREQ_RE.search(line)
    # The name is the leading text before the first dose/route/freq token.
    cut = len(line)
    for m in (dose, route, freq):
        if m:
            cut = min(cut, m.start())
    name = line[:cut].strip(" ,-")
    if not name:
        return None
    return {
        "name": name,
        "dose": dose.group(1).strip() if dose else "",
        "freq": freq.group(1).strip() if freq else "",
        "route": route.group(1).strip() if route else "",
    }


def extract_medications_spacy(text: str) -> list[dict]:
    """Extract medication entities, using spaCy NER when available.

    Falls back to line-by-line regex parsing if spaCy or a suitable model is not
    installed. Returns a list of {name, dose, freq, route} dicts.
    """
    meds: list[dict] = []
    nlp = _load_spacy()
    if nlp is not None:
        doc = nlp(text or "")
        seen = set()
        for ent in doc.ents:
            if ent.label_ in {"CHEMICAL", "DRUG", "PRODUCT"}:
                parsed = _parse_med_line(ent.sent.text)
                if parsed and parsed["name"].lower() not in seen:
                    seen.add(parsed["name"].lower())
                    meds.append(parsed)
        if meds:
            return meds

    # Regex fallback: one medication per line.
    for line in _LIST_SPLIT.split(text or ""):
        parsed = _parse_med_line(line)
        if parsed:
            meds.append(parsed)
    return meds


_SPACY_CACHE: list = []  # holds [nlp] or [None]


def _load_spacy():
    if _SPACY_CACHE:
        return _SPACY_CACHE[0]
    nlp = None
    try:
        import spacy

        for model in ("en_core_sci_sm", "en_core_web_sm"):
            try:
                nlp = spacy.load(model)
                break
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        nlp = None
    _SPACY_CACHE.append(nlp)
    return nlp


def validate_against_drugbank(
    medications: list[dict], vocab: set[str], fuzzy_threshold: int = 88
) -> list[dict]:
    """Cross-validate medications against DrugBank; tag each as recognized.

    Returns the medication list with a boolean ``_recognized`` annotation; the
    caller decides whether to drop or flag unrecognized entries.
    """
    validated = []
    for med in medications:
        name = normalize_name(med.get("name", ""))
        recognized = bool(name) and _is_recognized(name, vocab, fuzzy_threshold)
        validated.append({**med, "_recognized": recognized})
    return validated


def build_chosen_summary(note: str, vocab: set[str]) -> dict:
    """Full pipeline for one note: extract, validate, format to output schema.

    Only DrugBank-recognized medications are kept in the chosen summary;
    unrecognized extractions are surfaced in confidence_flags so the chosen
    target itself never carries hallucinated drugs.
    """
    sections = extract_sections(note)
    meds = extract_medications_spacy(sections["medications_raw"])
    validated = validate_against_drugbank(meds, vocab)

    kept = []
    flags = []
    for med in validated:
        recognized = med.pop("_recognized")
        if recognized:
            kept.append(med)
        elif med.get("name"):
            flags.append(f"extraction: unverified medication '{med['name']}' dropped")

    return {
        "diagnoses": sections["diagnoses"],
        "medications": kept,
        "procedures": sections["procedures"],
        "discharge_instructions": sections["discharge_instructions"],
        "confidence_flags": flags,
    }


def process_dataset(
    input_path: str, output_path: str, sample_size: int = 10000
) -> int:
    """Extract chosen summaries over MIMIC-IV with deterministic sampling.

    Args:
        input_path: Path to a MIMIC-IV notes CSV (or directory containing one).
        output_path: Directory to write chosen_summaries.jsonl.
        sample_size: Max number of records to process.

    Returns:
        Number of chosen summaries written.
    """
    csv_path = input_path
    if os.path.isdir(input_path):
        candidates = [f for f in os.listdir(input_path) if f.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError(f"No CSV notes file found under {input_path}")
        csv_path = os.path.join(input_path, sorted(candidates)[0])

    df = pd.read_csv(csv_path)
    df = filter_discharge_summaries(df)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    text_col = next(
        (c for c in ("text", "note", "TEXT", "note_text") if c in df.columns), None
    )
    if text_col is None:
        raise ValueError("Could not find a note text column in the input CSV.")

    vocab = get_vocabulary()
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "chosen_summaries.jsonl")

    written = 0
    with open(out_file, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            note = row[text_col]
            summary = build_chosen_summary(note, vocab)
            # Skip empty extractions.
            if not (summary["diagnoses"] or summary["medications"]):
                continue
            record = {"note": str(note), "chosen": summary}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    """Entry point: parse arguments and run the extraction pipeline."""
    parser = argparse.ArgumentParser(description="MIMIC-IV extraction pipeline")
    parser.add_argument("--input", type=str, default=os.getenv("MIMIC_DATA_DIR"),
                        help="Path to MIMIC-IV data directory or CSV")
    parser.add_argument("--output", type=str, default=os.getenv("SPLITS_DIR", "data/splits"),
                        help="Output directory for JSONL splits")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Number of records to sample")
    args = parser.parse_args()
    if not args.input:
        parser.error("No input provided and MIMIC_DATA_DIR is unset.")
    count = process_dataset(args.input, args.output, args.sample)
    print(f"Wrote {count} chosen summaries to {args.output}")


if __name__ == "__main__":
    main()
