"""Adversarial rejection generation for ORPO preference pairs.

Implements 5 failure classes from the PRD as deterministic, schema-preserving
perturbations of a *chosen* (good) summary. Structural perturbation — rather
than free-form LLM generation — yields controllable, reproducible hard
negatives that remain valid JSON, which is exactly what ORPO needs to learn the
right preference boundary.

    A: Medication hallucination — invent medications not in the note
    B: Diagnosis omission — drop critical diagnoses
    C: Dosage mutation — alter dose, frequency, or route
    D: Timeline inversion — swap chronological ordering
    E: Contraindication reversal — reverse contraindication advice

Usage:
    python -m data.rejection_gen --input data/chosen_summaries.jsonl \
        --output data/preference_dataset.jsonl --batch_size 500
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random

from dotenv import load_dotenv

from api.inference import build_prompt

load_dotenv()

FAILURE_CLASSES = ["A", "B", "C", "D", "E"]
FAILURE_WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]  # Weighted sampling for "random" mode

# Fabricated drug names: plausibly clinical, deliberately NOT real DrugBank
# entries, so the guardrail/DEER metric would catch them.
_FAKE_DRUGS = [
    "Cardizopine", "Neurosil", "Hepatraxone", "Pulmocaine",
    "Renalox", "Glucovanze", "Thrombexa", "Osteonil",
]

# Replacement values for dosage mutation.
_DOSES = ["5 mg", "10 mg", "20 mg", "40 mg", "80 mg", "100 mg", "250 mg", "500 mg"]
_FREQS = ["once daily", "twice daily", "three times daily", "every 6 hours", "weekly"]
_ROUTES = ["oral", "IV", "subcutaneous", "intramuscular", "topical"]

# Contraindication reversals: each pair is bidirectionally swapped in text.
_REVERSALS = [
    ("do not", "make sure to"),
    ("avoid", "continue"),
    ("should not", "should"),
    ("must not", "must"),
    ("discontinue", "keep taking"),
    ("stop taking", "keep taking"),
    ("without", "with"),
]


def _rng(chosen: dict, salt: str) -> random.Random:
    """RNG seeded deterministically from the summary content and a salt.

    Uses a stable hash (SHA-256) rather than the builtin ``hash()``, which is
    salted per process (PYTHONHASHSEED) and would make dataset builds
    irreproducible across runs and Python versions.
    """
    payload = (json.dumps(chosen, sort_keys=True, default=str) + "\x00" + salt).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def generate_medication_hallucination(chosen: dict) -> dict:
    """Class A: invent medications not present in the note."""
    rej = copy.deepcopy(chosen)
    rng = _rng(chosen, "A")
    meds = rej.setdefault("medications", [])
    n_fake = rng.randint(1, 2)
    for name in rng.sample(_FAKE_DRUGS, n_fake):
        meds.append(
            {
                "name": name,
                "dose": rng.choice(_DOSES),
                "freq": rng.choice(_FREQS),
                "route": rng.choice(_ROUTES),
            }
        )
    return rej


def generate_diagnosis_omission(chosen: dict) -> dict:
    """Class B: drop one or more diagnoses (or instructions if none exist)."""
    rej = copy.deepcopy(chosen)
    diagnoses = rej.get("diagnoses") or []
    if diagnoses:
        rng = _rng(chosen, "B")
        # Drop at least one, up to half.
        n_drop = max(1, len(diagnoses) // 2)
        drop_idx = set(rng.sample(range(len(diagnoses)), n_drop))
        rej["diagnoses"] = [d for i, d in enumerate(diagnoses) if i not in drop_idx]
    else:
        # No diagnoses to omit: degrade by dropping discharge guidance instead.
        rej["discharge_instructions"] = ""
    return rej


def generate_dosage_mutation(chosen: dict) -> dict:
    """Class C: alter a medication's dose, frequency, or route."""
    rej = copy.deepcopy(chosen)
    meds = rej.get("medications") or []
    if not meds:
        # No meds: invent a wrong instruction about dosing.
        rej["discharge_instructions"] = (
            (rej.get("discharge_instructions") or "")
            + " Double all medication doses."
        ).strip()
        return rej
    rng = _rng(chosen, "C")
    idx = rng.randrange(len(meds))
    med = dict(meds[idx])
    field = rng.choice(["dose", "freq", "route"])
    pool = {"dose": _DOSES, "freq": _FREQS, "route": _ROUTES}[field]
    current = med.get(field, "")
    choices = [v for v in pool if v != current] or pool
    med[field] = rng.choice(choices)
    meds[idx] = med
    return rej


def generate_timeline_inversion(chosen: dict) -> dict:
    """Class D: invert chronological ordering of events/instructions."""
    rej = copy.deepcopy(chosen)
    text = rej.get("discharge_instructions") or ""
    if text:
        # Swap temporal connectives and reverse sentence order.
        swapped = text
        for a, b in [("before", "\0"), ("after", "before"), ("\0", "after")]:
            swapped = swapped.replace(a, b)
        sentences = [s.strip() for s in swapped.split(".") if s.strip()]
        rej["discharge_instructions"] = ". ".join(reversed(sentences))
        if rej["discharge_instructions"]:
            rej["discharge_instructions"] += "."
    procedures = rej.get("procedures") or []
    if len(procedures) > 1:
        rej["procedures"] = list(reversed(procedures))
    # Guarantee a visible change for ORPO: a single-sentence instruction with no
    # temporal connectives reverses to itself, so scramble word order instead.
    if rej == chosen and text:
        words = text.split()
        rej["discharge_instructions"] = " ".join(reversed(words)) if len(words) > 1 else (
            text + " Do this before being admitted."
        )
    elif rej == chosen and not text:
        rej["discharge_instructions"] = "Complete follow-up before the procedure was performed."
    return rej


def generate_contraindication_reversal(chosen: dict) -> dict:
    """Class E: reverse contraindication/safety advice."""
    rej = copy.deepcopy(chosen)
    text = rej.get("discharge_instructions") or ""
    if not text:
        rej["discharge_instructions"] = "Resume all activities immediately; no restrictions apply."
        return rej
    lowered = text
    replaced = False
    for phrase, opposite in _REVERSALS:
        if phrase in lowered.lower():
            # Case-insensitive replace of first occurrence.
            idx = lowered.lower().find(phrase)
            lowered = lowered[:idx] + opposite + lowered[idx + len(phrase):]
            replaced = True
    if not replaced:
        lowered = "Disregard previous precautions. " + lowered
    rej["discharge_instructions"] = lowered
    return rej


_GENERATORS = {
    "A": generate_medication_hallucination,
    "B": generate_diagnosis_omission,
    "C": generate_dosage_mutation,
    "D": generate_timeline_inversion,
    "E": generate_contraindication_reversal,
}


def generate_rejected(chosen: dict, failure_class: str = "random") -> dict:
    """Dispatch to a failure-class generator (or weighted-random selection).

    Guarantees the result differs from ``chosen`` — an ORPO pair where the
    rejected equals the chosen carries no preference signal. In random mode,
    if the sampled class no-ops on this input, other classes are tried in a
    deterministic order; the final fallback (medication hallucination) always
    appends, so it always differs.
    """
    if failure_class == "random":
        rng = _rng(chosen, "dispatch")
        first = rng.choices(FAILURE_CLASSES, weights=FAILURE_WEIGHTS, k=1)[0]
        order = [first] + [c for c in FAILURE_CLASSES if c != first]
        for cls in order:
            rejected = _GENERATORS[cls](chosen)
            if rejected != chosen:
                return rejected
        return generate_medication_hallucination(chosen)

    if failure_class not in _GENERATORS:
        raise ValueError(f"Unknown failure class: {failure_class!r}")
    rejected = _GENERATORS[failure_class](chosen)
    if rejected == chosen:
        # Requested class no-opped on this input; fall back to a guaranteed change.
        rejected = generate_medication_hallucination(chosen)
    return rejected


def _extract_note_and_summary(record: dict) -> tuple[str, dict]:
    """Pull (note, chosen_summary) from a flexible input record shape."""
    note = ""
    for key in ("note", "prompt", "text", "input"):
        if isinstance(record.get(key), str):
            note = record[key]
            break
    summary = record
    if "medications" not in record:
        for key in ("chosen", "summary", "chosen_summary"):
            if isinstance(record.get(key), dict):
                summary = record[key]
                break
    return note, summary


def build_preference_pairs(
    chosen_path: str,
    output_path: str,
    batch_size: int = 500,
    failure_class: str = "random",
) -> int:
    """Generate ORPO preference pairs from chosen summaries.

    Writes JSONL records with keys: prompt, chosen, rejected, failure_class —
    the shape TRL's ORPOTrainer consumes (chosen/rejected as text completions).

    Returns the number of pairs written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0
    with open(chosen_path, encoding="utf-8") as src, \
            open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line or written >= batch_size:
                if written >= batch_size:
                    break
                continue
            record = json.loads(line)
            note, summary = _extract_note_and_summary(record)
            rejected = generate_rejected(summary, failure_class)
            label = failure_class if failure_class != "random" else "mixed"
            pair = {
                "prompt": build_prompt(note),
                "chosen": json.dumps(summary, ensure_ascii=False),
                "rejected": json.dumps(rejected, ensure_ascii=False),
                "failure_class": label,
            }
            dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
    return written


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
    count = build_preference_pairs(
        args.input, args.output, args.batch_size, args.failure_class
    )
    print(f"Wrote {count} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
