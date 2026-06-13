"""Fetch a clinical drug-name vocabulary from RxNorm (U.S. NLM, public domain).

A free, no-account alternative to the gated DrugBank Vocabulary download. Writes
a CSV in the exact format api/guardrail.py expects (a "Common name" column, plus
an empty "Synonyms" column), so it drops straight into DRUGBANK_VOCAB_PATH.

Source: RxNav `allconcepts` endpoint, restricted to clinical term types —
    IN  (ingredient), PIN (precise ingredient),
    BN  (brand name),  MIN (multi-ingredient) —
which yields real drug/brand names (e.g. "aspirin", "Lipitor") rather than the
raw chemical-structure strings in the full display-name list.

Usage:
    python scripts/fetch_drug_vocab.py --output data/drugbank_vocabulary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request

RXNAV_URL = "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN+PIN+BN+MIN"
_MIN_EXPECTED = 1000  # sanity floor; the live set is ~27k


def fetch_names(url: str = RXNAV_URL, timeout: int = 120) -> list[str]:
    """Fetch and de-duplicate clinical drug names from RxNorm."""
    req = urllib.request.Request(url, headers={"User-Agent": "clinical-notes-vocab/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    concepts = data.get("minConceptGroup", {}).get("minConcept", [])
    names = {(c.get("name") or "").strip() for c in concepts}
    names.discard("")
    return sorted(names)


def write_vocab(names: list[str], output: str) -> None:
    """Write names to a CSV in the guardrail's expected (Common name, Synonyms) shape."""
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Common name", "Synonyms"])
        for name in names:
            writer.writerow([name, ""])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a drug-name vocabulary from RxNorm (public domain)."
    )
    parser.add_argument(
        "--output", type=str, default="data/drugbank_vocabulary.csv",
        help="CSV path to write (point DRUGBANK_VOCAB_PATH here).",
    )
    parser.add_argument("--url", type=str, default=RXNAV_URL, help="RxNav allconcepts URL")
    args = parser.parse_args()

    try:
        names = fetch_names(args.url)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not fetch RxNorm vocabulary: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if len(names) < _MIN_EXPECTED:
        print(
            f"ERROR: fetched only {len(names)} names (< {_MIN_EXPECTED}); "
            "the endpoint may have changed. Aborting.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    write_vocab(names, args.output)
    print(f"Wrote {len(names)} drug names to {args.output}")


if __name__ == "__main__":
    main()
