"""Inference backends for the Clinical Note Summarizer.

Two interchangeable summarizers implement the same protocol:

- ``BioMistralSummarizer``: the production path. Loads BioMistral-7B in 4-bit
  with the trained QLoRA/ORPO adapter and generates structured JSON.
- ``StubSummarizer``: a deterministic, GPU-free backend used for tests, local
  development, and demos. It produces schema-valid output without a model.

Selection happens in ``api.main`` based on environment configuration so a
clinical deployment never silently serves stub output: the stub is only used
when ``ALLOW_STUB_INFERENCE`` is explicitly enabled.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol, runtime_checkable

# The canonical empty output, used as a fallback when parsing fails.
EMPTY_OUTPUT: dict = {
    "diagnoses": [],
    "medications": [],
    "procedures": [],
    "discharge_instructions": "",
    "confidence_flags": [],
}

_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Read the clinical note and "
    "produce a structured discharge summary as a single JSON object with keys: "
    "diagnoses (list of strings), medications (list of objects with name, dose, "
    "freq, route), procedures (list of strings), discharge_instructions (string), "
    "and confidence_flags (list of strings). Only include information present in "
    "the note. Respond with JSON only."
)


@runtime_checkable
class Summarizer(Protocol):
    """Contract every inference backend must satisfy."""

    def summarize(self, note: str) -> dict:
        """Return a dict conforming to the output schema for one clinical note."""
        ...


def build_prompt(note: str) -> str:
    """Build the instruction prompt for a single clinical note."""
    return f"<s>[INST] {_SYSTEM_PROMPT}\n\nCLINICAL NOTE:\n{note} [/INST]"


def coerce_to_schema(obj: object) -> dict:
    """Coerce an arbitrary parsed object into the strict output schema.

    Missing keys are filled with empty defaults; medication entries are
    normalized to the four-field shape. Never raises.
    """
    result = {k: (list(v) if isinstance(v, list) else v) for k, v in EMPTY_OUTPUT.items()}
    if not isinstance(obj, dict):
        return result

    for key in ("diagnoses", "procedures", "confidence_flags"):
        val = obj.get(key)
        if isinstance(val, list):
            result[key] = [str(x) for x in val]

    instructions = obj.get("discharge_instructions")
    if isinstance(instructions, str):
        result["discharge_instructions"] = instructions

    meds = obj.get("medications")
    if isinstance(meds, list):
        normalized = []
        for med in meds:
            if isinstance(med, dict):
                normalized.append(
                    {
                        "name": str(med.get("name", "")),
                        "dose": str(med.get("dose", "")),
                        "freq": str(med.get("freq", "")),
                        "route": str(med.get("route", "")),
                    }
                )
            elif isinstance(med, str):
                normalized.append({"name": med, "dose": "", "freq": "", "route": ""})
        result["medications"] = normalized

    return result


def extract_json(text: str) -> dict:
    """Extract and parse the first balanced JSON object from model output.

    Returns the empty schema (with a confidence flag) if no valid JSON is found.
    """
    if not isinstance(text, str) or not text.strip():
        out = coerce_to_schema(None)
        out["confidence_flags"].append("inference: model produced no output")
        return out

    # Find the first balanced top-level object.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return coerce_to_schema(json.loads(candidate))
                    except json.JSONDecodeError:
                        break  # try the next "{"
        start = text.find("{", start + 1)

    out = coerce_to_schema(None)
    out["confidence_flags"].append("inference: could not parse JSON from model output")
    return out


class StubSummarizer:
    """Deterministic, GPU-free summarizer for tests, dev, and demos.

    Produces schema-valid output derived from light heuristics over the note
    text. It is intentionally not a real model — it exists so the API surface
    can be exercised end to end without weights.
    """

    # A few recognizable medications so the guardrail has something to validate.
    _KNOWN_MEDS = ("aspirin", "metformin", "warfarin", "atorvastatin", "ibuprofen")

    def summarize(self, note: str) -> dict:
        text = note if isinstance(note, str) else ""
        lowered = text.lower()

        medications = [
            {"name": name.capitalize(), "dose": "", "freq": "", "route": "oral"}
            for name in self._KNOWN_MEDS
            if name in lowered
        ]

        # Pull a crude "diagnosis" cue if present.
        diagnoses = []
        for cue in ("diagnosis:", "impression:", "assessment:"):
            idx = lowered.find(cue)
            if idx != -1:
                snippet = text[idx + len(cue) : idx + len(cue) + 120].strip()
                snippet = snippet.splitlines()[0] if snippet else ""
                if snippet:
                    diagnoses.append(snippet)
                break

        out = coerce_to_schema(
            {
                "diagnoses": diagnoses,
                "medications": medications,
                "procedures": [],
                "discharge_instructions": (
                    "Follow up with your primary care provider. Continue "
                    "medications as prescribed."
                ),
                "confidence_flags": ["stub_inference: not generated by a trained model"],
            }
        )
        return out


class BioMistralSummarizer:
    """Production summarizer: BioMistral-7B + QLoRA/ORPO adapter in 4-bit.

    The model and tokenizer are loaded lazily on first construction. Heavy
    imports (torch, transformers, peft) are deferred so importing this module
    has no GPU/ML dependency.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        base_model: str | None = None,
        max_new_tokens: int = 768,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.max_new_tokens = max_new_tokens
        base = base_model or os.getenv("BASE_MODEL_NAME", "BioMistral/BioMistral-7B")

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=quant, device_map="auto"
        )
        # Attach the trained adapter.
        self.model = PeftModel.from_pretrained(model, checkpoint_dir)
        self.model.eval()
        self._torch = torch

    def summarize(self, note: str) -> dict:
        prompt = build_prompt(note)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy; temperature is intentionally unset
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Decode only the newly generated portion.
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return extract_json(text)
