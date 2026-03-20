# Model Card: Clinical Note Summarizer

## Model Details

- **Model Name:** Clinical Note Summarizer v1.0
- **Base Model:** BioMistral/BioMistral-7B
- **Fine-Tuning Method:** ORPO (Odds Ratio Preference Optimization)
- **Quantization:** 4-bit NF4 via BitsAndBytes
- **Adapter:** QLoRA (r=32, alpha=64)
- **Framework:** Transformers 4.44.0, TRL 0.9.6, PEFT 0.12.0

## Intended Use

Summarize unstructured clinical discharge notes into structured JSON containing:
- Diagnoses
- Medications (name, dose, frequency, route)
- Procedures
- Discharge instructions
- Confidence flags for uncertain extractions

**Not intended for:** Clinical deployment without regulatory approval, real-time EHR integration, or use as sole source of clinical decision-making.

## Training Data

- **Source:** MIMIC-IV discharge summaries (PhysioNet, requires DUA)
- **Sample Size:** 10,000 stratified records
- **Extraction:** Deterministic pipeline using regex + spaCy NER
- **Preference Pairs:** ORPO format with 5 rejection failure classes (A–E)
- **Validation:** DrugBank cross-validation for medication entities

## Evaluation Results

| Metric | Target | Actual |
|---|---|---|
| Drug Entity Error Rate | ≤ 2% | TBD |
| HHEM | ≥ 0.80 | TBD |
| BERTScore F1 | ≥ 0.88 | TBD |
| ROUGE-L | ≥ 0.42 | TBD |
| FactScore | ≥ 0.75 | TBD |
| GPT-4o Preference | ≥ 70% | TBD |
| Latency | ≤ 8 sec | TBD |

## Bias Analysis

Demographic bias audit conducted across:
- Age distribution
- Race/ethnicity
- Admission type

Oversampling applied to underrepresented groups where imbalance detected.

Results: TBD

## Limitations

- Trained on English-language MIMIC-IV discharge notes only
- Performance on non-discharge note types is unknown
- Medication validation limited to DrugBank vocabulary coverage
- Hallucination suppression is probabilistic, not guaranteed

## Ethical Considerations

- Model outputs must pass DrugBank guardrail validation before reaching users
- Confidence flags must be surfaced to end users for uncertain extractions
- Not a substitute for clinical judgment
- HIPAA compliance required for any deployment involving patient data
