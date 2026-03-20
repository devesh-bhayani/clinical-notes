# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands

```bash
# Train
python train/orpo_train.py --config configs/orpo_base.yaml

# Evaluate
python eval/run_eval_suite.py --split test

# Serve
uvicorn api.main:app --reload --port 8000
```

## Architecture

Two-stage fine-tuning pipeline on BioMistral-7B for clinical note summarization:

- **Quantization**: 4-bit NF4 via `BitsAndBytesConfig` (bitsandbytes==0.43.3)
- **Adapters**: QLoRA r=32, alpha=64 via PEFT (peft==0.12.0)
- **Alignment**: ORPO preference optimization via TRL `ORPOTrainer` (trl==0.9.6)
- **Serving**: FastAPI app at `api/main.py`, port 8000

### Key directory layout (expected)
- `train/orpo_train.py` — ORPO trainer entry point, reads `configs/orpo_base.yaml`
- `eval/run_eval_suite.py` — evaluation suite (Drug Entity Error Rate, HHEM, BERTScore)
- `api/main.py` — FastAPI inference server
- `configs/orpo_base.yaml` — hyperparameters and training config
- `/data/drugbank_vocabulary.csv` — DrugBank entity vocabulary for NER evaluation
- `/data/mimic/` — **HIPAA-regulated training data, never commit or log**

## Pinned Dependencies

```
transformers==4.44.0
peft==0.12.0
trl==0.9.6
bitsandbytes==0.43.3
```

## Data Rules

- **Never commit or log any file under `/data/mimic/`** — HIPAA-regulated patient data.
- All training data paths must be loaded from `.env`, never hardcoded in source.
- DrugBank vocab lives at `/data/drugbank_vocabulary.csv`.

## Output Schema

Every model output must be valid JSON conforming to:

```json
{
  "diagnoses": [],
  "medications": [{"name": "", "dose": "", "freq": "", "route": ""}],
  "procedures": [],
  "discharge_instructions": "",
  "confidence_flags": []
}
```

## Evaluation Targets

| Metric | Target |
|---|---|
| Drug Entity Error Rate | ≤ 2% |
| HHEM (hallucination) | ≥ 0.80 |
| BERTScore | ≥ 0.88 |
