# Runbook: first real training run

Step-by-step to go from a fresh GPU box to a trained checkpoint with real
evaluation numbers. No external accounts required (data and drug vocabulary are
both public / no-DUA).

Each step has a **✓ check** — confirm it before moving on. If a step fails, see
[Troubleshooting](#troubleshooting) and, for the smoke run, send the traceback.

---

## 0. Prerequisites

- A Linux box with an NVIDIA GPU (~12 GB+ VRAM) and recent CUDA drivers.
- Python 3.10–3.12.
- This repo at `master`.

```bash
git clone https://github.com/devesh-bhayani/clinical-notes.git
cd clinical-notes        # or: git checkout master && git pull
```
✓ `ls configs/` shows `orpo_base.yaml` and `orpo_smoke.yaml`.

## 1. Environment + dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
✓ `pip show trl peft bitsandbytes` all report versions.

## 2. GPU preflight

```bash
python train/vram_test.py
```
✓ Prints `PASS: model loads in 4-bit.`
✗ If it fails → [Troubleshooting → CUDA / bitsandbytes](#cuda--bitsandbytes).

## 3. Drug vocabulary + `.env`

```bash
python scripts/fetch_drug_vocab.py --output data/drugbank_vocabulary.csv
cp .env.example .env
```
Edit `.env` so these are set (leave the rest blank):
```
DRUGBANK_VOCAB_PATH=data/drugbank_vocabulary.csv
SPLITS_DIR=data/splits
BASE_MODEL_NAME=BioMistral/BioMistral-7B
```
✓ `python -c "from api.guardrail import get_vocabulary; print(len(get_vocabulary()))"`
prints ~27000.

## 4. Smoke run — validate the whole loop (~10 min)

This is the gate. It builds ~500 pairs → trains 20 steps → saves → evaluates.

```bash
make train-smoke
```
✓ Finishes with no traceback and prints an eval gates JSON.
✗ **If it errors, stop and send the full traceback.** Do not skip to step 5.

(Manual equivalent, if you prefer:)
```bash
python -m data.asclepius --output data/splits --limit 500 --make-splits
python train/orpo_train.py --config configs/orpo_smoke.yaml
SPLITS_DIR=data/splits CHECKPOINT_DIR=models/smoke \
  python eval/run_eval_suite.py --split test --smoke
```

## 5. Real run

```bash
python -m data.asclepius --output data/splits --limit 20000 --make-splits
python train/orpo_train.py --config configs/orpo_base.yaml
python eval/run_eval_suite.py --split test
```
✓ `eval/results/latest.json` contains real metric values (not `TBD`).
Training time scales with `--limit` and `num_train_epochs`; expect hours.

> Optional: enable the two currently-skipped eval gates by `pip install bert-score`
> (BERTScore) and setting `OPENAI_API_KEY` (GPT-4o preference).

## 6. Serve and sanity-check (optional)

```bash
CHECKPOINT_DIR=models/checkpoints uvicorn api.main:app --port 8000
# in another shell:
curl -X POST localhost:8000/summarize -H "Content-Type: application/json" \
  -d '{"note":"Discharge Diagnosis: Acute appendicitis. Discharge Medications: Aspirin 81 mg PO daily."}'
```
✓ Returns structured JSON with `diagnoses`, `medications`, `confidence_flags`.

---

## Troubleshooting

### CUDA / bitsandbytes
`python train/vram_test.py` fails to load in 4-bit.
- Confirm the GPU is visible: `nvidia-smi`.
- bitsandbytes is pinned to `0.43.3`; if it complains about the CUDA version,
  reinstall it matched to your CUDA toolkit, then re-run the preflight.

### Out of memory (OOM) during training
- The smoke config already uses batch size 1 / `max_length` 1024.
- For the real run, lower `per_device_train_batch_size` or `max_length` in
  `configs/orpo_base.yaml`, or raise `gradient_accumulation_steps`.

### Dataset download (`data.asclepius`) fails
- First run needs network to Hugging Face. If behind a proxy, set `HF_HOME` and
  standard `HTTPS_PROXY`.
- Verify with: `python -c "from datasets import load_dataset; print('ok')"`.

### Vocabulary fetch fails
- `scripts/fetch_drug_vocab.py` needs network to `rxnav.nlm.nih.gov`.
- As a stopgap you can point `DRUGBANK_VOCAB_PATH` at the committed sample
  (`data/drugbank_vocabulary.sample.csv`) to exercise the pipeline, but the DEER
  metric will only recognize that small set.

---

## What to send back

- If the **smoke run** fails: the full traceback.
- After the **real run**: the contents of `eval/results/latest.json` — those are
  the numbers that fill in the model card's TBDs.
