# Clinical Note Summarizer — developer task runner.
#
# Lightweight targets (test, eval-smoke, synthetic, serve, ui) work with only
# requirements-dev.txt installed — no GPU or training stack required.

PYTHON ?= python
SPLITS_DIR ?= data/splits
SAMPLE_VOCAB := data/drugbank_vocabulary.sample.csv

# Env that lets the API/eval run without a checkpoint, MIMIC, or network.
STUB_ENV := ALLOW_STUB_INFERENCE=1 \
	DRUGBANK_VOCAB_PATH=$(SAMPLE_VOCAB) \
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.PHONY: help install install-dev test test-cov synthetic eval-smoke \
	compliance serve ui train clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the full runtime (GPU/training stack).
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install the lightweight test/CI dependencies only.
	$(PYTHON) -m pip install -r requirements-dev.txt

test:  ## Run the full unit + integration test suite.
	$(PYTHON) -m pytest -q

test-cov:  ## Run tests with coverage and enforce the guardrail >=90% gate.
	$(PYTHON) -m pytest --cov=api --cov=eval --cov=data --cov-report=term-missing
	$(PYTHON) -m coverage report --include="api/guardrail.py" --fail-under=90

synthetic:  ## Generate the synthetic PHI-free dataset under $(SPLITS_DIR).
	$(STUB_ENV) SPLITS_DIR=$(SPLITS_DIR) \
		$(PYTHON) scripts/make_synthetic_data.py --output $(SPLITS_DIR)

asclepius:  ## Build real ORPO splits from the public Asclepius dataset (needs `datasets`).
	DRUGBANK_VOCAB_PATH=$${DRUGBANK_VOCAB_PATH:-$(SAMPLE_VOCAB)} SPLITS_DIR=$(SPLITS_DIR) \
		$(PYTHON) -m data.asclepius --output $(SPLITS_DIR) --limit $${LIMIT:-20000} --make-splits

eval-smoke: synthetic  ## Build synthetic data, then run the eval suite on it.
	$(STUB_ENV) SPLITS_DIR=$(SPLITS_DIR) \
		$(PYTHON) eval/run_eval_suite.py --split test --smoke

compliance:  ## Run the HIPAA PHI check across tracked data/source files.
	@# Exclude the compliance tooling and Claude docs, which contain the pattern
	@# strings (and "mimic" in the checker's filename) as documentation.
	@git ls-files '*.py' '*.csv' '*.md' '*.yaml' '*.yml' \
		| grep -vE '^\.claude/|^scripts/check_no_mimic_data\.py$$' \
		| while read -r f; do \
			$(PYTHON) scripts/check_no_mimic_data.py "$$f" || exit 1; \
		done
	@echo "HIPAA compliance check passed."

serve:  ## Serve the API with the stub backend (no GPU required).
	$(STUB_ENV) $(PYTHON) -m uvicorn api.main:app --reload --port 8000

ui:  ## Launch the Streamlit demo UI.
	$(PYTHON) -m streamlit run ui/streamlit_app.py

train:  ## Launch ORPO training (requires GPU + requirements.txt).
	$(PYTHON) train/orpo_train.py --config configs/orpo_base.yaml

train-smoke:  ## ~10-min end-to-end train->eval validation on a few hundred pairs (GPU).
	DRUGBANK_VOCAB_PATH=$${DRUGBANK_VOCAB_PATH:-$(SAMPLE_VOCAB)} SPLITS_DIR=$(SPLITS_DIR) \
		$(PYTHON) -m data.asclepius --output $(SPLITS_DIR) --limit 500 --make-splits
	SPLITS_DIR=$(SPLITS_DIR) $(PYTHON) train/orpo_train.py --config configs/orpo_smoke.yaml
	SPLITS_DIR=$(SPLITS_DIR) CHECKPOINT_DIR=models/smoke \
		$(PYTHON) eval/run_eval_suite.py --split test --smoke

clean:  ## Remove caches and generated synthetic artifacts.
	-rm -rf .pytest_cache
	-find . -type d -name __pycache__ -prune -exec rm -rf {} +
	-rm -f $(SPLITS_DIR)/synthetic_notes.csv $(SPLITS_DIR)/chosen_summaries.jsonl \
		$(SPLITS_DIR)/train_orpo.jsonl $(SPLITS_DIR)/test.jsonl \
		$(SPLITS_DIR)/test_predictions.jsonl eval/results/latest.json
