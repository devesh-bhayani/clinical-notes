"""ORPO trainer for Clinical Note Summarizer.

Fine-tunes BioMistral-7B with QLoRA (r=32, alpha=64) and 4-bit NF4 quantization
using ORPO preference optimization via TRL ORPOTrainer.

Usage:
    python train/orpo_train.py --config configs/orpo_base.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinical_notes.train")

_ENV_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _resolve_env(value):
    """Recursively resolve ${VAR} references in strings within a config tree."""
    if isinstance(value, str):
        def repl(m):
            env = os.getenv(m.group(1))
            if env is None:
                raise KeyError(f"Environment variable {m.group(1)} is not set")
            return env
        return _ENV_VAR.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve environment variable references."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return _resolve_env(cfg)


def build_quantization_config():
    """Build the 4-bit NF4 quantization config for BioMistral."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: dict):
    """Build the QLoRA config (r=32, alpha=64) from YAML settings."""
    from peft import LoraConfig

    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora.get("lora_dropout", 0.05),
        target_modules=lora["target_modules"],
        task_type=lora.get("task_type", "CAUSAL_LM"),
        bias="none",
    )


def load_model_and_tokenizer(cfg: dict):
    """Load BioMistral-7B with quantization and prepare for k-bit training."""
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["name"]
    logger.info("Loading base model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_quantization_config(),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model, tokenizer


def load_data(cfg: dict):
    """Load ORPO preference pairs (prompt/chosen/rejected) from JSONL splits."""
    from datasets import load_dataset

    data_files = {"train": cfg["data"]["train_file"]}
    val_file = cfg["data"].get("val_file")
    if val_file and os.path.isfile(val_file):
        data_files["validation"] = val_file
    dataset = load_dataset("json", data_files=data_files)

    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset["train"].column_names)
    if missing:
        raise ValueError(f"Preference dataset missing required columns: {missing}")
    return dataset


def build_trainer(cfg: dict, model, tokenizer, dataset):
    """Construct the ORPOTrainer with config-driven hyperparameters."""
    from trl import ORPOConfig, ORPOTrainer

    t = cfg["training"]
    args = ORPOConfig(
        output_dir=cfg["output"]["output_dir"],
        logging_dir=cfg["output"].get("logging_dir", "logs"),
        beta=t["orpo_beta"],
        max_length=t["max_length"],
        max_prompt_length=t.get("max_prompt_length", t["max_length"] // 2),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.1),
        logging_steps=t.get("logging_steps", 10),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 200),
        eval_strategy=t.get("eval_strategy", "no") if "validation" in dataset else "no",
        eval_steps=t.get("eval_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),
        bf16=t.get("bf16", True),
        report_to=t.get("report_to", "none"),
        gradient_checkpointing=True,
    )

    trainer = ORPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        peft_config=build_lora_config(cfg),
    )
    return trainer


def train(cfg: dict) -> None:
    """Orchestrate ORPO training: model loading, data prep, ORPOTrainer run."""
    if cfg["training"].get("report_to") == "wandb" and os.getenv("WANDB_API_KEY"):
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "clinical-note-summarizer"),
            config=cfg,
        )

    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = load_data(cfg)
    trainer = build_trainer(cfg, model, tokenizer, dataset)

    resume = cfg.get("resume_from_checkpoint")
    logger.info("Starting ORPO training (resume=%s)", resume)
    trainer.train(resume_from_checkpoint=resume)

    output_dir = cfg["output"]["output_dir"]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved adapter and tokenizer to %s", output_dir)


def main() -> None:
    """Entry point: parse --config argument and launch training."""
    parser = argparse.ArgumentParser(description="ORPO trainer for Clinical Note Summarizer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["resume_from_checkpoint"] = args.resume
    train(cfg)


if __name__ == "__main__":
    main()
