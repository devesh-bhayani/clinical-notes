"""GPU memory verification for Clinical Note Summarizer training.

Loads the base model in 4-bit quantization to verify VRAM headroom before
launching a full training run.

Usage:
    python train/vram_test.py
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinical_notes.vram")

# BioMistral-7B in 4-bit needs roughly 5-6 GB of weights plus activation/optimizer
# headroom for QLoRA training; require a comfortable floor.
MIN_RECOMMENDED_VRAM_MB = 12_000


def check_gpu_available() -> bool:
    """Return True if a CUDA GPU is available."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not import torch / query CUDA: %s", exc)
        return False


def estimate_vram_usage() -> dict:
    """Report total, used, and free VRAM in MB for the current device."""
    import torch

    if not torch.cuda.is_available():
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0}
    free, total = torch.cuda.mem_get_info()
    return {
        "total_mb": total // (1024 * 1024),
        "used_mb": (total - free) // (1024 * 1024),
        "free_mb": free // (1024 * 1024),
    }


def run_load_test() -> bool:
    """Attempt to load BioMistral-7B in 4-bit. Return True if successful."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_name = os.getenv("BASE_MODEL_NAME", "BioMistral/BioMistral-7B")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant, device_map="auto"
        )
        logger.info("Loaded %s in 4-bit. Post-load VRAM: %s", model_name,
                    estimate_vram_usage())
        del model
        torch.cuda.empty_cache()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("4-bit load failed: %s", exc)
        return False


def main() -> None:
    """Run GPU verification checks and print results."""
    if not check_gpu_available():
        print("FAIL: no CUDA GPU available.")
        raise SystemExit(1)

    vram = estimate_vram_usage()
    print(f"VRAM: total={vram['total_mb']}MB free={vram['free_mb']}MB")
    if vram["total_mb"] < MIN_RECOMMENDED_VRAM_MB:
        print(
            f"WARNING: {vram['total_mb']}MB total is below the recommended "
            f"{MIN_RECOMMENDED_VRAM_MB}MB for QLoRA training."
        )

    ok = run_load_test()
    print("PASS: model loads in 4-bit." if ok else "FAIL: model failed to load.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
