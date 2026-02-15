"""Load model and tokenizer once, expose them for reuse."""

import logging
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM,
    AutoModelForCausalLM, pipeline as hf_pipeline,
)
from config import MODEL_NAME, DEVICE

logger = logging.getLogger(__name__)

logger.info("Loading model: %s on %s (may take a while on first run)...", MODEL_NAME, DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
cfg = AutoConfig.from_pretrained(MODEL_NAME)
is_seq2seq = getattr(cfg, "is_encoder_decoder", False)

model_arch = cfg.architectures[0] if hasattr(cfg, "architectures") and cfg.architectures else ""
is_t5_based = "T5" in model_arch or "t5" in MODEL_NAME.lower()

# Load model once
try:
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    logger.info("Loaded %s model on %s", "seq2seq" if is_seq2seq else "causal LM", DEVICE)
except Exception as e:
    model = None
    logger.warning("Could not load model directly: %s", e)

# Reuse model in pipeline (no duplicate load)
pipeline_type = "text-generation"
summarizer_pipeline = hf_pipeline(
    pipeline_type,
    model=model if model is not None else MODEL_NAME,
    tokenizer=tokenizer,
    device=DEVICE,
)
logger.info("Pipeline ready: %s", pipeline_type)


def count_tokens(text: str) -> int:
    """Count tokens using the model's tokenizer."""
    if not text:
        return 0
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        logger.warning("Tokenizer failed, falling back to whitespace split")
        return len(text.split())
