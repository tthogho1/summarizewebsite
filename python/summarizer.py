"""Core summarization logic."""

import time
import logging
from transformers import GenerationConfig

from config import DEFAULT_TARGET_TOKENS, TOKEN_SAFETY_MARGIN
from model_loader import (
    tokenizer, model, is_seq2seq, is_t5_based,
    summarizer_pipeline, pipeline_type, count_tokens,
)
from scraper import fetch_page_text

logger = logging.getLogger(__name__)


def calculate_generation_params(target_tokens: int, model_max_length: int):
    """Convert target token count into safe (max_new, min_new, max_input) parameters."""
    max_new = target_tokens
    min_new = max(20, int(target_tokens * 0.7))
    reserved = max_new + TOKEN_SAFETY_MARGIN

    if reserved >= model_max_length:
        logger.warning("Target %d + margin >= model max %d. Reducing budget.", target_tokens, model_max_length)
        max_new = max(50, model_max_length - TOKEN_SAFETY_MARGIN - 100)
        min_new = max(20, int(max_new * 0.7))
        reserved = max_new + TOKEN_SAFETY_MARGIN

    max_input = max(100, model_max_length - reserved)
    logger.debug("Generation params: max_new=%d, min_new=%d, max_input=%d", max_new, min_new, max_input)
    return max_new, min_new, max_input


def _get_model_max() -> int:
    """Get the model's max context length with a sane fallback."""
    model_max = getattr(tokenizer, "model_max_length", 1024)
    if model_max is None or model_max <= 0 or model_max > 1_000_000:
        model_max = 1024
    return model_max


def summarize_text(text: str, target_tokens: int) -> str:
    """Summarize raw text (internal helper, no fetching)."""
    model_max = _get_model_max()
    max_new, min_new, max_input = calculate_generation_params(target_tokens, model_max)

    inputs = tokenizer(text, max_length=max_input, truncation=True, return_tensors="pt")
    truncated = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    if is_t5_based:
        truncated = f"summarize: {truncated}"

    input_tokens = len(inputs["input_ids"][0])
    logger.info("Input: %d tokens. Budget: %d-%d tokens", input_tokens, min_new, max_new)

    start = time.monotonic()
    summary = ""

    if is_seq2seq and model is not None:
        input_ids = tokenizer(truncated, return_tensors="pt", truncation=True, max_length=max_input)["input_ids"]
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=False,
            early_stopping=True,
        )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        gen_config = GenerationConfig(
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=False,
            early_stopping=is_seq2seq,
        )
        result = summarizer_pipeline(truncated, generation_config=gen_config, return_full_text=False)
        if isinstance(result, list) and result:
            summary = result[0].get("generated_text", "")

        # Strip leaked prompt for causal models
        if not is_seq2seq and summary.startswith(truncated[:100]):
            summary = summary[len(truncated):]

    summary = summary.strip()
    duration = time.monotonic() - start
    output_tokens = count_tokens(summary)

    if output_tokens < min_new:
        logger.warning("Short output: %d tokens < min %d", output_tokens, min_new)

    logger.info("Inference: %.2fs, output: %d tokens (target: %d)", duration, output_tokens, target_tokens)
    return summary


def summarize_wiki(url: str, target_tokens: int = None) -> str:
    """Summarize a Wikipedia page.

    Args:
        url: Wikipedia article URL.
        target_tokens: Desired summary length in tokens.

    Returns:
        Summary text, or error message.
    """
    if target_tokens is None:
        target_tokens = DEFAULT_TARGET_TOKENS

    logger.info("Summarize request: %s (target=%d)", url, target_tokens)
    start = time.monotonic()

    wiki_text = fetch_page_text(url)
    if not wiki_text:
        return "Could not fetch or parse the page."

    try:
        summary = summarize_text(wiki_text, target_tokens)
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        return f"Error during summarization: {e}"

    logger.info("Total: %.2fs, %d chars, %d tokens",
                time.monotonic() - start, len(summary), count_tokens(summary))
    return summary
