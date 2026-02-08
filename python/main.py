import os
import time
import logging
import asyncio
import requests
from bs4 import BeautifulSoup
import gradio as gr
from transformers import pipeline, AutoTokenizer, GenerationConfig, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging: allow override with LOG_LEVEL env var
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

# Get model name from environment variable with default fallback
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/bart-large-cnn")
logger.info("Downloading and loading model: %s (this may take a while on first run...)", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summarizer = pipeline("text-generation", model=MODEL_NAME)
logger.info("Model and tokenizer loaded successfully: %s", MODEL_NAME)
# Configurable generation budget: how many new tokens to allow (and a small safety margin)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "130"))
TOKEN_SAFETY_MARGIN = int(os.environ.get("TOKEN_SAFETY_MARGIN", "5"))
logger.info("MAX_NEW_TOKENS=%d TOKEN_SAFETY_MARGIN=%d", MAX_NEW_TOKENS, TOKEN_SAFETY_MARGIN)

# Dedicated summarization model to try when generic pipeline/model returns empty or very short output
SUMMARIZATION_MODEL = os.environ.get("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
logger.info("SUMMARIZATION_MODEL=%s", SUMMARIZATION_MODEL)

# Load model object for direct generation fallback (seq2seq vs causal)
try:
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    if getattr(cfg, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        logger.info("Loaded seq2seq model for fallback generation: %s", MODEL_NAME)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        logger.info("Loaded causal LM model for fallback generation: %s", MODEL_NAME)
except Exception as e_load:
    model = None
    logger.debug("Could not load model for direct generate fallback: %s", e_load)


def fetch_page_text(page_url: str) -> str:
    if not page_url or not (page_url.startswith("http://") or page_url.startswith("https://")):
        logger.warning("Invalid or missing URL passed to fetch_page_text: %r", page_url)
        return ""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SummarizeWikiBot/1.0)"}
    start = time.monotonic()
    try:
        logger.info("Fetching page: %s", page_url)
        resp = requests.get(page_url, timeout=10, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        logger.exception("Error fetching %s: %s", page_url, e)
        return ""
    duration = time.monotonic() - start
    logger.info("Fetched %s in %.2fs (status=%s, bytes=%d)", page_url, duration, resp.status_code, len(resp.content or b""))
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find(id="mw-content-text") or soup
    paragraphs = [p.get_text(strip=True) for p in content.find_all("p") if p.get_text(strip=True)]
    text = "\n\n".join(paragraphs)
    logger.debug("Extracted %d paragraphs, total length=%d", len(paragraphs), len(text))
    return text



def count_tokens(text: str) -> int:
    # Return 0 for empty/None input to avoid errors
    if not text:
        return 0
    # Use the model tokenizer for accurate token counts when available
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(ids)
    except Exception:
        # fallback to whitespace split
        return len(text.split())

def summarize_wiki(url: str) -> str:
    logger.info("Summarize request started for URL: %s", url)
    start = time.monotonic()
    wiki_text = fetch_page_text(url)
    if not wiki_text:
        logger.warning("No text extracted for URL %s", url)
        return "Could not fetch or parse the page."
    # Determine safe truncation length: respect tokenizer/model maximum and reserve space
    model_max = getattr(tokenizer, "model_max_length", None)
    requested = 1024
    max_new = MAX_NEW_TOKENS
    if model_max and model_max > 0:
        reserved = max_new + TOKEN_SAFETY_MARGIN
        if reserved >= model_max:
            logger.warning(
                "Model max (%d) <= reserved tokens (%d). Adjusting max_new_tokens to fit.", model_max, reserved
            )
            # Reduce generation budget to leave a tiny room for the input and safety margin.
            max_new = max(1, model_max - TOKEN_SAFETY_MARGIN - 1)
            reserved = max_new + TOKEN_SAFETY_MARGIN
        trunc_len = min(requested, max(1, model_max - reserved))
        logger.debug("model_max=%s max_new=%d reserved=%d trunc_len=%d", str(model_max), max_new, reserved, trunc_len)
    else:
        trunc_len = requested
    # Truncate wiki_text to trunc_len tokens using model tokenizer
    inputs = tokenizer(wiki_text, max_length=trunc_len, truncation=True, return_tensors="pt")
    truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    logger.debug("Passing text to model (chars=%d, tokens=%d, model_max=%s)", len(truncated_text), len(inputs["input_ids"][0]), str(model_max))
    send_start = time.monotonic()
    try:
        logger.info("Running local inference with model=%s (max_new_tokens=%d)", MODEL_NAME, max_new)
        # Use max_new_tokens to control generated tokens precisely (transformers v5)
        # Build a GenerationConfig to avoid deprecated mixing of parameters
        min_len = min(30, max_new)
        gen_config = GenerationConfig(max_new_tokens=max_new, do_sample=False, min_length=min_len)
        summary = summarizer(
            truncated_text,
            generation_config=gen_config,
            return_full_text=False,
        )
        logger.info("Pipeline raw output: %s", summary)
        # Use the correct output key for the pipeline
        summary_raw = summary[0].get("generated_text") or summary[0].get("summary_text") or summary[0].get("text")
        # Ensure we have a string to avoid tokenizer errors
        summary_raw = summary_raw or ""
        # Some pipelines/models may return prompt+generation; try to strip the prompt by token ids
        try:
            prompt_len = len(inputs["input_ids"][0])
            summary_ids = tokenizer(summary_raw, add_special_tokens=False)["input_ids"]
            if len(summary_ids) > prompt_len:
                gen_ids = summary_ids[prompt_len:]
                summary_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                logger.debug("Stripped prompt from model output (prompt_tokens=%d, total_tokens=%d, gen_tokens=%d)", prompt_len, len(summary_ids), len(gen_ids))
            else:
                summary_text = summary_raw
        except Exception as e_strip:
            logger.debug("Failed to strip prompt from output: %s", e_strip)
            summary_text = summary_raw
        # If after processing we still have no text, include more debug info
        if not summary_text:
            logger.warning("Model returned empty summary. summary_raw repr=%r; pipeline_output=%s", summary_raw, summary)
            # Fallback: try direct model.generate() if we have a model loaded
            if model is not None:
                try:
                    logger.info("Attempting fallback generation via model.generate() (max_new=%d)", max_new)
                    gen_inputs = tokenizer(truncated_text, return_tensors="pt", truncation=True)
                    gen_ids = model.generate(**gen_inputs, max_new_tokens=max_new)
                    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    # For causal models the returned text may include the prompt; strip it if present
                    if gen_text.startswith(truncated_text.strip()):
                        fallback_text = gen_text[len(truncated_text):].strip()
                    else:
                        fallback_text = gen_text
                    if fallback_text:
                        summary_text = fallback_text
                        logger.info("Fallback generation succeeded (chars=%d tokens=%d)", len(summary_text), count_tokens(summary_text))
                    else:
                        logger.warning("Fallback generation returned empty text")
                except Exception as e_fb:
                    logger.exception("Fallback generation failed: %s", e_fb)
            # If fallback generate is empty or too short, try a dedicated summarization model with an instruction
            if not summary_text or count_tokens(summary_text) < 10:
                try:
                    logger.info("Attempting dedicated summarization model %s", SUMMARIZATION_MODEL)
                    # prefer text2text-generation for seq2seq summarizers
                    summarizer2 = pipeline("text2text-generation", model=SUMMARIZATION_MODEL)
                    instr = "Summarize the following article in 4-6 sentences:\n\n"
                    attempt_input = instr + truncated_text
                    gen_cfg = GenerationConfig(max_new_tokens=max_new, do_sample=False, min_length=min(30, max_new))
                    out = summarizer2(attempt_input, generation_config=gen_cfg)
                    logger.info("Dedicated summarizer raw output: %s", out)
                    out_text = out[0].get("generated_text") or out[0].get("summary_text") or out[0].get("text") or ""
                    summary_text = out_text or summary_text
                except Exception as e_ds:
                    logger.exception("Dedicated summarizer failed: %s", e_ds)
    except Exception as e:
        logger.exception("Inference failed for %s: %s", url, e)
        return "Error during summarization: %s" % str(e)
    send_duration = time.monotonic() - send_start
    total_duration = time.monotonic() - start
    token_count = count_tokens(summary_text)
    logger.info("Inference completed in %.2fs (model inference %.2fs). Summary length=%d, Token count=%d", total_duration, send_duration, len(summary_text), token_count)
    print(f"Token count in summary: {token_count}")
    return summary_text

# --- FastAPI REST API ---
app = FastAPI()


@app.post("/summarize")
async def summarize_api(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body.")
    try:
        logger.info("Received /summarize request for %s", url)
        # run blocking work in a thread to avoid blocking the ASGI event loop
        summary = await asyncio.to_thread(summarize_wiki, url)
    except Exception as e:
        logger.exception("Error summarizing %s", url)
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"summary": summary})


# --- Gradio UI ---
def launch_gradio():
    demo = gr.Interface(fn=summarize_wiki, inputs="text", outputs="text")
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)


if __name__ == "__main__":
    # Run Gradio in a separate thread so FastAPI can run on 7860
    threading.Thread(target=launch_gradio, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=7860)
