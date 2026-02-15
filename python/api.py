"""FastAPI REST API for Wikipedia summarization."""

import asyncio
import logging
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import DEFAULT_TARGET_TOKENS, MIN_TARGET_TOKENS, MAX_TARGET_TOKENS
from summarizer import summarize_wiki
from model_loader import count_tokens

logger = logging.getLogger(__name__)
app = FastAPI(title="Wikipedia Summarizer")


def _validate_url(url: str) -> None:
    """Validate that a URL points to Wikipedia."""
    if not url:
        raise ValueError("Missing 'url'.")
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.hostname.endswith("wikipedia.org"):
        raise ValueError(f"Only Wikipedia URLs are supported: {url}")


def _validate_target_tokens(target_tokens) -> int:
    """Validate and return target_tokens as int."""
    if not isinstance(target_tokens, (int, float)) or not (MIN_TARGET_TOKENS <= target_tokens <= MAX_TARGET_TOKENS):
        raise ValueError(f"target_tokens must be between {MIN_TARGET_TOKENS} and {MAX_TARGET_TOKENS}.")
    return int(target_tokens)


async def _summarize_one(url: str, target_tokens: int) -> dict:
    """Summarize a single URL and return a result dict."""
    try:
        _validate_url(url)
        target_tokens = _validate_target_tokens(target_tokens)
    except ValueError as e:
        return {"url": url, "error": str(e)}

    try:
        summary = await asyncio.to_thread(summarize_wiki, url, target_tokens)
        return {
            "url": url,
            "summary": summary,
            "token_count": count_tokens(summary),
            "target_tokens": target_tokens,
        }
    except Exception as e:
        logger.exception("Error summarizing %s", url)
        return {"url": url, "error": str(e)}


@app.post("/summarize")
async def summarize_api(request: Request):
    """Summarize one or more Wikipedia articles.

    Accepts either a single object or an array:
      Single:  {"url": "...", "target_tokens": 150}
      Batch:   [{"url": "..."}, {"url": "...", "target_tokens": 200}]

    Returns:
      Single:  {"url": "...", "summary": "...", "token_count": N, "target_tokens": N}
      Batch:   [{"url": "...", "summary": "...", ...}, ...]
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    # Batch request: JSON array
    if isinstance(data, list):
        if not data:
            raise HTTPException(status_code=400, detail="Empty array.")

        tasks = []
        for item in data:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="Each item must be a JSON object with 'url'.")
            url = item.get("url", "")
            target_tokens = item.get("target_tokens", DEFAULT_TARGET_TOKENS)
            tasks.append(_summarize_one(url, target_tokens))

        results = await asyncio.gather(*tasks)
        return JSONResponse(list(results))

    # Single request: JSON object
    if isinstance(data, dict):
        url = data.get("url")
        target_tokens = data.get("target_tokens", DEFAULT_TARGET_TOKENS)

        try:
            _validate_url(url)
            target_tokens = _validate_target_tokens(target_tokens)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            summary = await asyncio.to_thread(summarize_wiki, url, target_tokens)
        except Exception as e:
            logger.exception("Error summarizing %s", url)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse({
            "url": url,
            "summary": summary,
            "token_count": count_tokens(summary),
            "target_tokens": target_tokens,
        })

    raise HTTPException(status_code=400, detail="Request body must be a JSON object or array.")
