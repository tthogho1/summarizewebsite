"""Fetch and extract text from Wikipedia pages."""

import time
import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Reusable session for connection pooling
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (compatible; SummarizeWikiBot/1.0)"


def fetch_page_text(page_url: str) -> str:
    """Fetch and extract paragraph text from a Wikipedia page.

    Args:
        page_url: Full URL to a Wikipedia article.

    Returns:
        Extracted plain text, or empty string on failure.
    """
    if not page_url or not page_url.startswith(("http://", "https://")):
        logger.warning("Invalid URL: %r", page_url)
        return ""

    start = time.monotonic()
    try:
        logger.info("Fetching: %s", page_url)
        resp = _session.get(page_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.exception("Error fetching %s: %s", page_url, e)
        return ""

    logger.info("Fetched in %.2fs (%d bytes)", time.monotonic() - start, len(resp.content or b""))

    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find(id="mw-content-text") or soup
    paragraphs = [p.get_text(strip=True) for p in content.find_all("p") if p.get_text(strip=True)]
    text = "\n\n".join(paragraphs)
    logger.debug("Extracted %d paragraphs, %d chars", len(paragraphs), len(text))
    return text
