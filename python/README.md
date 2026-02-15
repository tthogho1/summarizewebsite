---
title: Wikipedia Summarizer
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Wikipedia Summarizer

Summarize Wikipedia articles using `facebook/bart-large-cnn`.

## Usage

### Web UI

Visit the root URL to use the Gradio interface.

### REST API

**Single URL:**
```bash
curl -X POST https://YOUR-SPACE.hf.space/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "target_tokens": 150}'
```

**Batch (multiple URLs):**
```bash
curl -X POST https://YOUR-SPACE.hf.space/summarize \
  -H "Content-Type: application/json" \
  -d '[
    {"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"},
    {"url": "https://en.wikipedia.org/wiki/Java_(programming_language)", "target_tokens": 200}
  ]'
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `facebook/bart-large-cnn` | Hugging Face model name |
| `DEVICE` | `cpu` | Device for inference |
| `DEFAULT_TARGET_TOKENS` | `150` | Default summary length |
| `API_PORT` | `7860` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
