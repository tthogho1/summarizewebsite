FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py model_loader.py scraper.py summarizer.py api.py ui.py main.py ./

# Hugging Face Spaces expects port 7860
ENV API_PORT=7860
ENV DEVICE=cpu

# Pre-download model during build so startup is fast
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
AutoTokenizer.from_pretrained('facebook/bart-large-cnn'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')"

EXPOSE 7860

CMD ["python", "main.py"]
