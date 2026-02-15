"""Centralized configuration loaded from environment variables."""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d - %(message)s",
)

# Model
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/bart-large-cnn")
DEVICE = os.environ.get("DEVICE", "cpu")

# Generation
DEFAULT_TARGET_TOKENS = int(os.environ.get("DEFAULT_TARGET_TOKENS", "150"))
TOKEN_SAFETY_MARGIN = int(os.environ.get("TOKEN_SAFETY_MARGIN", "10"))
MAX_TARGET_TOKENS = int(os.environ.get("MAX_TARGET_TOKENS", "500"))
MIN_TARGET_TOKENS = int(os.environ.get("MIN_TARGET_TOKENS", "50"))

# Server
API_PORT = int(os.environ.get("API_PORT", "7860"))
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7861"))
