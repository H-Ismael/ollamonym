"""Configuration management via environment variables."""

import os
from pathlib import Path

# Paths
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", "./templates"))
TEMPLATES_DIR.mkdir(exist_ok=True)

# Secrets (required)
PSEUDONYM_SECRET = os.getenv("PSEUDONYM_SECRET")
if not PSEUDONYM_SECRET:
    raise ValueError("PSEUDONYM_SECRET environment variable is required")

# Service configuration
DEFAULT_TEMPLATE_ID = os.getenv("DEFAULT_TEMPLATE_ID", "default-pii-v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_FALLBACK_URLS = [
    url.strip()
    for url in os.getenv("OLLAMA_FALLBACK_URLS", "").split(",")
    if url.strip()
]
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")

# Concurrency and chunking
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "1"))
CHUNKING_ENABLED = os.getenv("CHUNKING_ENABLED", "true").lower() == "true"
CHUNK_CHAR_TARGET = int(os.getenv("CHUNK_CHAR_TARGET", "8000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
CHUNK_MAX_PARALLEL = int(os.getenv("CHUNK_MAX_PARALLEL", "2"))
RULE_PREEXTRACT_ENABLED = os.getenv("RULE_PREEXTRACT_ENABLED", "true").lower() == "true"

# Logging
LOG_SENSITIVE = os.getenv("LOG_SENSITIVE", "false").lower() == "true"

# Token generation
TOKEN_ID_LEN = int(os.getenv("TOKEN_ID_LEN", "6"))

# Health check settings
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))
