"""Configuration constants and model mappings."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "esg_platform.log"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
COLLECTION_NAME = "esg_factors"
EVIDENCE_COLLECTION_NAME = "esg_evidence"  # Evidence Vault & Agentic Auditor use this
EVIDENCE_STORE_DIR = BASE_DIR / "evidence_store"
EVIDENCE_FILES_DIR = EVIDENCE_STORE_DIR / "files"
CALCULATORS_DIR = BASE_DIR / "calculator_modules"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding models (name -> LangChain class or model name)
EMBEDDING_MODELS = {
    "Local (all-MiniLM-L6-v2)": "sentence-transformers/all-MiniLM-L6-v2",
    "Local (BAAI/bge-small-en)": "BAAI/bge-small-en",
    "OpenAI (text-embedding-ada-002)": "openai",
}

# LLM models (provider -> model name or identifier)
LLM_PROVIDERS = {
    "OpenAI GPT-3.5 Turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
    "OpenAI GPT-4": {"provider": "openai", "model": "gpt-4"},
    "Google Gemini Pro": {"provider": "google", "model": "gemini-pro"},
    "Google Gemini 1.5 Pro": {"provider": "google", "model": "gemini-1.5-pro"},
    "Groq Llama3 70B": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Groq Mixtral 8x7B": {"provider": "groq", "model": "mixtral-8x7b-32768"},
    "Ollama (llama3)": {"provider": "ollama", "model": "llama3"},
    "Ollama (mistral)": {"provider": "ollama", "model": "mistral"},
}
