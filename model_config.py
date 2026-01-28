"""
Centralized model configuration for encoder and decoder models.

This file defines the default model names used throughout the Latent Search framework.
Users can override these via command-line arguments without modifying code.
"""

# Default encoder model (SentenceTransformer for code embeddings)
DEFAULT_ENCODER = "BAAI/bge-code-v1"

# Default decoder model (LLM for code generation)
DEFAULT_DECODER = "Qwen/Qwen3-4B-Instruct-2507"

# Alternative decoders (for reference)
DECODER_ALTERNATIVES = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen2.5-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
}
