"""
Centralized model configuration for encoder and decoder models.

This file defines the default model names used throughout the Latent Search framework.
Users can override these via command-line arguments without modifying code.
"""

# Default encoder model (SentenceTransformer for code embeddings)
DEFAULT_ENCODER = "Qwen/Qwen3-Embedding-0.6B"

# Matryoshka embedding dimension
# Set to None to use the encoder's full native dimension (e.g., 1024 for bge-code-v1)
# Set to a specific value (e.g., 256, 512, 768) to truncate embeddings
# Smaller dimensions = faster training/inference, slightly lower quality
DEFAULT_MATRYOSHKA_DIM = 128

# Default decoder model (LLM for code generation)
DEFAULT_DECODER = "Qwen/Qwen3-4B-Instruct-2507"

# Alternative decoders (for reference)
DECODER_ALTERNATIVES = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen2.5-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
}
