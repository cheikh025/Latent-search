"""
Minimal loader for the encoder and decoder models using the same settings as
train_unified_mapper_optimized.py.
"""

import os
import argparse

# Match training script allocator setting
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_encoder(model_name: str = "BAAI/bge-code-v1", device: str = "cuda"):
    """Load encoder with the same settings as training."""
    encoder_model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    ).to(device)
    encoder_model.eval()
    return encoder_model


def load_decoder(model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
    """Load Qwen decoder with Flash Attention 2 (same as training)."""
    decoder_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    decoder_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    return decoder_model, decoder_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Load encoder + decoder models")
    parser.add_argument("--encoder", type=str, default="BAAI/bge-code-v1", help="Encoder model name")
    parser.add_argument("--decoder", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Decoder model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device for encoder")
    args = parser.parse_args()

    # Encoder (uses same config as training)
    if args.encoder != "BAAI/bge-code-v1":
        print("Warning: encoder name differs from training default.")
    encoder = load_encoder(model_name=args.encoder, device=args.device)

    # Decoder (uses same config as training)
    decoder, tokenizer = load_decoder(model_name=args.decoder)

    print("Loaded encoder:", args.encoder, "on", args.device)
    print("Loaded decoder:", args.decoder)
    print("Decoder dtype:", decoder.dtype)


if __name__ == "__main__":
    main()
