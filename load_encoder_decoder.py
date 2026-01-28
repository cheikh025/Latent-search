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

from model_config import DEFAULT_ENCODER, DEFAULT_DECODER


def load_encoder(model_name: str = None, device: str = "cuda"):
    """Load encoder with the same settings as training.

    Args:
        model_name: Encoder model name. Defaults to DEFAULT_ENCODER from model_config.py
        device: Device to load the model on.
    """
    if model_name is None:
        model_name = DEFAULT_ENCODER

    encoder_model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    ).to(device)
    encoder_model.eval()
    return encoder_model


def load_decoder(model_name: str = None, device: str = "auto"):
    """Load decoder with Flash Attention 2 (same as training).

    Args:
        model_name: Decoder model name. Defaults to DEFAULT_DECODER from model_config.py
        device: Device map for model placement. Defaults to "auto".
    """
    if model_name is None:
        model_name = DEFAULT_DECODER

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
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER, help=f"Encoder model name (default: {DEFAULT_ENCODER})")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, help=f"Decoder model name (default: {DEFAULT_DECODER})")
    parser.add_argument("--device", type=str, default="cuda", help="Device for encoder")
    args = parser.parse_args()

    # Encoder (uses same config as training)
    if args.encoder != DEFAULT_ENCODER:
        print(f"Warning: encoder name differs from training default ({DEFAULT_ENCODER}).")
    encoder = load_encoder(model_name=args.encoder, device=args.device)

    # Decoder (uses same config as training)
    decoder, tokenizer = load_decoder(model_name=args.decoder)

    print("Loaded encoder:", args.encoder, "on", args.device)
    print("Loaded decoder:", args.decoder)
    print("Decoder dtype:", decoder.dtype)


if __name__ == "__main__":
    main()
