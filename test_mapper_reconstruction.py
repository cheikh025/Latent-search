"""
Simple Mapper Reconstruction Test

Shows 5 random programs and their reconstructions from the trained mapper.
"""

import os
import json
import glob
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from mapper import Mapper
from utils import is_valid_python, extract_python_code


# Task prompts (same as training)
TASK_PROMPTS = {
    'tsp_construct': "Write a Python function that implements a heuristic for the Traveling Salesman Problem (TSP).",
    'cvrp_construct': "Write a Python function that implements a heuristic for the Capacitated Vehicle Routing Problem (CVRP).",
    'vrptw_construct': "Write a Python function that implements a heuristic for the Vehicle Routing Problem with Time Windows (VRPTW).",
    'jssp_construct': "Write a Python function that implements a heuristic for the Job Shop Scheduling Problem (JSSP).",
    'knapsack_construct': "Write a Python function that implements a heuristic for the 0/1 Knapsack Problem.",
    'online_bin_packing': "Write a Python function that implements an online heuristic for the Bin Packing Problem.",
    'qap_construct': "Write a Python function that implements a heuristic for the Quadratic Assignment Problem (QAP).",
    'cflp_construct': "Write a Python function that implements a heuristic for the Capacitated Facility Location Problem (CFLP).",
    'set_cover_construct': "Write a Python function that implements a greedy heuristic for the Set Cover Problem.",
    'admissible_set': "Write a Python function that implements a heuristic for computing admissible sets."
}


def load_heuristics_from_tasks():
    """Load heuristics from task directories (same as training)."""
    all_heuristics = []
    augmented_files = glob.glob(os.path.join("task", "*", "augmented.json"))

    print(f"Found {len(augmented_files)} task directories\n")

    for aug_file in sorted(augmented_files):
        task_name = Path(aug_file).parent.name
        skeleton_prompt = TASK_PROMPTS.get(task_name)

        if skeleton_prompt is None:
            continue

        with open(aug_file, 'r', encoding='utf-8') as f:
            heuristics_dict = json.load(f)

        for heuristic_name, code in heuristics_dict.items():
            if code and isinstance(code, str) and is_valid_python(code):
                all_heuristics.append((code, task_name, skeleton_prompt, heuristic_name))

    print(f"Loaded {len(all_heuristics)} heuristics\n")
    return all_heuristics


def encode_heuristics(heuristics_list, encoder_model, device):
    """Encode heuristics using the same encoder as training."""
    codes = [h[0] for h in heuristics_list]

    print(f"Encoding {len(codes)} programs...")
    encoder_model.eval()
    encoder_model.to(device)

    with torch.no_grad():
        embeddings = encoder_model.encode(
            codes,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=True
        )

    return embeddings.cpu().numpy()


def load_mapper(checkpoint_path, device):
    """Load mapper from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    num_tokens = checkpoint['num_tokens']

    mapper = Mapper(input_dim=input_dim, output_dim=output_dim, num_tokens=num_tokens)
    mapper.load_state_dict(checkpoint['model_state_dict'])
    mapper.to(device)
    mapper.eval()

    print(f"✓ Loaded mapper (input={input_dim}, output={output_dim}, tokens={num_tokens})\n")
    return mapper


def reconstruct_code(mapper, z, decoder_model, decoder_tokenizer, skeleton_prompt):
    """Reconstruct code from latent z using mapper and decoder."""

    # Get device info from decoder
    embed_layer = decoder_model.get_input_embeddings()
    embed_device = embed_layer.weight.device
    embed_dtype = embed_layer.weight.dtype

    # Convert z to tensor and generate soft prompts
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(mapper.mlp[0].weight.device)

    with torch.no_grad():
        soft_prompts = mapper(z_tensor).to(embed_device, dtype=embed_dtype)

    # Prepare instruction prompt
    messages = [{"role": "user", "content": skeleton_prompt}]
    prompt_ids = decoder_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = torch.tensor([prompt_ids]).to(embed_device)
    prompt_embeds = embed_layer(prompt_ids)

    # Concatenate soft prompts + instruction
    inputs_embeds = torch.cat([soft_prompts, prompt_embeds], dim=1)

    # Generate
    with torch.no_grad():
        outputs = decoder_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
        )

    # Decode
    generated_text = decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract code
    code_blocks = extract_python_code(generated_text)
    if code_blocks:
        return code_blocks[0]
    return generated_text.strip()


def main():
    print("="*80)
    print("MAPPER RECONSTRUCTION TEST")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load heuristics from tasks (same as training)
    print("Loading heuristics from task directories...")
    heuristics_list = load_heuristics_from_tasks()

    # Load encoder (same as training)
    print("Loading encoder: BAAI/bge-code-v1")
    encoder = SentenceTransformer(
        "BAAI/bge-code-v1",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16}
    ).to(device)

    # Encode all heuristics
    embeddings = encode_heuristics(heuristics_list, encoder, device)

    # Free encoder
    del encoder
    torch.cuda.empty_cache()

    # Load mapper
    print("\nLoading mapper checkpoint...")
    mapper = load_mapper("Mapper_Checkpoints/unified_mapper.pth", device)

    # Load decoder
    print("Loading decoder: Qwen/Qwen2.5-Coder-7B-Instruct")
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True
    )
    decoder_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    decoder_model.eval()

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    print("✓ Decoder loaded\n")

    # Select 5 random samples
    np.random.seed(42)
    indices = np.random.choice(len(heuristics_list), size=5, replace=False)

    print("="*80)
    print("RECONSTRUCTION RESULTS (5 SAMPLES)")
    print("="*80)
    print()

    # Test reconstruction for each sample
    for i, idx in enumerate(indices):
        original_code, task_name, skeleton_prompt, heuristic_name = heuristics_list[idx]
        z = embeddings[idx]

        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/5")
        print(f"{'='*80}")
        print(f"Task: {task_name}")
        print(f"Heuristic: {heuristic_name}")
        print(f"\n--- ORIGINAL CODE ---")
        print(original_code)

        print(f"\n--- RECONSTRUCTED CODE ---")
        reconstructed = reconstruct_code(mapper, z, decoder_model, decoder_tokenizer, skeleton_prompt)
        print(reconstructed)

        print(f"\n--- STATUS ---")
        print(f"Valid Python: {is_valid_python(reconstructed)}")
        print()

    print("="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
