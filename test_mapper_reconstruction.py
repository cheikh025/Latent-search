"""
Simple test: encode 5 random TSP heuristics, pass through mapper, reconstruct and compare.
"""

import torch
import json
import random
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from mapper import Mapper
from utils import extract_python_code_robust

# ============================================================================
# Config
# ============================================================================

TASK = "tsp_construct"
NUM_SAMPLES = 5
MAPPER_PATH = "Mapper_Checkpoints/unified_mapper.pth"
SKELETON_PROMPT = "Based on the above context, write a Python function that implements a heuristic for the Traveling Salesman Problem (TSP)."

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ============================================================================
# Load Models
# ============================================================================

print("Loading encoder (BAAI/bge-code-v1)...")
encoder = SentenceTransformer(
    "BAAI/bge-code-v1",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16}
).to(device)
encoder.eval()

print("Loading decoder (Qwen3-4B-Instruct-2507)...")
decoder = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading mapper...")
checkpoint = torch.load(MAPPER_PATH, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)

# Handle torch.compile() prefix
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    print("  Removing torch.compile() prefix...")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

input_dim = checkpoint.get('input_dim') or state_dict['mlp.0.weight'].shape[1]
output_dim = checkpoint.get('output_dim') or decoder.config.hidden_size
num_tokens = checkpoint.get('num_tokens') or state_dict['mlp.4.weight'].shape[0] // output_dim

mapper = Mapper(input_dim, output_dim, num_tokens)
mapper.load_state_dict(state_dict)
mapper.eval()

embed_layer = decoder.get_input_embeddings()
mapper = mapper.to(embed_layer.weight.device)

print(f"Mapper: {input_dim}D -> {num_tokens} tokens x {output_dim}D\n")

# ============================================================================
# Load Heuristics
# ============================================================================

print(f"Loading heuristics from task/{TASK}/heuristics.json...")
with open(f"task/{TASK}/heuristics.json", "r") as f:
    heuristics = json.load(f)

names = list(heuristics.keys())
random.shuffle(names)
selected = names[:NUM_SAMPLES]

print(f"Selected {NUM_SAMPLES} random heuristics: {selected}\n")

# ============================================================================
# Encode -> Mapper -> Decode
# ============================================================================

for i, name in enumerate(selected):
    original_code = heuristics[name]

    print("="*80)
    print(f"Sample {i+1}/{NUM_SAMPLES}: {name}")
    print("="*80)

    # Encode
    with torch.no_grad():
        z = encoder.encode([original_code], convert_to_tensor=True, device=device)

    # Map to soft prompts
    with torch.no_grad():
        z_input = z.to(embed_layer.weight.device).float()
        soft_prompts = mapper(z_input).to(embed_layer.weight.device, dtype=embed_layer.weight.dtype)

    # Prepare instruction
    messages = [{"role": "user", "content": SKELETON_PROMPT}]
    instruction_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(embed_layer.weight.device)

    instruction_embeds = embed_layer(instruction_ids)
    inputs_embeds = torch.cat([soft_prompts, instruction_embeds], dim=1)

    # Generate
    with torch.no_grad():
        output_ids = decoder.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reconstructed = extract_python_code_robust(raw_output, include_preface=True)

    # Print comparison
    print("\n--- ORIGINAL ---")
    print(original_code[:600])
    if len(original_code) > 600:
        print("... [truncated]")

    print("\n--- RECONSTRUCTED ---")
    print(reconstructed[:600] if reconstructed else "[Failed to extract code]")
    if reconstructed and len(reconstructed) > 600:
        print("... [truncated]")

    print("\n")

print("Done!")
