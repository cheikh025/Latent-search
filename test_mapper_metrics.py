"""
Mapper Reconstruction Metrics Evaluation

Computes quantitative metrics for mapper reconstruction quality:
- Exact match rate
- AST structural similarity
- BLEU score
- Valid Python syntax rate
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
from collections import Counter

from mapper import Mapper
from utils import is_valid_python, extract_python_code, compare_code


# Task prompts (same as training)
TASK_PROMPTS = {
    'tsp_construct': "Based on the above context, write a Python function that implements a TSP heuristic. Visit each node once and return to the start by choosing the next node step-by-step from the current node.",
    'cvrp_construct': "Based on the above context, write a Python function that implements a CVRP heuristic. Build routes from a depot while respecting vehicle capacity and choosing the next customer each step.",
    'vrptw_construct': "Based on the above context, write a Python function that implements a VRPTW heuristic. Choose the next customer step-by-step while respecting time windows, capacity, and travel times.",
    'jssp_construct': "Based on the above context, write a Python function that implements a JSSP heuristic. Schedule the next feasible operation to reduce overall makespan given machine and job constraints.",
    'knapsack_construct': "Based on the above context, write a Python function that implements a 0/1 knapsack heuristic. Select the next item to pack given remaining capacity and item weights/values.",
    'online_bin_packing': "Based on the above context, write a Python function that implements an online bin packing heuristic. Return a priority score for each bin to place the incoming item.",
    'qap_construct': "Based on the above context, write a Python function that implements a QAP heuristic. Extend a partial assignment of facilities to locations to reduce interaction cost.",
    'cflp_construct': "Based on the above context, write a Python function that implements a CFLP heuristic. Assign customers to facilities step-by-step while respecting capacity and minimizing cost.",
    'set_cover_construct': "Based on the above context, write a Python function that implements a greedy set cover heuristic. Pick the next subset to cover remaining elements.",
    'admissible_set': "Based on the above context, write a Python function that implements a heuristic priority score for adding a candidate vector to an admissible set."
}

NUM_TEST_SAMPLES = 100  # Number of samples to evaluate


def load_heuristics_from_tasks():
    """Load heuristics from task directories (same as training)."""
    all_heuristics = []
    augmented_files = glob.glob(os.path.join("task", "*", "augmented.json"))

    print(f"Found {len(augmented_files)} task directories")

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

    print(f"Loaded {len(all_heuristics)} total heuristics\n")
    return all_heuristics


def encode_heuristics(heuristics_list, encoder_model, device, batch_size=32):
    """Encode heuristics using the same encoder as training."""
    codes = [h[0] for h in heuristics_list]

    print(f"Encoding {len(codes)} programs...")
    encoder_model.eval()
    encoder_model.to(device)

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Encoding"):
            batch_codes = codes[i:i+batch_size]
            batch_embeddings = encoder_model.encode(
                batch_codes,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"✓ Encoded all programs (shape: {embeddings.shape})\n")
    return embeddings


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

    print(f"✓ Loaded mapper: input={input_dim}, output={output_dim}, tokens={num_tokens}")
    print(f"  Trained on {checkpoint.get('total_programs', 'unknown')} programs")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}\n")
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


def compute_bleu_score(reference, hypothesis):
    """Compute simple word-level BLEU score."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Unigram precision
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)

    common = sum((hyp_counter & ref_counter).values())
    precision = common / len(hyp_tokens)

    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return bp * precision


def evaluate_metrics(heuristics_list, embeddings, mapper, decoder_model, decoder_tokenizer, num_samples):
    """Evaluate reconstruction metrics on test samples."""

    # Sample random test set
    np.random.seed(42)
    num_samples = min(num_samples, len(heuristics_list))
    indices = np.random.choice(len(heuristics_list), size=num_samples, replace=False)

    print(f"Evaluating on {num_samples} random samples...")
    print()

    # Metrics storage
    exact_matches = 0
    ast_matches = 0
    valid_syntax = 0
    bleu_scores = []
    task_results = {}

    # Evaluate each sample
    for idx in tqdm(indices, desc="Evaluating"):
        original_code, task_name, skeleton_prompt, heuristic_name = heuristics_list[idx]
        z = embeddings[idx]

        # Reconstruct
        try:
            reconstructed = reconstruct_code(mapper, z, decoder_model, decoder_tokenizer, skeleton_prompt)
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            reconstructed = ""

        # Compute metrics
        is_exact = (original_code.strip() == reconstructed.strip())
        is_ast_match = compare_code(original_code, reconstructed)
        is_valid = is_valid_python(reconstructed)
        bleu = compute_bleu_score(original_code, reconstructed)

        # Update counters
        if is_exact:
            exact_matches += 1
        if is_ast_match:
            ast_matches += 1
        if is_valid:
            valid_syntax += 1
        bleu_scores.append(bleu)

        # Track per-task metrics
        if task_name not in task_results:
            task_results[task_name] = {
                'count': 0,
                'exact': 0,
                'ast': 0,
                'valid': 0,
                'bleu': []
            }

        task_results[task_name]['count'] += 1
        task_results[task_name]['exact'] += int(is_exact)
        task_results[task_name]['ast'] += int(is_ast_match)
        task_results[task_name]['valid'] += int(is_valid)
        task_results[task_name]['bleu'].append(bleu)

    # Compute overall metrics
    metrics = {
        'num_samples': num_samples,
        'exact_match_rate': exact_matches / num_samples,
        'ast_match_rate': ast_matches / num_samples,
        'valid_syntax_rate': valid_syntax / num_samples,
        'mean_bleu': np.mean(bleu_scores),
        'median_bleu': np.median(bleu_scores),
        'std_bleu': np.std(bleu_scores),
    }

    return metrics, task_results


def print_results(metrics, task_results):
    """Print evaluation results."""

    print("\n" + "="*80)
    print("OVERALL RECONSTRUCTION METRICS")
    print("="*80)
    print(f"Total samples tested: {metrics['num_samples']}")
    print(f"Exact match rate:     {metrics['exact_match_rate']*100:.2f}%")
    print(f"AST match rate:       {metrics['ast_match_rate']*100:.2f}%")
    print(f"Valid syntax rate:    {metrics['valid_syntax_rate']*100:.2f}%")
    print(f"Mean BLEU score:      {metrics['mean_bleu']:.4f}")
    print(f"Median BLEU score:    {metrics['median_bleu']:.4f}")
    print(f"Std BLEU score:       {metrics['std_bleu']:.4f}")
    print("="*80)

    print("\n" + "="*80)
    print("PER-TASK BREAKDOWN")
    print("="*80)
    print(f"{'Task':<25} {'Samples':>8} {'Exact%':>8} {'AST%':>8} {'Valid%':>8} {'BLEU':>8}")
    print("-"*80)

    for task_name in sorted(task_results.keys()):
        result = task_results[task_name]
        count = result['count']
        exact_pct = (result['exact'] / count) * 100
        ast_pct = (result['ast'] / count) * 100
        valid_pct = (result['valid'] / count) * 100
        mean_bleu = np.mean(result['bleu'])

        print(f"{task_name:<25} {count:>8} {exact_pct:>7.1f}% {ast_pct:>7.1f}% {valid_pct:>7.1f}% {mean_bleu:>8.4f}")

    print("="*80)


def save_results(metrics, task_results, output_path="mapper_metrics.json"):
    """Save metrics to JSON file."""

    # Convert task results to serializable format
    task_results_serializable = {}
    for task, result in task_results.items():
        task_results_serializable[task] = {
            'count': result['count'],
            'exact_match_rate': result['exact'] / result['count'],
            'ast_match_rate': result['ast'] / result['count'],
            'valid_syntax_rate': result['valid'] / result['count'],
            'mean_bleu': float(np.mean(result['bleu'])),
            'median_bleu': float(np.median(result['bleu'])),
        }

    output = {
        'overall_metrics': metrics,
        'per_task_metrics': task_results_serializable
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Metrics saved to {output_path}")


def main():
    print("="*80)
    print("MAPPER RECONSTRUCTION METRICS EVALUATION")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load heuristics
    print("Loading heuristics from task directories...")
    heuristics_list = load_heuristics_from_tasks()

    # Load encoder
    print("Loading encoder: BAAI/bge-code-v1")
    encoder = SentenceTransformer(
        "BAAI/bge-code-v1",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16}
    ).to(device)

    # Encode
    embeddings = encode_heuristics(heuristics_list, encoder, device)

    # Free encoder
    del encoder
    torch.cuda.empty_cache()

    # Load mapper
    print("Loading mapper checkpoint...")
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

    # Evaluate
    metrics, task_results = evaluate_metrics(
        heuristics_list,
        embeddings,
        mapper,
        decoder_model,
        decoder_tokenizer,
        num_samples=NUM_TEST_SAMPLES
    )

    # Print results
    print_results(metrics, task_results)

    # Save results
    save_results(metrics, task_results)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
