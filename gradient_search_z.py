"""
Gradient-based search in z-space using trained ranking predictor.

Simpler pipeline than u-space version - no normalizing flow required:
1. Load trained ranking predictor R(z)
2. Gradient ascent to find z* = argmax R(z)
3. Generate code using mapper + decoder
4. Evaluate and save successful programs

Uses same models as train_unified_mapper_optimized.py:
- Encoder: BAAI/bge-code-v1
- Decoder: Qwen/Qwen3-4B-Instruct-2507 (with Flash Attention 2)
"""

import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from mapper import Mapper
from ranking_score_predictor_z import (
    RankingScorePredictor,
    load_ranking_predictor,
    get_encoder_model,
    load_heuristics,
    get_evaluator
)
from utils import extract_python_code_robust, is_valid_python
from base.code import TextFunctionProgramConverter
from base.evaluate import SecureEvaluator


# ============================================================================
# Task-Specific Prompts
# ============================================================================

TASK_PROMPTS = {
    'tsp_construct': "Based on the above context, write a Python function that implements a heuristic for the Traveling Salesman Problem (TSP).",
    'cvrp_construct': "Based on the above context, write a Python function that implements a heuristic for the Capacitated Vehicle Routing Problem (CVRP).",
    'vrptw_construct': "Based on the above context, write a Python function that implements a heuristic for the Vehicle Routing Problem with Time Windows (VRPTW).",
    'jssp_construct': "Based on the above context, write a Python function that implements a heuristic for the Job Shop Scheduling Problem (JSSP).",
    'knapsack_construct': "Based on the above context, write a Python function that implements a heuristic for the 0/1 Knapsack Problem.",
    'online_bin_packing': "Based on the above context, write a Python function that implements an online heuristic for the Bin Packing Problem.",
    'qap_construct': "Based on the above context, write a Python function that implements a heuristic for the Quadratic Assignment Problem (QAP).",
    'cflp_construct': "Based on the above context, write a Python function that implements a heuristic for the Capacitated Facility Location Problem (CFLP).",
    'set_cover_construct': "Based on the above context, write a Python function that implements a greedy heuristic for the Set Cover Problem.",
    'admissible_set': "Based on the above context, write a Python function that implements a heuristic for computing admissible sets."
}


# ============================================================================
# Model Loading
# ============================================================================

def load_decoder(model_name: str = "Qwen/Qwen3-4B-Instruct-2507", device: str = "cuda"):
    """Load decoder model and tokenizer (same as train_unified_mapper_optimized.py)."""
    print(f"Loading decoder: {model_name}...")

    decoder_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Same as optimized training
        device_map="auto",
        attn_implementation="flash_attention_2",  # 2-3x attention speedup
        trust_remote_code=True
    )

    decoder_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    print(f"  Decoder loaded with Flash Attention 2.")
    return decoder_model, decoder_tokenizer


def load_mapper(mapper_path: str, decoder_model, device: str = "cuda"):
    """Load trained mapper model."""
    print(f"Loading mapper from {mapper_path}...")

    checkpoint = torch.load(mapper_path, map_location=device)

    # Handle both old and new checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        input_dim = checkpoint.get('input_dim')
        output_dim = checkpoint.get('output_dim')
        num_tokens = checkpoint.get('num_tokens')
    else:
        state_dict = checkpoint
        input_dim = None
        output_dim = None
        num_tokens = None

    # Infer dimensions if not in checkpoint
    if input_dim is None or output_dim is None or num_tokens is None:
        print("  Inferring dimensions from weights...")
        input_dim = state_dict['mlp.0.weight'].shape[1]
        final_output_size = state_dict['mlp.4.weight'].shape[0]
        output_dim = decoder_model.config.hidden_size
        num_tokens = final_output_size // output_dim

    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Num tokens: {num_tokens}")

    mapper_model = Mapper(input_dim, output_dim, num_tokens)
    mapper_model.load_state_dict(state_dict)
    mapper_model.eval()

    return mapper_model


# ============================================================================
# Gradient Search in Z-Space
# ============================================================================

def gradient_ascent_z(
    predictor: RankingScorePredictor,
    init_z: torch.Tensor,
    steps: int = 100,
    lr: float = 0.01,
    momentum: float = 0.9,
    noise_scale: float = 0.0,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Perform gradient ascent in z-space to maximize predicted ranking score.

    Args:
        predictor: Trained ranking score predictor R(z)
        init_z: Initial z vectors [num_starts, dim]
        steps: Number of gradient ascent steps
        lr: Learning rate
        momentum: Momentum coefficient (0 = no momentum)
        noise_scale: Scale of noise to add for exploration (0 = no noise)
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized z vectors [num_starts, dim]
    """
    predictor.eval()

    z = init_z.clone().to(device).requires_grad_(True)
    num_samples = z.shape[0]

    # Momentum buffer
    velocity = torch.zeros_like(z) if momentum > 0 else None

    if verbose:
        print(f"Gradient ascent: {num_samples} starts, {steps} steps, lr={lr}")

    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()

        # Forward pass
        scores = predictor(z)

        # Maximize score (negative for gradient ascent)
        loss = -scores.mean()
        loss.backward()

        with torch.no_grad():
            if momentum > 0:
                velocity = momentum * velocity + z.grad
                z += lr * velocity
            else:
                z += lr * z.grad

            # Add noise for exploration
            if noise_scale > 0:
                z += noise_scale * torch.randn_like(z)

        if verbose and (step + 1) % 20 == 0:
            avg_score = -loss.item()
            print(f"  Step {step+1}/{steps} | Avg score: {avg_score:.4f}")

    return z.detach()


def multi_start_gradient_search(
    predictor: RankingScorePredictor,
    num_starts: int = 10,
    steps: int = 100,
    lr: float = 0.01,
    init_from_data: torch.Tensor = None,
    init_noise: float = 0.1,
    top_k_init: int = 5,
    device: str = 'cuda',
    verbose: bool = True
) -> torch.Tensor:
    """
    Multi-start gradient search with different initialization strategies.

    Args:
        predictor: Trained ranking predictor
        num_starts: Number of parallel searches
        steps: Gradient steps per search
        lr: Learning rate
        init_from_data: Optional tensor of existing z vectors to initialize from
        init_noise: Noise to add to data-initialized starts
        top_k_init: Number of top embeddings to use for initialization
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized z vectors [num_starts, dim]
    """
    dim = predictor.input_dim

    if init_from_data is not None and len(init_from_data) > 0:
        # Initialize from top-k data points + noise
        n_from_data = min(num_starts, len(init_from_data), top_k_init)
        n_random = num_starts - n_from_data

        # Sample from top embeddings
        init_z_data = init_from_data[:n_from_data].clone()
        init_z_data += init_noise * torch.randn_like(init_z_data)

        if n_random > 0:
            # Add some random starts for exploration
            init_z_random = torch.randn(n_random, dim, device=device)
            init_z = torch.cat([init_z_data.to(device), init_z_random], dim=0)
        else:
            init_z = init_z_data.to(device)

        if verbose:
            print(f"Initialized {n_from_data} from data, {n_random} random")
    else:
        # Pure random initialization
        init_z = torch.randn(num_starts, dim, device=device)
        if verbose:
            print(f"Initialized {num_starts} random starts")

    # Run gradient ascent
    optimized_z = gradient_ascent_z(
        predictor=predictor,
        init_z=init_z,
        steps=steps,
        lr=lr,
        device=device,
        verbose=verbose
    )

    # Get final scores and sort
    with torch.no_grad():
        final_scores = predictor(optimized_z).squeeze()

    if verbose:
        print(f"\nFinal scores:")
        print(f"  Min:  {final_scores.min().item():.4f}")
        print(f"  Max:  {final_scores.max().item():.4f}")
        print(f"  Mean: {final_scores.mean().item():.4f}")

    # Sort by score (descending)
    sorted_indices = torch.argsort(final_scores, descending=True)
    optimized_z = optimized_z[sorted_indices]

    return optimized_z


# ============================================================================
# Code Generation
# ============================================================================

def generate_code_from_z(
    z_vector: np.ndarray,
    mapper_model,
    decoder_model,
    decoder_tokenizer,
    skeleton_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024
) -> tuple:
    """Generate code from a latent vector z using mapper and decoder."""

    embed_layer = decoder_model.get_input_embeddings()
    device = embed_layer.weight.device
    dtype = embed_layer.weight.dtype

    # Ensure z is the right shape
    if isinstance(z_vector, torch.Tensor):
        z_tensor = z_vector.clone().detach().float()
    else:
        z_tensor = torch.tensor(z_vector, dtype=torch.float32)

    if z_tensor.dim() == 1:
        z_tensor = z_tensor.unsqueeze(0)

    z_tensor = z_tensor.to(device)

    with torch.no_grad():
        # Map z to soft prompts
        soft_prompt_embeds = mapper_model(z_tensor).to(device, dtype=dtype)

        # Prepare instruction
        instruction_messages = [{"role": "user", "content": skeleton_prompt}]
        instruction_ids = decoder_tokenizer.apply_chat_template(
            instruction_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        instruction_embeds = embed_layer(instruction_ids)

        # Concatenate soft prompts + instruction
        inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)

        # Generate
        generated_ids = decoder_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id
        )

        raw_output = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        clean_code = extract_python_code_robust(raw_output, include_preface=True)

        return clean_code, raw_output


# ============================================================================
# Main Search Pipeline
# ============================================================================

def gradient_search_pipeline(
    task_name: str,
    predictor_path: str,
    mapper_path: str,
    num_iterations: int = 5,
    num_searches_per_iter: int = 10,
    gradient_steps: int = 100,
    lr: float = 0.01,
    temperature: float = 0.7,
    decoder_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    device: str = "cuda",
    output_dir: str = "gradient_search_results",
    verbose: bool = True
):
    """
    Full gradient search pipeline in z-space.

    Args:
        task_name: Task to search for (e.g., 'tsp_construct')
        predictor_path: Path to trained ranking predictor
        mapper_path: Path to trained mapper
        num_iterations: Number of search iterations
        num_searches_per_iter: Number of gradient searches per iteration
        gradient_steps: Steps per gradient search
        lr: Learning rate for gradient ascent
        temperature: Sampling temperature for decoder
        decoder_name: Decoder model name
        device: Device to use
        output_dir: Directory to save results
        verbose: Print progress
    """

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Gradient Search in Z-Space")
    print("="*70)
    print(f"Task: {task_name}")
    print(f"Predictor: {predictor_path}")
    print(f"Mapper: {mapper_path}")
    print()

    # ===== Load Models =====

    # Load ranking predictor
    print("Loading ranking predictor...")
    predictor, predictor_info = load_ranking_predictor(predictor_path, device)
    predictor.to(device)
    predictor.eval()

    # Load decoder
    decoder_model, decoder_tokenizer = load_decoder(decoder_name, device)

    # Load mapper
    mapper_model = load_mapper(mapper_path, decoder_model, device)
    embed_layer = decoder_model.get_input_embeddings()
    mapper_model = mapper_model.to(embed_layer.weight.device)

    # Load encoder (for encoding existing heuristics)
    print("Loading encoder (BAAI/bge-code-v1)...")
    encoder_model = get_encoder_model(device)

    # Load evaluator
    print(f"Loading evaluator for {task_name}...")
    evaluator = get_evaluator(task_name)
    secure_eval = SecureEvaluator(evaluator, debug_mode=False)

    # Get skeleton prompt
    skeleton_prompt = TASK_PROMPTS.get(task_name)
    if skeleton_prompt is None:
        raise ValueError(f"No prompt defined for task: {task_name}")

    # ===== Load Initial Data for Warm Start =====

    print(f"\nLoading existing heuristics for warm start...")
    try:
        heuristics = load_heuristics(task_name)
        codes = list(heuristics.values())

        # Encode existing heuristics
        with torch.no_grad():
            init_embeddings = encoder_model.encode(
                codes,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=True
            )

        # Score existing heuristics with predictor
        with torch.no_grad():
            init_scores = predictor(init_embeddings).squeeze()

        # Sort by predicted score
        sorted_indices = torch.argsort(init_scores, descending=True)
        init_embeddings = init_embeddings[sorted_indices]

        print(f"  Loaded {len(codes)} heuristics")
        print(f"  Top predicted score: {init_scores.max().item():.4f}")

    except Exception as e:
        print(f"  Could not load heuristics: {e}")
        init_embeddings = None

    # Free encoder memory
    del encoder_model
    torch.cuda.empty_cache()

    # ===== Run Search =====

    all_results = []
    successful_programs = {}

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        # Gradient search
        print(f"\nRunning gradient search ({num_searches_per_iter} starts, {gradient_steps} steps)...")

        optimized_z = multi_start_gradient_search(
            predictor=predictor,
            num_starts=num_searches_per_iter,
            steps=gradient_steps,
            lr=lr,
            init_from_data=init_embeddings,
            init_noise=0.1,
            top_k_init=10,
            device=device,
            verbose=verbose
        )

        # Generate and evaluate
        print(f"\nGenerating and evaluating {len(optimized_z)} candidates...")

        successful = 0
        failed = 0

        for idx, z in enumerate(optimized_z):
            if verbose:
                print(f"\n--- Candidate {idx+1}/{len(optimized_z)} ---")

            try:
                # Generate code
                clean_code, raw_output = generate_code_from_z(
                    z.cpu().numpy(),
                    mapper_model,
                    decoder_model,
                    decoder_tokenizer,
                    skeleton_prompt,
                    temperature=temperature
                )

                # Validate
                if not clean_code or not is_valid_python(clean_code):
                    if verbose:
                        print("  Invalid Python syntax")
                    failed += 1
                    continue

                # Parse function
                program = TextFunctionProgramConverter.text_to_program(clean_code)
                if program is None or len(program.functions) == 0:
                    if verbose:
                        print("  Could not parse function")
                    failed += 1
                    continue

                func_name = program.functions[0].name

                # Check for duplicates
                code_hash = hash(clean_code.strip())
                if code_hash in [hash(c.strip()) for c in successful_programs.values()]:
                    if verbose:
                        print("  Duplicate program")
                    failed += 1
                    continue

                # Evaluate
                try:
                    score = secure_eval.evaluate_program(clean_code)

                    if score is None or not np.isfinite(score):
                        if verbose:
                            print(f"  Invalid score: {score}")
                        failed += 1
                        continue

                except Exception as e:
                    if verbose:
                        print(f"  Evaluation error: {e}")
                    failed += 1
                    continue

                # Success!
                program_id = f"iter{iteration}_idx{idx}"
                successful_programs[program_id] = clean_code

                all_results.append({
                    'program_id': program_id,
                    'iteration': iteration,
                    'score': score,
                    'code': clean_code,
                    'function_name': func_name
                })

                successful += 1

                if verbose:
                    print(f"  Score: {score:.4f} | Function: {func_name}")

            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                failed += 1
                continue

        # Iteration summary
        print(f"\nIteration {iteration + 1} Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total discovered: {len(successful_programs)}")

        if all_results:
            scores = [r['score'] for r in all_results]
            print(f"  Best score so far: {max(scores):.4f}")
            print(f"  Avg score: {np.mean(scores):.4f}")

        # Update init_embeddings with successful programs for next iteration
        if successful > 0:
            # Re-encode successful programs
            new_codes = [r['code'] for r in all_results[-successful:]]
            encoder_model = get_encoder_model(device)
            with torch.no_grad():
                new_embeddings = encoder_model.encode(
                    new_codes,
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False
                )
            del encoder_model
            torch.cuda.empty_cache()

            # Combine with existing
            if init_embeddings is not None:
                init_embeddings = torch.cat([new_embeddings, init_embeddings], dim=0)
            else:
                init_embeddings = new_embeddings

    # ===== Final Results =====

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal programs discovered: {len(successful_programs)}")

    if all_results:
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)

        scores = [r['score'] for r in all_results]
        print(f"Best score: {max(scores):.4f}")
        print(f"Avg score: {np.mean(scores):.4f}")
        print(f"Std score: {np.std(scores):.4f}")

        # Show top 5
        print(f"\nTop 5 Programs:")
        print("-"*70)
        for i, result in enumerate(all_results[:5]):
            print(f"\n{i+1}. Score: {result['score']:.4f} | {result['function_name']}")
            print(f"   ID: {result['program_id']}")
            code_preview = result['code'][:200].replace('\n', '\n   ')
            print(f"   {code_preview}...")

    # ===== Save Results =====

    # Save programs as JSON
    programs_path = os.path.join(output_dir, f"{task_name}_gradient_searched.json")
    with open(programs_path, 'w') as f:
        json.dump(successful_programs, f, indent=2)
    print(f"\nSaved {len(successful_programs)} programs to {programs_path}")

    # Save detailed results
    results_path = os.path.join(output_dir, f"{task_name}_search_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed results to {results_path}")

    return all_results, successful_programs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Gradient search in z-space')
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--predictor', type=str, default='ranking_predictor_z.pth', help='Path to ranking predictor')
    parser.add_argument('--mapper', type=str, default='Mapper_Checkpoints/unified_mapper.pth', help='Path to mapper')
    parser.add_argument('--decoder', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='Decoder model')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of search iterations')
    parser.add_argument('--num_searches', type=int, default=10, help='Searches per iteration')
    parser.add_argument('--gradient_steps', type=int, default=100, help='Gradient steps per search')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Decoder temperature')
    parser.add_argument('--output_dir', type=str, default='gradient_search_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    results, programs = gradient_search_pipeline(
        task_name=args.task,
        predictor_path=args.predictor,
        mapper_path=args.mapper,
        num_iterations=args.num_iterations,
        num_searches_per_iter=args.num_searches,
        gradient_steps=args.gradient_steps,
        lr=args.lr,
        temperature=args.temperature,
        decoder_name=args.decoder,
        device=args.device,
        output_dir=args.output_dir,
        verbose=True
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
