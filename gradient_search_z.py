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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    # Handle torch.compile() prefix: remove '_orig_mod.' from keys
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Detected torch.compile() checkpoint, removing '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

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
    trust_region_lambda: float = 0.0,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Simple gradient ascent in z-space to maximize predicted ranking score.

    Args:
        predictor: Trained ranking score predictor R(z)
        init_z: Initial z vectors [num_starts, dim]
        steps: Number of gradient ascent steps
        lr: Learning rate
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
                            Loss = -R(z) + λ||z - z_init||²
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized z vectors [num_starts, dim]
    """
    predictor.eval()

    z = init_z.clone().to(device).requires_grad_(True)
    z_init = init_z.clone().to(device)  # Keep initial point for trust region
    num_samples = z.shape[0]

    if verbose:
        trust_msg = f", trust_region_λ={trust_region_lambda}" if trust_region_lambda > 0 else ""
        print(f"Gradient ascent: {num_samples} starts, {steps} steps, lr={lr}{trust_msg}")

    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()

        # Forward pass
        scores = predictor(z)

        # Maximize score (negative for gradient ascent)
        loss = -scores.mean()

        # Add trust region penalty if enabled
        if trust_region_lambda > 0:
            trust_region_penalty = trust_region_lambda * torch.mean((z - z_init) ** 2)
            loss = loss + trust_region_penalty

        loss.backward()

        # Simple gradient ascent update
        with torch.no_grad():
            z += lr * z.grad

        if verbose and (step + 1) % 20 == 0:
            avg_score = -loss.item()
            if trust_region_lambda > 0:
                with torch.no_grad():
                    distance = torch.mean(torch.sqrt(torch.sum((z - z_init) ** 2, dim=1))).item()
                print(f"  Step {step+1}/{steps} | Avg score: {avg_score:.4f} | Dist from init: {distance:.4f}")
            else:
                print(f"  Step {step+1}/{steps} | Avg score: {avg_score:.4f}")

    return z.detach()


def multi_start_gradient_search(
    predictor: RankingScorePredictor,
    num_starts: int = 10,
    steps: int = 100,
    lr: float = 0.01,
    trust_region_lambda: float = 0.0,
    init_from_data: torch.Tensor = None,
    device: str = 'cuda',
    verbose: bool = True
) -> torch.Tensor:
    """
    Multi-start gradient search initialized from existing data.

    Args:
        predictor: Trained ranking predictor
        num_starts: Number of parallel searches
        steps: Gradient steps per search
        lr: Learning rate
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        init_from_data: Tensor of existing z vectors to initialize from (required)
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized z vectors [num_starts, dim]
    """
    if init_from_data is None or len(init_from_data) == 0:
        raise ValueError("init_from_data is required - must provide existing z vectors")

    # Initialize from top data points (no noise)
    n_available = len(init_from_data)
    n_to_use = min(num_starts, n_available)

    # Take top-n embeddings directly (already sorted by score)
    init_z = init_from_data[:n_to_use].clone().to(device)

    if verbose:
        print(f"Initialized {n_to_use} searches from top-{n_to_use} embeddings (no noise)")

    # Run gradient ascent
    optimized_z = gradient_ascent_z(
        predictor=predictor,
        init_z=init_z,
        steps=steps,
        lr=lr,
        trust_region_lambda=trust_region_lambda,
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


def generate_code_from_z_batch(
    z_vectors: torch.Tensor,
    mapper_model,
    decoder_model,
    decoder_tokenizer,
    skeleton_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024
) -> list:
    """
    Generate code from multiple latent vectors in a single batch.

    Args:
        z_vectors: Tensor of shape [batch_size, dim]
        mapper_model: Trained mapper
        decoder_model: Decoder LLM
        decoder_tokenizer: Tokenizer
        skeleton_prompt: Task instruction prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of (clean_code, raw_output) tuples
    """
    embed_layer = decoder_model.get_input_embeddings()
    device = embed_layer.weight.device
    dtype = embed_layer.weight.dtype

    # Ensure correct shape
    if isinstance(z_vectors, np.ndarray):
        z_vectors = torch.from_numpy(z_vectors)

    if z_vectors.dim() == 1:
        z_vectors = z_vectors.unsqueeze(0)

    batch_size = z_vectors.shape[0]
    z_vectors = z_vectors.to(device).float()

    with torch.no_grad():
        # Batch map z -> soft prompts: [batch_size, num_tokens, hidden_dim]
        soft_prompt_embeds = mapper_model(z_vectors).to(device, dtype=dtype)

        # Prepare instruction (same for all samples)
        instruction_messages = [{"role": "user", "content": skeleton_prompt}]
        instruction_ids = decoder_tokenizer.apply_chat_template(
            instruction_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Get instruction embeddings and repeat for batch
        instruction_embeds = embed_layer(instruction_ids)  # [1, seq_len, hidden]
        instruction_embeds = instruction_embeds.expand(batch_size, -1, -1)  # [batch, seq_len, hidden]

        # Concatenate: [batch, num_soft_tokens + seq_len, hidden]
        inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)

        # Create attention mask (all 1s since no padding in inputs)
        attention_mask = torch.ones(
            batch_size, inputs_embeds.shape[1],
            dtype=torch.long, device=device
        )

        # Batched generation
        generated_ids = decoder_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id
        )

        # Decode all outputs
        results = []
        for i in range(batch_size):
            raw_output = decoder_tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            clean_code = extract_python_code_robust(raw_output, include_preface=True)
            results.append((clean_code, raw_output))

        return results


# ============================================================================
# Main Search Pipeline
# ============================================================================

def gradient_search_pipeline(
    task_name: str,
    predictor_path: str,
    mapper_path: str,
    num_iterations: int = 5,
    num_searches_per_iter: int = 10,
    top_k_pool_size: int = 30,
    gradient_steps: int = 100,
    lr: float = 0.01,
    trust_region_lambda: float = 0.0,
    temperature: float = 0.7,
    decoder_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    device: str = "cuda",
    output_dir: str = "gradient_search_results",
    num_evaluators: int = 4,
    generation_batch_size: int = 4,
    verbose: bool = True
):
    """
    Full gradient search pipeline in z-space.

    Key Features:
    1. Evaluates ALL baseline heuristics on ACTUAL test instances (not predicted scores)
    2. Maintains database of all programs (baselines + discoveries) with actual scores
    3. Each iteration: sorts database by actual score, samples from top-k pool using softmax probabilities
    4. Guarantees starting from PROVEN best programs, not predicted best
    5. Softmax-weighted sampling: better programs have higher selection probability, but exploration is maintained
    6. No repetition in sampling (samples without replacement)
    7. Batched code generation for efficiency
    8. Parallel evaluation using ThreadPoolExecutor

    Args:
        task_name: Task to search for (e.g., 'tsp_construct')
        predictor_path: Path to trained ranking predictor
        mapper_path: Path to trained mapper
        num_iterations: Number of search iterations
        num_searches_per_iter: Number of gradient searches per iteration (how many to sample)
        top_k_pool_size: Pool size to sample seeds from (e.g., 30 means sample from best 30 programs)
        gradient_steps: Steps per gradient search
        lr: Learning rate for gradient ascent
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        temperature: Sampling temperature for decoder
        decoder_name: Decoder model name
        device: Device to use
        output_dir: Directory to save results
        num_evaluators: Number of parallel evaluation workers (default: 4)
        generation_batch_size: Batch size for code generation (default: 4, adjust based on GPU memory)
        verbose: Print progress
    """

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Gradient Search in Z-Space")
    print("="*70)
    print(f"Task: {task_name}")
    print(f"Predictor: {predictor_path}")
    print(f"Mapper: {mapper_path}")
    print(f"Parallel evaluators: {num_evaluators}")
    print(f"Generation batch size: {generation_batch_size}")
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

    # Load encoder ONCE (for encoding existing heuristics and new programs)
    print("Loading encoder (BAAI/bge-code-v1)...")
    encoder_model = get_encoder_model(device)
    encoder_model.eval()  # Set to eval mode

    # Load evaluator
    print(f"Loading evaluator for {task_name}...")
    evaluator = get_evaluator(task_name)
    secure_eval = SecureEvaluator(evaluator, debug_mode=False)

    # Get skeleton prompt
    skeleton_prompt = TASK_PROMPTS.get(task_name)
    if skeleton_prompt is None:
        raise ValueError(f"No prompt defined for task: {task_name}")

    # ===== Load and Evaluate Initial Baselines =====

    print(f"\nLoading baseline heuristics from task/{task_name}/heuristics.json...")
    heuristics = load_heuristics(task_name)
    baseline_codes = list(heuristics.values())

    if len(baseline_codes) == 0:
        raise ValueError(f"No heuristics found in task/{task_name}/heuristics.json")

    print(f"  Loaded {len(baseline_codes)} baseline heuristics")

    # Initialize program database: {code_hash: {'code', 'actual_score', 'embedding', 'source'}}
    program_database = {}

    # Evaluate ALL baselines to get ACTUAL scores (not predicted!) - PARALLEL
    print(f"\nEvaluating baseline heuristics on actual test instances (parallel, {num_evaluators} workers)...")
    print("(This may take a few minutes but only runs once)")

    def eval_baseline(idx_code):
        """Evaluate a single baseline heuristic."""
        idx, code = idx_code
        try:
            score = secure_eval.evaluate_program(code)
            return idx, code, score, None
        except Exception as e:
            return idx, code, None, str(e)

    # Submit all baseline evaluations in parallel
    baseline_results = []
    with ThreadPoolExecutor(max_workers=num_evaluators) as executor:
        futures = {executor.submit(eval_baseline, (idx, code)): idx
                   for idx, code in enumerate(baseline_codes)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating baselines"):
            baseline_results.append(future.result())

    # Process results and encode (encoding stays sequential - GPU bound)
    valid_baselines = [(idx, code, score) for idx, code, score, error in baseline_results
                       if score is not None and np.isfinite(score)]

    print(f"  Valid baselines: {len(valid_baselines)}/{len(baseline_codes)}")
    print("  Encoding valid baselines...")

    for idx, code, score in tqdm(valid_baselines, desc="Encoding baselines"):
        try:
            # Encode
            with torch.no_grad():
                z_embedding = encoder_model.encode(
                    [code],
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False
                )[0].float()

            # Add to database
            code_hash = hash(code.strip())
            program_database[code_hash] = {
                'code': code,
                'actual_score': float(score),
                'embedding': z_embedding,
                'source': 'baseline',
                'iteration': -1
            }

        except Exception as e:
            if verbose:
                print(f"  Baseline {idx+1}: Encoding error - {e}")
            continue

    if len(program_database) == 0:
        raise ValueError("No valid baseline programs after evaluation!")

    print(f"\n✓ Successfully evaluated {len(program_database)} baseline programs")

    # Sort by actual score to show statistics (DESCENDING - higher is better!)
    sorted_baselines = sorted(program_database.values(), key=lambda x: x['actual_score'], reverse=True)
    print(f"  Best baseline score: {sorted_baselines[0]['actual_score']:.4f}")
    print(f"  Worst baseline score: {sorted_baselines[-1]['actual_score']:.4f}")
    print(f"  Mean baseline score: {np.mean([p['actual_score'] for p in program_database.values()]):.4f}")

    # ===== Run Search =====

    all_results = []
    successful_programs = {}

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        # Sort ALL programs (baselines + discoveries) by ACTUAL score
        # Higher score is better (evaluators return negative costs)
        sorted_programs = sorted(
            program_database.values(),
            key=lambda x: x['actual_score'],
            reverse=True  # DESCENDING - higher (less negative) is better
        )

        # Create top-k pool and sample seeds based on softmax probabilities
        # This adds stochasticity to avoid getting stuck in same neighborhoods
        # while favoring better programs (higher scores have higher selection probability)
        pool_size = min(top_k_pool_size, len(sorted_programs))
        top_k_pool = sorted_programs[:pool_size]

        num_seeds = min(num_searches_per_iter, pool_size)

        # Compute softmax probabilities based on scores
        # Higher score (less negative) = higher probability
        pool_scores = np.array([p['actual_score'] for p in top_k_pool])

        # Apply softmax: exp(score) / sum(exp(score))
        # Use temperature=1.0 for standard softmax (can be tuned for more/less exploration)
        exp_scores = np.exp(pool_scores - np.max(pool_scores))  # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)

        # Sample from top-k pool using softmax probabilities (no replacement)
        sampled_indices = np.random.choice(pool_size, num_seeds, replace=False, p=probabilities)
        top_k_programs = [top_k_pool[i] for i in sampled_indices]

        # Extract embeddings from sampled programs
        init_embeddings = torch.stack([p['embedding'] for p in top_k_programs])

        # Get score range and probabilities for logging
        sampled_scores = [top_k_programs[i]['actual_score'] for i in range(num_seeds)]
        sampled_probs = [probabilities[i] for i in sampled_indices]
        best_sampled = max(sampled_scores)
        worst_sampled = min(sampled_scores)

        print(f"\nStarting gradient search by sampling {num_seeds} from top-{pool_size} programs (softmax-weighted):")
        print(f"  Best score in database: {sorted_programs[0]['actual_score']:.4f} ({sorted_programs[0]['source']})")
        print(f"  Best in sampled seeds: {best_sampled:.4f} (P={max(sampled_probs):.3f})")
        print(f"  Worst in sampled seeds: {worst_sampled:.4f} (P={min(sampled_probs):.3f})")
        print(f"  Database size: {len(program_database)} programs")

        # Gradient search in z-space from sampled seeds
        print(f"\nRunning gradient search in z-space ({num_seeds} starts, {gradient_steps} steps)...")

        optimized_z = multi_start_gradient_search(
            predictor=predictor,
            num_starts=num_seeds,
            steps=gradient_steps,
            lr=lr,
            trust_region_lambda=trust_region_lambda,
            init_from_data=init_embeddings,
            device=device,
            verbose=verbose
        )

        # ===== BATCHED GENERATION =====
        print(f"\nGenerating {len(optimized_z)} candidates in batches (batch_size={generation_batch_size})...")

        all_generated = []
        for batch_start in range(0, len(optimized_z), generation_batch_size):
            batch_end = min(batch_start + generation_batch_size, len(optimized_z))
            z_batch = optimized_z[batch_start:batch_end]

            batch_results = generate_code_from_z_batch(
                z_batch,
                mapper_model,
                decoder_model,
                decoder_tokenizer,
                skeleton_prompt,
                temperature=temperature
            )
            all_generated.extend(batch_results)

            if verbose:
                print(f"  Generated batch {batch_start//generation_batch_size + 1}/{(len(optimized_z) + generation_batch_size - 1)//generation_batch_size}")

        # ===== VALIDATE AND FILTER =====
        print(f"Validating {len(all_generated)} generated programs...")

        valid_candidates = []
        existing_hashes = set(hash(c.strip()) for c in successful_programs.values())

        for idx, (clean_code, raw_output) in enumerate(all_generated):
            # Validate syntax
            if not clean_code or not is_valid_python(clean_code):
                continue

            # Parse function
            program = TextFunctionProgramConverter.text_to_program(clean_code)
            if program is None or len(program.functions) == 0:
                continue

            func_name = program.functions[0].name

            # Check for duplicates
            code_hash = hash(clean_code.strip())
            if code_hash in existing_hashes:
                continue

            existing_hashes.add(code_hash)
            valid_candidates.append((idx, clean_code, func_name))

        print(f"  Valid candidates: {len(valid_candidates)}/{len(all_generated)}")

        # ===== PARALLEL EVALUATION =====
        print(f"Evaluating {len(valid_candidates)} candidates in parallel ({num_evaluators} workers)...")

        def eval_candidate(item):
            """Evaluate a single candidate."""
            idx, clean_code, func_name = item
            try:
                score = secure_eval.evaluate_program(clean_code)
                return idx, clean_code, func_name, score, None
            except Exception as e:
                return idx, clean_code, func_name, None, str(e)

        eval_results = []
        with ThreadPoolExecutor(max_workers=num_evaluators) as executor:
            futures = {executor.submit(eval_candidate, item): item[0]
                       for item in valid_candidates}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating candidates"):
                eval_results.append(future.result())

        # ===== PROCESS RESULTS =====
        successful = 0
        failed = len(all_generated) - len(valid_candidates)  # Already failed validation

        for idx, clean_code, func_name, score, error in eval_results:
            if score is None or not np.isfinite(score):
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

            # Add to program database with actual score
            code_hash_new = hash(clean_code.strip())
            if code_hash_new not in program_database:
                # Encode the new program
                with torch.no_grad():
                    z_new = encoder_model.encode(
                        [clean_code],
                        convert_to_tensor=True,
                        device=device,
                        show_progress_bar=False
                    )[0].float()

                program_database[code_hash_new] = {
                    'code': clean_code,
                    'actual_score': float(score),
                    'embedding': z_new,
                    'source': 'discovered',
                    'iteration': iteration
                }

            successful += 1

            if verbose:
                print(f"  ✓ Score: {score:.4f} | Function: {func_name}")

        # Iteration summary
        print(f"\nIteration {iteration + 1} Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total discovered: {len(successful_programs)}")

        if all_results:
            scores = [r['score'] for r in all_results]
            print(f"  Best score so far: {max(scores):.4f}")

        # Show best score in entire database
        db_scores = [p['actual_score'] for p in program_database.values()]
        print(f"  Best score in database: {max(db_scores):.4f}")
        print(f"  Database size: {len(program_database)} programs")

    # ===== Final Results =====

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal programs discovered: {len(successful_programs)}")
    print(f"Total programs in database: {len(program_database)}")

    # Database statistics
    db_programs = list(program_database.values())
    db_programs.sort(key=lambda x: x['actual_score'], reverse=True)

    if db_programs:
        db_scores = [p['actual_score'] for p in db_programs]
        print(f"\nDatabase Statistics:")
        print(f"  Best score: {max(db_scores):.4f}")
        print(f"  Worst score: {min(db_scores):.4f}")
        print(f"  Mean score: {np.mean(db_scores):.4f}")
        print(f"  Std score: {np.std(db_scores):.4f}")

    if all_results:
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)

        scores = [r['score'] for r in all_results]
        print(f"\nDiscovered Programs Statistics:")
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Avg score: {np.mean(scores):.4f}")
        print(f"  Std score: {np.std(scores):.4f}")

        # Show top 5
        print(f"\nTop 5 Discovered Programs:")
        print("-"*70)
        for i, result in enumerate(all_results[:5]):
            print(f"\n{i+1}. Score: {result['score']:.4f} | {result['function_name']}")
            print(f"   ID: {result['program_id']}")
            code_preview = result['code'][:200].replace('\n', '\n   ')
            print(f"   {code_preview}...")

    # ===== Save Results =====

    # Save discovered programs as JSON
    programs_path = os.path.join(output_dir, f"{task_name}_gradient_searched.json")
    with open(programs_path, 'w') as f:
        json.dump(successful_programs, f, indent=2)
    print(f"\nSaved {len(successful_programs)} discovered programs to {programs_path}")

    # Save detailed results
    results_path = os.path.join(output_dir, f"{task_name}_search_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed results to {results_path}")

    # Save full database (baselines + discoveries) with actual scores
    database_export = []
    for prog in db_programs:
        database_export.append({
            'code': prog['code'],
            'actual_score': prog['actual_score'],
            'source': prog['source'],
            'iteration': prog['iteration']
        })

    database_path = os.path.join(output_dir, f"{task_name}_full_database.json")
    with open(database_path, 'w') as f:
        json.dump(database_export, f, indent=2)
    print(f"Saved full database ({len(database_export)} programs) to {database_path}")

    # Clean up encoder
    del encoder_model
    torch.cuda.empty_cache()

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
    parser.add_argument('--num_searches', type=int, default=10, help='Searches per iteration (how many seeds to sample)')
    parser.add_argument('--top_k_pool', type=int, default=30, help='Pool size for softmax-weighted sampling (e.g., 30 = sample from top 30 programs using softmax probabilities)')
    parser.add_argument('--gradient_steps', type=int, default=100, help='Gradient steps per search')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--trust_region_lambda', type=float, default=0.0, help='Trust region penalty coefficient (0 = disabled)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Decoder temperature')
    parser.add_argument('--output_dir', type=str, default='gradient_search_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_evaluators', type=int, default=4, help='Number of parallel evaluation workers')
    parser.add_argument('--generation_batch_size', type=int, default=4, help='Batch size for code generation (adjust based on GPU memory)')

    args = parser.parse_args()

    results, programs = gradient_search_pipeline(
        task_name=args.task,
        predictor_path=args.predictor,
        mapper_path=args.mapper,
        num_iterations=args.num_iterations,
        num_searches_per_iter=args.num_searches,
        top_k_pool_size=args.top_k_pool,
        gradient_steps=args.gradient_steps,
        lr=args.lr,
        trust_region_lambda=args.trust_region_lambda,
        temperature=args.temperature,
        decoder_name=args.decoder,
        device=args.device,
        output_dir=args.output_dir,
        num_evaluators=args.num_evaluators,
        generation_batch_size=args.generation_batch_size,
        verbose=True
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
