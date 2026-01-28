"""
Gradient-based search in u-space (prior space) using z-space trained ranking predictor.

HYBRID APPROACH:
- Predictor R(z) trained on z-space (from ranking_score_predictor_z.py)
- Gradient search in u-space through the composition: R(F^(-1)(u))
- Benefits: simpler predictor training (no flow needed) + manifold-constrained search

Pipeline:
1. Load trained ranking predictor R(z) (z-space predictor)
2. Load normalizing flow F
3. Gradient ascent in u-space: u* = argmax [R(F^(-1)(u)) - lambda||u - u0||^2]
   - The gradient flows through: u -> z (flow inverse) -> score (predictor)
4. Map u* -> z* using flow inverse F^(-1)(u)
5. Generate code using mapper + decoder
6. Evaluate and save successful programs

Key insight: Even though the predictor was trained on z-vectors, we can optimize
in u-space by chaining the flow inverse with the predictor. This keeps the search
constrained to the learned manifold while using a simpler z-space predictor.

Uses same models as train_unified_mapper_optimized.py and train_unified_flow.py:
- Encoder: BAAI/bge-code-v1
- Decoder: Qwen/Qwen3-4B-Instruct-2507 (with Flash Attention 2)
- Flow: Unified normalizing flow trained on all tasks
- Predictor: Z-space ranking predictor (NOT u-space!)
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
from normalizing_flow import NormalizingFlow
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
    """Load decoder model and tokenizer."""
    print(f"Loading decoder: {model_name}...")

    decoder_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
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

    # Handle torch.compile() prefix
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


def load_flow(flow_path: str, device: str = "cuda"):
    """Load trained normalizing flow."""
    print(f"Loading normalizing flow from {flow_path}...")

    checkpoint = torch.load(flow_path, map_location=device)

    # Extract architecture parameters
    dim = checkpoint.get('dim', checkpoint.get('embedding_dim', 768))
    num_layers = checkpoint.get('num_layers', 4)
    hidden_dim = checkpoint.get('hidden_dim', 128)
    dropout = checkpoint.get('dropout', 0.0)

    print(f"  Dimension: {dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout} (disabled in eval mode)")

    # Create flow model
    flow_model = NormalizingFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    flow_model.to(device)
    flow_model.eval()

    print(f"  Flow loaded")
    return flow_model


# ============================================================================
# Gradient Search in U-Space with Z-Space Predictor
# ============================================================================

def gradient_ascent_u_with_z_predictor(
    predictor: RankingScorePredictor,
    flow_model: NormalizingFlow,
    init_u: torch.Tensor,
    steps: int = 100,
    lr: float = 0.01,
    trust_region_lambda: float = 0.0,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Gradient ascent in u-space using z-space trained predictor.

    KEY DIFFERENCE from gradient_search_u.py:
    - gradient_search_u.py: R(u) - predictor takes u directly
    - This function: R(F^(-1)(u)) - predictor takes z, we optimize through flow inverse

    Maximizes: R(F^(-1)(u)) - lambda||u - u0||^2

    The gradient computation:
    d/du [R(F^(-1)(u))] = dR/dz * dF^(-1)/du

    This is computed automatically via autograd since flow.inverse() is differentiable.

    Args:
        predictor: Trained ranking score predictor R(z) (z-space!)
        flow_model: Trained normalizing flow F
        init_u: Initial u vectors [num_starts, dim]
        steps: Number of gradient ascent steps
        lr: Learning rate
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized u vectors [num_starts, dim]
    """
    predictor.eval()
    flow_model.eval()

    u = init_u.clone().to(device).requires_grad_(True)
    u_init = init_u.clone().to(device)  # Keep initial u for trust region
    num_samples = u.shape[0]

    trust_msg = f", trust_region_lambda={trust_region_lambda}" if trust_region_lambda > 0 else ""
    if verbose:
        print(f"Gradient ascent in u-space (with z-predictor): {num_samples} starts, {steps} steps, lr={lr}{trust_msg}")

    for step in range(steps):
        if u.grad is not None:
            u.grad.zero_()

        # Map u -> z through flow inverse (this is differentiable!)
        z = flow_model.inverse(u)

        # Predict score R(z) using z-space predictor
        scores = predictor(z)

        # Maximize score (negative for gradient ascent)
        loss = -scores.mean()

        # Add trust region penalty if enabled
        if trust_region_lambda > 0:
            trust_region_penalty = trust_region_lambda * torch.mean((u - u_init) ** 2)
            loss = loss + trust_region_penalty

        loss.backward()

        # Gradient ascent update
        with torch.no_grad():
            u += lr * u.grad

        if verbose and (step + 1) % 20 == 0:
            avg_score = -loss.item()
            if trust_region_lambda > 0:
                with torch.no_grad():
                    distance = torch.mean(torch.sqrt(torch.sum((u - u_init) ** 2, dim=1))).item()
                print(f"  Step {step+1}/{steps} | Avg score: {avg_score:.4f} | Dist from init: {distance:.4f}")
            else:
                print(f"  Step {step+1}/{steps} | Avg score: {avg_score:.4f}")

    return u.detach()


def multi_start_gradient_search_u_with_z_predictor(
    predictor: RankingScorePredictor,
    flow_model: NormalizingFlow,
    num_starts: int = 10,
    steps: int = 100,
    lr: float = 0.01,
    trust_region_lambda: float = 0.0,
    init_from_data: torch.Tensor = None,
    device: str = 'cuda',
    verbose: bool = True
) -> torch.Tensor:
    """
    Multi-start gradient search in u-space using z-space predictor.

    Args:
        predictor: Trained ranking predictor R(z) (z-space!)
        flow_model: Trained normalizing flow
        num_starts: Number of parallel searches
        steps: Gradient steps per search
        lr: Learning rate
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        init_from_data: Tensor of existing z vectors to initialize from (required)
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized z vectors [num_starts, dim] (mapped back from u-space)
    """
    if init_from_data is None or len(init_from_data) == 0:
        raise ValueError("init_from_data is required - must provide existing z vectors")

    # Initialize from top data points
    n_available = len(init_from_data)
    n_to_use = min(num_starts, n_available)

    # Take top-n embeddings
    init_z = init_from_data[:n_to_use].clone().to(device)

    # Map z -> u using flow forward
    with torch.no_grad():
        init_u, _ = flow_model(init_z)

    if verbose:
        print(f"Initialized {n_to_use} searches from top-{n_to_use} embeddings")
        print(f"  Mapped z -> u (prior space)")
        print(f"  Using z-space predictor with flow inverse for gradient computation")

    # Run gradient ascent in u-space with z-space predictor
    optimized_u = gradient_ascent_u_with_z_predictor(
        predictor=predictor,
        flow_model=flow_model,
        init_u=init_u,
        steps=steps,
        lr=lr,
        trust_region_lambda=trust_region_lambda,
        device=device,
        verbose=verbose
    )

    # Map optimized u -> z using flow inverse (for decoding and scoring)
    with torch.no_grad():
        optimized_z = flow_model.inverse(optimized_u)

    # Get final scores using z-predictor
    with torch.no_grad():
        final_scores = predictor(optimized_z).squeeze()

    if verbose:
        print(f"\nFinal scores (from z-predictor):")
        print(f"  Min:  {final_scores.min().item():.4f}")
        print(f"  Max:  {final_scores.max().item():.4f}")
        print(f"  Mean: {final_scores.mean().item():.4f}")

    # Sort by score (descending)
    sorted_indices = torch.argsort(final_scores, descending=True)
    optimized_z = optimized_z[sorted_indices]

    return optimized_z


# ============================================================================
# Code Generation (Same as other versions)
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
    """Generate code from multiple latent vectors in a single batch."""
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
        # Batch map z -> soft prompts
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
        instruction_embeds = embed_layer(instruction_ids)
        instruction_embeds = instruction_embeds.expand(batch_size, -1, -1)

        # Concatenate
        inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)

        # Create attention mask
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

def gradient_search_pipeline_u_with_z_predictor(
    task_name: str,
    predictor_path: str,
    flow_path: str,
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
    output_dir: str = "gradient_search_u_z_pred_results",
    num_evaluators: int = 4,
    generation_batch_size: int = 4,
    verbose: bool = True
):
    """
    Full gradient search pipeline in u-space using z-space predictor.

    HYBRID APPROACH:
    - Predictor R(z) trained on z-space (simpler training, no flow needed)
    - Gradient search in u-space through R(F^(-1)(u)) (manifold-constrained)

    Key Features:
    1. Evaluates ALL baseline heuristics on ACTUAL test instances
    2. Maintains database of all programs with actual scores
    3. Each iteration: sorts by actual score, samples from top-k using softmax
    4. Gradient flows through: u -> z (flow inverse) -> score (z-predictor)
    5. Batched code generation and parallel evaluation

    Args:
        task_name: Task to search for (e.g., 'tsp_construct')
        predictor_path: Path to trained z-space ranking predictor
        flow_path: Path to trained normalizing flow
        mapper_path: Path to trained mapper
        num_iterations: Number of search iterations
        num_searches_per_iter: Number of gradient searches per iteration
        top_k_pool_size: Pool size for softmax-weighted sampling
        gradient_steps: Steps per gradient search
        lr: Learning rate for gradient ascent
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        temperature: Sampling temperature for decoder
        decoder_name: Decoder model name
        device: Device to use
        output_dir: Directory to save results
        num_evaluators: Number of parallel evaluation workers
        generation_batch_size: Batch size for code generation
        verbose: Print progress
    """

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Gradient Search in U-Space with Z-Space Predictor (Hybrid)")
    print("="*70)
    print(f"Task: {task_name}")
    print(f"Predictor (z-space): {predictor_path}")
    print(f"Flow: {flow_path}")
    print(f"Mapper: {mapper_path}")
    print(f"Parallel evaluators: {num_evaluators}")
    print(f"Generation batch size: {generation_batch_size}")
    print()
    print("Hybrid approach: R(z) predictor + u-space optimization via R(F^(-1)(u))")
    print()

    # ===== Load Models =====

    # Load ranking predictor (z-space!)
    print("Loading z-space ranking predictor...")
    predictor, predictor_info = load_ranking_predictor(predictor_path, device)

    # Verify it's a z-space predictor
    predictor_space = predictor_info.get('space', 'z')
    if predictor_space != 'z':
        print(f"  WARNING: Predictor was trained on '{predictor_space}'-space, expected 'z'-space")
        print(f"           This may work but is not the intended use case")
    else:
        print(f"  Confirmed z-space predictor")

    predictor.to(device)
    predictor.eval()

    # Load normalizing flow
    flow_model = load_flow(flow_path, device)

    # Load decoder
    decoder_model, decoder_tokenizer = load_decoder(decoder_name, device)

    # Load mapper
    mapper_model = load_mapper(mapper_path, decoder_model, device)
    embed_layer = decoder_model.get_input_embeddings()
    mapper_model = mapper_model.to(embed_layer.weight.device)

    # Load encoder
    print("Loading encoder (BAAI/bge-code-v1)...")
    encoder_model = get_encoder_model(device)
    encoder_model.eval()

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

    # Initialize program database
    program_database = {}

    # Evaluate baselines in parallel
    print(f"\nEvaluating baseline heuristics (parallel, {num_evaluators} workers)...")

    def eval_baseline(idx_code):
        idx, code = idx_code
        try:
            score = secure_eval.evaluate_program(code)
            return idx, code, score, None
        except Exception as e:
            return idx, code, None, str(e)

    baseline_results = []
    with ThreadPoolExecutor(max_workers=num_evaluators) as executor:
        futures = {executor.submit(eval_baseline, (idx, code)): idx
                   for idx, code in enumerate(baseline_codes)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating baselines"):
            baseline_results.append(future.result())

    # Process results and encode
    valid_baselines = [(idx, code, score) for idx, code, score, error in baseline_results
                       if score is not None and np.isfinite(score)]

    print(f"  Valid baselines: {len(valid_baselines)}/{len(baseline_codes)}")
    print("  Encoding valid baselines...")

    for idx, code, score in tqdm(valid_baselines, desc="Encoding baselines"):
        try:
            with torch.no_grad():
                z_embedding = encoder_model.encode(
                    [code],
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False
                )[0].float()

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

    print(f"\nSuccessfully evaluated {len(program_database)} baseline programs")

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

        # Sort by actual score (descending)
        sorted_programs = sorted(
            program_database.values(),
            key=lambda x: x['actual_score'],
            reverse=True
        )

        # Softmax-weighted sampling from top-k pool
        pool_size = min(top_k_pool_size, len(sorted_programs))
        top_k_pool = sorted_programs[:pool_size]
        num_seeds = min(num_searches_per_iter, pool_size)

        pool_scores = np.array([p['actual_score'] for p in top_k_pool])
        exp_scores = np.exp(pool_scores - np.max(pool_scores))
        probabilities = exp_scores / np.sum(exp_scores)

        sampled_indices = np.random.choice(pool_size, num_seeds, replace=False, p=probabilities)
        top_k_programs = [top_k_pool[i] for i in sampled_indices]

        init_embeddings = torch.stack([p['embedding'] for p in top_k_programs])

        sampled_scores = [top_k_programs[i]['actual_score'] for i in range(num_seeds)]
        sampled_probs = [probabilities[i] for i in sampled_indices]

        print(f"\nStarting gradient search (u-space with z-predictor):")
        print(f"  Sampling {num_seeds} seeds from top-{pool_size} (softmax-weighted)")
        print(f"  Best in database: {sorted_programs[0]['actual_score']:.4f} ({sorted_programs[0]['source']})")
        print(f"  Best in seeds: {max(sampled_scores):.4f}")
        print(f"  Database size: {len(program_database)} programs")

        # Gradient search in u-space with z-predictor
        print(f"\nRunning gradient search ({num_seeds} starts, {gradient_steps} steps)...")

        optimized_z = multi_start_gradient_search_u_with_z_predictor(
            predictor=predictor,
            flow_model=flow_model,
            num_starts=num_seeds,
            steps=gradient_steps,
            lr=lr,
            trust_region_lambda=trust_region_lambda,
            init_from_data=init_embeddings,
            device=device,
            verbose=verbose
        )

        # Batched generation
        print(f"\nGenerating {len(optimized_z)} candidates (batch_size={generation_batch_size})...")

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

        # Validate
        print(f"Validating {len(all_generated)} generated programs...")

        valid_candidates = []
        existing_hashes = set(hash(c.strip()) for c in successful_programs.values())

        for idx, (clean_code, raw_output) in enumerate(all_generated):
            if not clean_code or not is_valid_python(clean_code):
                continue

            program = TextFunctionProgramConverter.text_to_program(clean_code)
            if program is None or len(program.functions) == 0:
                continue

            func_name = program.functions[0].name
            code_hash = hash(clean_code.strip())
            if code_hash in existing_hashes:
                continue

            existing_hashes.add(code_hash)
            valid_candidates.append((idx, clean_code, func_name))

        print(f"  Valid candidates: {len(valid_candidates)}/{len(all_generated)}")

        # Parallel evaluation
        print(f"Evaluating {len(valid_candidates)} candidates ({num_evaluators} workers)...")

        def eval_candidate(item):
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

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                eval_results.append(future.result())

        # Process results
        successful = 0
        failed = len(all_generated) - len(valid_candidates)

        for idx, clean_code, func_name, score, error in eval_results:
            if score is None or not np.isfinite(score):
                failed += 1
                continue

            program_id = f"iter{iteration}_idx{idx}"
            successful_programs[program_id] = clean_code

            code_hash_new = hash(clean_code.strip())
            if code_hash_new not in program_database:
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

        # Iteration summary
        print(f"\nIteration {iteration + 1} Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total in database: {len(program_database)}")
        print(f"  Total discovered: {len(successful_programs)}")

        if all_results:
            scores = [r['score'] for r in all_results]
            print(f"  Best discovered: {max(scores):.4f}")

        db_scores = [p['actual_score'] for p in program_database.values()]
        print(f"  Best in database: {max(db_scores):.4f}")

    # ===== Final Results =====

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal programs in database: {len(program_database)}")
    print(f"  Baselines: {sum(1 for p in program_database.values() if p['source']=='baseline')}")
    print(f"  Discovered: {sum(1 for p in program_database.values() if p['source']=='discovered')}")

    db_programs = list(program_database.values())
    db_programs.sort(key=lambda x: x['actual_score'], reverse=True)

    print(f"\nDatabase Statistics:")
    print(f"  Best: {db_programs[0]['actual_score']:.4f} ({db_programs[0]['source']}, iter {db_programs[0]['iteration']})")
    print(f"  Worst: {db_programs[-1]['actual_score']:.4f}")
    print(f"  Mean: {np.mean([p['actual_score'] for p in db_programs]):.4f}")

    if all_results:
        all_results.sort(key=lambda x: x['score'], reverse=True)
        scores = [r['score'] for r in all_results]
        print(f"\nDiscovered Programs:")
        print(f"  Count: {len(all_results)}")
        print(f"  Best: {max(scores):.4f}")
        print(f"  Avg: {np.mean(scores):.4f}")

        print(f"\nTop 5 Discovered:")
        print("-"*70)
        for i, result in enumerate(all_results[:5]):
            print(f"\n{i+1}. Score: {result['score']:.4f} | {result['function_name']}")
            code_preview = result['code'][:200].replace('\n', '\n   ')
            print(f"   {code_preview}...")

    # ===== Save Results =====

    programs_path = os.path.join(output_dir, f"{task_name}_gradient_searched_u_z_pred.json")
    with open(programs_path, 'w') as f:
        json.dump(successful_programs, f, indent=2)
    print(f"\nSaved {len(successful_programs)} discovered programs to {programs_path}")

    results_path = os.path.join(output_dir, f"{task_name}_search_results_u_z_pred.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed results to {results_path}")

    database_export = {}
    for code_hash, prog_data in program_database.items():
        if prog_data['source'] == 'baseline':
            name = f"baseline_{code_hash}"
        else:
            name = f"discovered_iter{prog_data['iteration']}_{code_hash}"

        database_export[name] = {
            'code': prog_data['code'],
            'actual_score': prog_data['actual_score'],
            'source': prog_data['source'],
            'iteration': prog_data['iteration']
        }

    database_path = os.path.join(output_dir, f"{task_name}_full_database_u_z_pred.json")
    with open(database_path, 'w') as f:
        json.dump(database_export, f, indent=2)
    print(f"Saved full database ({len(database_export)} programs) to {database_path}")

    # Cleanup
    del encoder_model
    torch.cuda.empty_cache()

    return all_results, successful_programs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gradient search in u-space with z-space predictor (Hybrid approach)'
    )
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--predictor', type=str, default='ranking_predictor_z.pth',
                        help='Path to z-space ranking predictor (NOT u-space!)')
    parser.add_argument('--flow', type=str, default='Flow_Checkpoints/unified_flow_final.pth',
                        help='Path to normalizing flow')
    parser.add_argument('--mapper', type=str, default='Mapper_Checkpoints/unified_mapper.pth',
                        help='Path to mapper')
    parser.add_argument('--decoder', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                        help='Decoder model')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of search iterations')
    parser.add_argument('--num_searches', type=int, default=10,
                        help='Searches per iteration')
    parser.add_argument('--top_k_pool', type=int, default=30,
                        help='Pool size for softmax-weighted sampling')
    parser.add_argument('--gradient_steps', type=int, default=100,
                        help='Gradient steps per search')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--trust_region_lambda', type=float, default=0.0,
                        help='Trust region penalty coefficient (0 = disabled)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Decoder temperature')
    parser.add_argument('--output_dir', type=str, default='gradient_search_u_z_pred_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--num_evaluators', type=int, default=4,
                        help='Number of parallel evaluation workers')
    parser.add_argument('--generation_batch_size', type=int, default=4,
                        help='Batch size for code generation')

    args = parser.parse_args()

    results, programs = gradient_search_pipeline_u_with_z_predictor(
        task_name=args.task,
        predictor_path=args.predictor,
        flow_path=args.flow,
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
