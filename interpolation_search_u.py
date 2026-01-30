"""
Interpolation-based search in u-space (prior space) using normalizing flow.

Simple crossover-style search without learned predictor:
1. Load trained normalizing flow F
2. For each search: sample 2 parents independently from top-10 programs
3. Map to u-space: u1 = F(z1), u2 = F(z2)
4. Interpolate: u_child = 0.5 * u1 + 0.5 * u2
5. Map back: z_child = F^(-1)(u_child)
6. Generate code using mapper + decoder
7. Evaluate and save successful programs

Advantages:
- No predictor training needed - simpler and faster
- Pure exploration via random parent sampling
- Guaranteed to stay on learned manifold via flow inverse
- Good for discovering novel combinations

Uses same models as gradient_search_u.py except NO predictor needed:
- Encoder: BAAI/bge-code-v1
- Decoder: Qwen/Qwen3-4B-Instruct-2507
- Flow: Unified normalizing flow trained on all tasks
- Mapper: Trained mapper for code generation
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

from mapper import Mapper, OriginalMapper, LowRankMapper
from normalizing_flow import NormalizingFlow
from model_config import DEFAULT_ENCODER, DEFAULT_DECODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder
from utils import extract_python_code_robust, is_valid_python
from base.code import TextFunctionProgramConverter
from base.evaluate import SecureEvaluator


# ============================================================================
# Task Evaluator Loading (copied from gradient_search_u.py)
# ============================================================================

def get_evaluator(task_name: str):
    """Get the evaluator class for a given task."""
    task_evaluators = {
        'tsp_construct': ('task.tsp_construct.evaluation', 'TSPEvaluation'),
        'cvrp_construct': ('task.cvrp_construct.evaluation', 'CVRPEvaluation'),
        'vrptw_construct': ('task.vrptw_construct.evaluation', 'VRPTWEvaluation'),
        'jssp_construct': ('task.jssp_construct.evaluation', 'JSSPEvaluation'),
        'knapsack_construct': ('task.knapsack_construct.evaluation', 'KnapsackEvaluation'),
        'online_bin_packing': ('task.online_bin_packing.evaluation', 'OBPEvaluation'),
        'qap_construct': ('task.qap_construct.evaluation', 'QAPEvaluation'),
        'set_cover_construct': ('task.set_cover_construct.evaluation', 'SCPEvaluation'),
        'cflp_construct': ('task.cflp_construct.evaluation', 'CFLPEvaluation'),
        'admissible_set': ('task.admissible_set.evaluation', 'ASPEvaluation'),
    }

    if task_name not in task_evaluators:
        raise ValueError(f"Unknown task: {task_name}")

    module_path, class_name = task_evaluators[task_name]
    module = __import__(module_path, fromlist=[class_name])
    EvaluatorClass = getattr(module, class_name)
    return EvaluatorClass()


def load_heuristics(task_name: str) -> dict:
    """Load heuristics from JSON file for a given task."""
    heuristics_path = Path(f"task/{task_name}/heuristics.json")

    if not heuristics_path.exists():
        raise FileNotFoundError(f"Heuristics file not found: {heuristics_path}")

    with open(heuristics_path, 'r') as f:
        programs = json.load(f)

    return programs


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

def load_decoder(model_name: str = None):
    """Load decoder model and tokenizer."""
    if model_name is None:
        model_name = DEFAULT_DECODER
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

    print(f"  ✓ Decoder loaded with Flash Attention 2.")
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
        internal_dim = checkpoint.get('internal_dim')
        mapper_type = checkpoint.get('mapper_type', None)
        attn_heads = checkpoint.get('attn_heads', 2)
        attn_dropout = checkpoint.get('attn_dropout', 0.1)
        ffn_dropout = checkpoint.get('ffn_dropout', 0.1)
        scale = checkpoint.get('scale', 0.1)
        use_ffn = checkpoint.get('use_ffn', True)
    else:
        state_dict = checkpoint
        input_dim = None
        output_dim = None
        num_tokens = None
        internal_dim = None
        mapper_type = None
        attn_heads = 2
        attn_dropout = 0.1
        ffn_dropout = 0.1
        scale = 0.1
        use_ffn = True

    # Handle torch.compile() prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Detected torch.compile() checkpoint, removing '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Detect mapper type from state_dict keys
    if mapper_type is None:
        if 'feature_expander.weight' in state_dict:
            mapper_type = 'LowRankMapper'
        elif 'mlp.0.weight' in state_dict:
            mapper_type = 'OriginalMapper'
        else:
            raise ValueError("Cannot detect mapper type from checkpoint keys")

    print(f"  Mapper type: {mapper_type}")

    # Infer dimensions based on mapper type
    if mapper_type == 'LowRankMapper':
        if input_dim is None:
            input_dim = state_dict['feature_expander.weight'].shape[1]
        if output_dim is None:
            if 'shared_mlp.3.weight' in state_dict:
                output_dim = state_dict['shared_mlp.3.weight'].shape[0]
            else:
                output_dim = state_dict['shared_mlp.2.weight'].shape[0]
        if num_tokens is None:
            num_tokens = state_dict['pos_embed'].shape[1]
        if internal_dim is None:
            internal_dim = state_dict['shared_mlp.0.weight'].shape[0]

        has_attention = 'ln_attn.weight' in state_dict
        if has_attention:
            use_ffn = 'ln_ffn.weight' in state_dict

        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num tokens: {num_tokens}")
        print(f"  Internal dim: {internal_dim}")

        mapper_model = LowRankMapper(
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            internal_dim=internal_dim,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            scale=scale,
            use_ffn=use_ffn
        )

    else:  # OriginalMapper
        if input_dim is None:
            input_dim = state_dict['mlp.0.weight'].shape[1]
        if output_dim is None:
            output_dim = decoder_model.config.hidden_size
        if num_tokens is None:
            final_output_size = state_dict['mlp.4.weight'].shape[0]
            num_tokens = final_output_size // output_dim

        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num tokens: {num_tokens}")

        mapper_model = OriginalMapper(input_dim, output_dim, num_tokens)

    mapper_model.load_state_dict(state_dict)
    mapper_model.eval()

    return mapper_model


def load_flow(flow_path: str, device: str = "cuda"):
    """Load trained normalizing flow."""
    print(f"Loading normalizing flow from {flow_path}...")

    checkpoint = torch.load(flow_path, map_location=device)

    dim = checkpoint.get('dim', checkpoint.get('embedding_dim', 768))
    num_layers = checkpoint.get('num_layers', 4)
    hidden_dim = checkpoint.get('hidden_dim', 128)
    dropout = checkpoint.get('dropout', 0.0)

    print(f"  Dimension: {dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")

    flow_model = NormalizingFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    flow_model.to(device)
    flow_model.eval()

    print(f"  ✓ Flow loaded")
    return flow_model


# ============================================================================
# Interpolation Search in U-Space
# ============================================================================

def interpolate_in_u_space(
    flow_model: NormalizingFlow,
    parent_embeddings: torch.Tensor,
    num_children: int,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Generate children by interpolating pairs of parents in u-space.

    For each child:
    1. Sample 2 parents independently from parent_embeddings
    2. Map to u-space: u1 = F(z1), u2 = F(z2)
    3. Interpolate with lambda=0.5: u_child = 0.5*u1 + 0.5*u2
    4. Map back: z_child = F^(-1)(u_child)

    Args:
        flow_model: Trained normalizing flow
        parent_embeddings: Tensor of parent z embeddings [num_parents, dim]
        num_children: Number of children to generate
        device: Device to use
        verbose: Print progress

    Returns:
        Children z embeddings [num_children, dim]
    """
    flow_model.eval()

    num_parents = parent_embeddings.shape[0]
    if num_parents < 2:
        raise ValueError("Need at least 2 parents for interpolation")

    parent_embeddings = parent_embeddings.to(device)

    if verbose:
        print(f"Generating {num_children} children via interpolation (lambda=0.5)")
        print(f"  Sampling from {num_parents} parents")

    children_z = []

    with torch.no_grad():
        # Map all parents to u-space once
        parents_u, _ = flow_model(parent_embeddings)

        for i in range(num_children):
            # Sample 2 parents independently
            idx1, idx2 = np.random.choice(num_parents, size=2, replace=False)

            u1 = parents_u[idx1]
            u2 = parents_u[idx2]

            # Interpolate with lambda=0.5 (midpoint)
            u_child = 0.5 * u1 + 0.5 * u2

            # Map back to z-space
            z_child = flow_model.inverse(u_child.unsqueeze(0))

            children_z.append(z_child.squeeze(0))

            if verbose and (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_children} children")

    return torch.stack(children_z)


# ============================================================================
# Code Generation (copied from gradient_search_u.py)
# ============================================================================

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

    if isinstance(z_vectors, np.ndarray):
        z_vectors = torch.from_numpy(z_vectors)

    if z_vectors.dim() == 1:
        z_vectors = z_vectors.unsqueeze(0)

    batch_size = z_vectors.shape[0]
    z_vectors = z_vectors.to(device).float()

    with torch.no_grad():
        # Batch map z -> soft prompts
        soft_prompt_embeds = mapper_model(z_vectors).to(device, dtype=dtype)

        # Prepare instruction
        instruction_messages = [{"role": "user", "content": skeleton_prompt}]
        instruction_ids = decoder_tokenizer.apply_chat_template(
            instruction_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        instruction_embeds = embed_layer(instruction_ids)
        instruction_embeds = instruction_embeds.expand(batch_size, -1, -1)

        # Concatenate
        inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)

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

def interpolation_search_pipeline(
    task_name: str,
    flow_path: str,
    mapper_path: str,
    num_iterations: int = 5,
    num_searches_per_iter: int = 5,
    top_k_pool_size: int = 10,
    temperature: float = 0.7,
    encoder_name: str = None,
    decoder_name: str = None,
    embedding_dim: int = None,
    device: str = "cuda",
    output_dir: str = "interpolation_search_results",
    num_evaluators: int = 4,
    generation_batch_size: int = 4,
    verbose: bool = True
):
    """
    Interpolation-based search pipeline in u-space without predictor.

    Simpler than gradient search:
    1. Load baselines and evaluate
    2. Each iteration: sample top-k, then for each search sample 2 parents and interpolate
    3. Generate and evaluate children
    4. Add successful programs to database

    Args:
        task_name: Task to search for
        flow_path: Path to trained normalizing flow
        mapper_path: Path to trained mapper
        num_iterations: Number of search iterations
        num_searches_per_iter: Number of children to generate per iteration
        top_k_pool_size: Pool size to sample parents from (default: 10)
        temperature: Sampling temperature for decoder
        encoder_name: Encoder model name
        decoder_name: Decoder model name
        embedding_dim: Matryoshka embedding dimension
        device: Device to use
        output_dir: Directory to save results
        num_evaluators: Number of parallel evaluation workers
        generation_batch_size: Batch size for code generation
        verbose: Print progress
    """

    # Set defaults
    if encoder_name is None:
        encoder_name = DEFAULT_ENCODER
    if decoder_name is None:
        decoder_name = DEFAULT_DECODER

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Interpolation Search in U-Space (No Predictor)")
    print("="*70)
    print(f"Task: {task_name}")
    print(f"Flow: {flow_path}")
    print(f"Mapper: {mapper_path}")
    print(f"Encoder: {encoder_name}")
    print(f"Decoder: {decoder_name}")
    print(f"Top-k pool size: {top_k_pool_size}")
    print(f"Searches per iteration: {num_searches_per_iter}")
    print()

    # Load models
    flow_model = load_flow(flow_path, device)
    decoder_model, decoder_tokenizer = load_decoder(decoder_name)
    mapper_model = load_mapper(mapper_path, decoder_model, device)

    embed_layer = decoder_model.get_input_embeddings()
    mapper_model = mapper_model.to(embed_layer.weight.device)

    print(f"Loading encoder ({encoder_name})...")
    encoder_model, actual_embedding_dim = load_encoder(
        model_name=encoder_name,
        device=device,
        truncate_dim=embedding_dim
    )
    print(f"Embedding dimension: {actual_embedding_dim}")

    print(f"Loading evaluator for {task_name}...")
    evaluator = get_evaluator(task_name)
    secure_eval = SecureEvaluator(evaluator, debug_mode=False)

    skeleton_prompt = TASK_PROMPTS.get(task_name)
    if skeleton_prompt is None:
        raise ValueError(f"No prompt defined for task: {task_name}")

    # Load and evaluate baselines
    print(f"\nLoading baseline heuristics from task/{task_name}/heuristics.json...")
    heuristics = load_heuristics(task_name)
    baseline_codes = list(heuristics.values())

    if len(baseline_codes) == 0:
        raise ValueError(f"No heuristics found")

    print(f"  Loaded {len(baseline_codes)} baseline heuristics")

    # Parallel evaluation of baselines
    print(f"\nEvaluating baseline heuristics ({num_evaluators} workers)...")

    def eval_baseline(idx_code):
        idx, code = idx_code
        try:
            score = secure_eval.evaluate_program(code)
            return idx, code, score, None
        except Exception as e:
            return idx, code, None, str(e)

    program_database = {}

    baseline_results = []
    with ThreadPoolExecutor(max_workers=num_evaluators) as executor:
        futures = {executor.submit(eval_baseline, (idx, code)): idx
                   for idx, code in enumerate(baseline_codes)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating baselines"):
            baseline_results.append(future.result())

    # Encode valid baselines
    valid_baselines = [(idx, code, score) for idx, code, score, error in baseline_results
                       if score is not None and np.isfinite(score)]

    print(f"  Valid baselines: {len(valid_baselines)}/{len(baseline_codes)}")
    print("  Encoding baselines...")

    for idx, code, score in tqdm(valid_baselines, desc="Encoding"):
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
                print(f"  Encoding error: {e}")
            continue

    if len(program_database) == 0:
        raise ValueError("No valid baseline programs!")

    # Show statistics
    sorted_programs = sorted(program_database.values(), key=lambda x: x['actual_score'], reverse=True)
    print(f"\n✓ Successfully initialized {len(program_database)} baseline programs")
    print(f"  Best score: {sorted_programs[0]['actual_score']:.4f}")
    print(f"  Worst score: {sorted_programs[-1]['actual_score']:.4f}")
    print(f"  Mean score: {np.mean([p['actual_score'] for p in program_database.values()]):.4f}")

    # Run search iterations
    all_results = []
    successful_programs = {}

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        # Sort programs by actual score
        sorted_programs = sorted(
            program_database.values(),
            key=lambda x: x['actual_score'],
            reverse=True
        )

        # Get top-k pool
        pool_size = min(top_k_pool_size, len(sorted_programs))
        top_k_pool = sorted_programs[:pool_size]

        print(f"\nSampling parents from top-{pool_size} programs:")
        print(f"  Best score: {top_k_pool[0]['actual_score']:.4f}")
        print(f"  Worst in pool: {top_k_pool[-1]['actual_score']:.4f}")
        print(f"  Database size: {len(program_database)}")

        # Extract embeddings from top-k
        top_k_embeddings = torch.stack([p['embedding'] for p in top_k_pool])

        # Generate children via interpolation
        print(f"\nGenerating {num_searches_per_iter} children via interpolation...")

        children_z = interpolate_in_u_space(
            flow_model=flow_model,
            parent_embeddings=top_k_embeddings,
            num_children=num_searches_per_iter,
            device=device,
            verbose=verbose
        )

        # Generate code in batches
        print(f"\nGenerating code for {len(children_z)} children (batch_size={generation_batch_size})...")

        all_generated = []
        for batch_start in range(0, len(children_z), generation_batch_size):
            batch_end = min(batch_start + generation_batch_size, len(children_z))
            z_batch = children_z[batch_start:batch_end]

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
                print(f"  Generated batch {batch_start//generation_batch_size + 1}/{(len(children_z) + generation_batch_size - 1)//generation_batch_size}")

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

            # Add to database
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
                print(f"  ✓ Score: {score:.4f} | Function: {func_name}")

        # Iteration summary
        print(f"\nIteration {iteration + 1} Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total in database: {len(program_database)}")
        print(f"  Total discovered: {len(successful_programs)}")

        if all_results:
            scores = [r['score'] for r in all_results]
            print(f"  Best score so far: {max(scores):.4f}")
            print(f"  Avg score (discoveries): {np.mean(scores):.4f}")

    # Final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal programs: {len(program_database)}")
    print(f"  Baselines: {sum(1 for p in program_database.values() if p['source']=='baseline')}")
    print(f"  Discovered: {sum(1 for p in program_database.values() if p['source']=='discovered')}")

    db_programs = list(program_database.values())
    db_programs.sort(key=lambda x: x['actual_score'], reverse=True)

    print(f"\nDatabase Statistics:")
    print(f"  Best score: {db_programs[0]['actual_score']:.4f} ({db_programs[0]['source']})")
    print(f"  Mean score: {np.mean([p['actual_score'] for p in db_programs]):.4f}")

    if all_results:
        all_results.sort(key=lambda x: x['score'], reverse=True)
        scores = [r['score'] for r in all_results]
        print(f"\nDiscovered Programs:")
        print(f"  Count: {len(all_results)}")
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Avg score: {np.mean(scores):.4f}")

    # Save results
    programs_path = os.path.join(output_dir, f"{task_name}_interpolation_searched.json")
    with open(programs_path, 'w') as f:
        json.dump(successful_programs, f, indent=2)
    print(f"\nSaved {len(successful_programs)} programs to {programs_path}")

    results_path = os.path.join(output_dir, f"{task_name}_search_results_interpolation.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Cleanup
    del encoder_model
    torch.cuda.empty_cache()

    return all_results, successful_programs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Interpolation search in u-space (no predictor needed)')
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--flow', type=str, default='Flow_Checkpoints/unified_flow_final.pth', help='Path to normalizing flow')
    parser.add_argument('--mapper', type=str, default='Mapper_Checkpoints/unified_mapper.pth', help='Path to mapper')
    parser.add_argument('--encoder', type=str, default=DEFAULT_ENCODER, help=f'Encoder model')
    parser.add_argument('--embedding-dim', type=int, default=None, help=f'Matryoshka embedding dimension')
    parser.add_argument('--decoder', type=str, default=DEFAULT_DECODER, help=f'Decoder model')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of search iterations')
    parser.add_argument('--num_searches', type=int, default=5, help='Number of children per iteration')
    parser.add_argument('--top_k_pool', type=int, default=10, help='Pool size to sample parents from (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Decoder temperature')
    parser.add_argument('--output_dir', type=str, default='interpolation_search_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_evaluators', type=int, default=4, help='Number of parallel evaluation workers')
    parser.add_argument('--generation_batch_size', type=int, default=4, help='Batch size for code generation')

    args = parser.parse_args()

    results, programs = interpolation_search_pipeline(
        task_name=args.task,
        flow_path=args.flow,
        mapper_path=args.mapper,
        num_iterations=args.num_iterations,
        num_searches_per_iter=args.num_searches,
        top_k_pool_size=args.top_k_pool,
        temperature=args.temperature,
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        embedding_dim=args.embedding_dim,
        device=args.device,
        output_dir=args.output_dir,
        num_evaluators=args.num_evaluators,
        generation_batch_size=args.generation_batch_size,
        verbose=True
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
