"""
Crossover-based evolutionary search using trained Mapper and Normalizing Flow.

Implements prior-space crossover:
1. Select high-performing parents from database
2. Map to prior space: u_A = F(z_A), u_B = F(z_B)
3. Interpolate: u_child = (1-α)u_A + αu_B
4. Map back: z_child = F^(-1)(u_child)
5. Generate code using mapper + decoder
6. Evaluate and add successful programs to database
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from programDB import ProgramDatabase
from task.tsp_construct.evaluation import TSPEvaluation
from mapper import Mapper
from normalizing_flow import NormalizingFlow, sample_from_flow
from crossover import PriorSpaceCrossover
from utils import extract_python_code_robust, is_valid_python
from base.code import TextFunctionProgramConverter
import os
import json


def load_trained_models(mapper_path, flow_path, device="cuda"):
    """Load trained mapper and flow models, reading dimensions from checkpoints."""

    # Load decoder (QWEN2.5 Coder)
    print("Loading decoder model...")
    decoder_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True
    )
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    # Load mapper checkpoint first to read dimensions
    print("Loading mapper...")
    mapper_checkpoint = torch.load(mapper_path, map_location=device)

    # Handle both old format (state_dict only) and new format (with metadata)
    if 'model_state_dict' in mapper_checkpoint:
        state_dict = mapper_checkpoint['model_state_dict']
        input_dim = mapper_checkpoint.get('input_dim')
        output_dim = mapper_checkpoint.get('output_dim')
        num_tokens = mapper_checkpoint.get('num_tokens')
    else:
        state_dict = mapper_checkpoint
        input_dim = None
        output_dim = None
        num_tokens = None

    # If not stored in checkpoint, infer from model weights
    if input_dim is None or output_dim is None or num_tokens is None:
        print("  Dimensions not found in checkpoint, inferring from model architecture...")
        input_dim = state_dict['mlp.0.weight'].shape[1]
        hidden_output_dim = state_dict['mlp.0.weight'].shape[0]
        final_output_size = state_dict['mlp.4.weight'].shape[0]

        output_dim = decoder_model.config.hidden_size
        num_tokens = final_output_size // output_dim

    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Num tokens: {num_tokens}")

    mapper_model = Mapper(input_dim, output_dim, num_tokens)
    mapper_model.load_state_dict(state_dict)
    mapper_model.eval()

    # Load flow checkpoint
    print("Loading normalizing flow...")
    flow_checkpoint = torch.load(flow_path, map_location=device)

    flow_dim = flow_checkpoint['dim']
    flow_num_layers = flow_checkpoint['num_layers']
    flow_hidden_dim = flow_checkpoint['hidden_dim']

    print(f"  Flow dim: {flow_dim}")
    print(f"  Flow layers: {flow_num_layers}")
    print(f"  Flow hidden dim: {flow_hidden_dim}")

    flow_model = NormalizingFlow(
        dim=flow_dim,
        num_layers=flow_num_layers,
        hidden_dim=flow_hidden_dim
    )
    flow_model.load_state_dict(flow_checkpoint['model_state_dict'])
    flow_model.eval()

    return decoder_model, decoder_tokenizer, mapper_model, flow_model


def generate_code_from_z(z_vector, mapper_model, decoder_model, decoder_tokenizer,
                         skeleton_prompt, temperature=0.7, top_p=0.9):
    """Generate code from a latent vector z using the mapper and decoder."""

    embed_layer = decoder_model.get_input_embeddings()
    device = embed_layer.weight.device
    dtype = embed_layer.weight.dtype

    z_tensor = torch.tensor(z_vector, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        soft_prompt_embeds = mapper_model(z_tensor).to(device, dtype=dtype)

        instruction_messages = [{"role": "user", "content": skeleton_prompt}]
        instruction_ids = decoder_tokenizer.apply_chat_template(
            instruction_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        instruction_embeds = embed_layer(instruction_ids)
        inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)

        generated_ids = decoder_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=1024,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id
        )

        raw_output = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        clean_code = extract_python_code_robust(raw_output, include_preface=True)

        return clean_code, raw_output


def crossover_evolution(
    program_db,
    flow_model,
    mapper_model,
    decoder_model,
    decoder_tokenizer,
    evaluator,
    skeleton_prompt,
    num_iterations=10,
    num_pairs_per_iter=5,
    offspring_per_pair=5,
    top_q_percent=0.2,
    alpha_range=(0.2, 0.8),
    also_sample_random=True,
    num_random_samples=2,
    device="cuda"
):
    """
    Perform evolutionary search using prior-space crossover only.

    Args:
        program_db: ProgramDatabase with initial programs
        flow_model: Trained normalizing flow
        mapper_model: Trained mapper
        decoder_model: Decoder LLM
        decoder_tokenizer: Tokenizer for decoder
        evaluator: Evaluation instance (e.g., TSPEvaluation)
        skeleton_prompt: Template prompt for generation
        num_iterations: Number of evolution iterations
        num_pairs_per_iter: Number of parent pairs to cross per iteration
        offspring_per_pair: Number of offspring per parent pair
        top_q_percent: Select parents from top q%
        alpha_range: Range for interpolation parameter (min, max)
        also_sample_random: Also sample from prior for exploration
        num_random_samples: Number of random samples if enabled
        device: Device to use

    Returns:
        ProgramDatabase with all discovered programs, generation statistics
    """

    crossover_op = PriorSpaceCrossover(flow_model, device=device)
    print("✓ Crossover operator initialized")

    print(f"\nStarting crossover-based evolution with {len(program_db)} initial programs")
    print(f"Best initial score: {program_db.df['score'].max():.4f}")
    print(f"Average initial score: {program_db.df['score'].mean():.4f}")

    generation_stats = []

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        candidates = []

        # ===== CROSSOVER: Interpolate between high-performing parents =====
        print(f"\n1. Prior-space crossover ({num_pairs_per_iter} pairs, {offspring_per_pair} offspring each)...")

        try:
            offspring_z, parent_pairs, alphas = crossover_op.crossover_from_database(
                program_db,
                num_pairs=num_pairs_per_iter,
                offspring_per_pair=offspring_per_pair,
                top_q_percent=top_q_percent,
                alpha_range=alpha_range
            )

            print(f"   Generated {len(offspring_z)} offspring from crossover")

            for i, z in enumerate(offspring_z):
                candidates.append({
                    'z': z.cpu().numpy(),
                    'origin': f'crossover_gen{iteration}',
                    'parent_pair': parent_pairs[i // offspring_per_pair],
                    'alpha': alphas[i]
                })

        except Exception as e:
            print(f"   ⚠ Crossover failed: {e}")

        # ===== RANDOM SAMPLING: Sample from prior for exploration =====
        if also_sample_random and num_random_samples > 0:
            print(f"\n2. Random sampling from prior ({num_random_samples} samples)...")
            random_z = sample_from_flow(flow_model, num_samples=num_random_samples, device=device)

            for z in random_z:
                candidates.append({
                    'z': z.cpu().numpy(),
                    'origin': f'random_gen{iteration}',
                    'parent_pair': None,
                    'alpha': None
                })

        print(f"\nTotal candidates to evaluate: {len(candidates)}")

        # ===== GENERATION & EVALUATION =====
        successful = 0
        failed = 0

        for idx, candidate in enumerate(candidates):
            print(f"\n--- Candidate {idx+1}/{len(candidates)} ({candidate['origin']}) ---")

            try:
                # Generate code
                clean_code, raw_output = generate_code_from_z(
                    candidate['z'],
                    mapper_model,
                    decoder_model,
                    decoder_tokenizer,
                    skeleton_prompt,
                    temperature=0.7,
                    top_p=0.9
                )

                print(f"Generated code length: {len(clean_code)} chars")

                # Validate Python syntax
                if not is_valid_python(clean_code):
                    print("❌ Invalid Python syntax")
                    failed += 1
                    continue

                # Parse to ensure it's a valid program
                program = TextFunctionProgramConverter.text_to_program(clean_code)
                if program is None or len(program.functions) == 0:
                    print("❌ Could not parse function")
                    failed += 1
                    continue

                function_name = program.functions[0].name
                print(f"✓ Valid Python | Function: {function_name}")

                # Check if already exists in database
                if program_db.exists(clean_code):
                    print("⚠ Already exists in database - skipping")
                    failed += 1
                    continue

                # Evaluate
                print("Evaluating...")
                score = None
                try:
                    local_scope = {}
                    exec(clean_code, {"np": np}, local_scope)
                    callable_func = local_scope[function_name]

                    score = evaluator.evaluate_program('_', callable_func)

                    if score is None:
                        print("❌ Evaluation returned None")
                        failed += 1
                        continue

                    if not np.isfinite(score):
                        print(f"❌ Non-finite score: {score}")
                        failed += 1
                        continue

                except Exception as e:
                    print(f"❌ Evaluation error: {type(e).__name__}: {str(e)[:100]}")
                    failed += 1
                    continue

                print(f"✓ Score: {score:.4f}")

                # Add to database
                program_db.add_program(
                    code=clean_code,
                    z=candidate['z'],
                    score=score,
                    origin=candidate['origin'],
                    generation=iteration
                )

                successful += 1

                # Show code snippet
                print(f"\nCode preview:")
                print("-" * 70)
                code_preview = str(program.functions[0])[:300]
                print(code_preview + "..." if len(str(program.functions[0])) > 300 else code_preview)
                print("-" * 70)

            except Exception as e:
                print(f"❌ Error: {e}")
                failed += 1
                continue

        # ===== ITERATION SUMMARY =====
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1} Summary:")
        print(f"{'='*70}")
        print(f"Successful: {successful} | Failed: {failed}")
        print(f"Database size: {len(program_db)}")

        current_best = program_db.df['score'].max()
        current_avg = program_db.df['score'].mean()
        print(f"Best score: {current_best:.4f}")
        print(f"Average score: {current_avg:.4f}")

        generation_stats.append({
            'iteration': iteration,
            'successful': successful,
            'failed': failed,
            'best_score': current_best,
            'avg_score': current_avg,
            'db_size': len(program_db)
        })

        # Show top 3 programs
        top_3 = program_db.get_top_n(3)
        print(f"\nTop 3 programs:")
        for rank, (pid, row) in enumerate(top_3.iterrows(), 1):
            print(f"  {rank}. Score: {row['score']:.4f} | Origin: {row['origin']} | ID: {pid}")

    return program_db, generation_stats


def main():
    """Main crossover evolution workflow."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== LOAD TRAINED MODELS =====
    mapper_path = "Mapper_Checkpoints/Mapper.pth"
    flow_path = "Flow_Checkpoints/normalizing_flow_final.pth"

    decoder_model, decoder_tokenizer, mapper_model, flow_model = load_trained_models(
        mapper_path, flow_path, device
    )

    # ===== LOAD ENCODER AND EVALUATOR =====
    print("\nLoading evaluator...")
    evaluator = TSPEvaluation()

    # ===== LOAD OR CREATE PROGRAM DATABASE =====
    program_db = ProgramDatabase()

    if os.path.exists("task/tsp_construct/heuristics.json"):
        print("\nLoading initial programs from JSON...")
        encoder_model = SentenceTransformer(
            "BAAI/bge-code-v1",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16}
        ).to(device)
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-code-v1",
            trust_remote_code=True
        )
        program_db.load_func_from_json(
            "task/tsp_construct/heuristics.json",
            encoder_model,
            encoder_tokenizer,
            evaluator,
            device
        )
    else:
        print("\n⚠ No initial programs found. Please provide heuristics.json")
        return

    skeleton_prompt = "Write a Python function that implements a heuristic for the Traveling Salesman Problem."

    # ===== RUN CROSSOVER EVOLUTION =====
    print("\n" + "="*70)
    print("Starting Crossover-Based Evolutionary Search")
    print("="*70)

    final_db, stats = crossover_evolution(
        program_db=program_db,
        flow_model=flow_model,
        mapper_model=mapper_model,
        decoder_model=decoder_model,
        decoder_tokenizer=decoder_tokenizer,
        evaluator=evaluator,
        skeleton_prompt=skeleton_prompt,
        num_iterations=10,
        num_pairs_per_iter=5,
        offspring_per_pair=5,
        top_q_percent=0.2,
        alpha_range=(0.2, 0.8),
        also_sample_random=True,
        num_random_samples=2,
        device=device
    )

    # ===== FINAL RESULTS =====
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\nFinal database size: {len(final_db)}")
    print(f"Best score: {final_db.df['score'].max():.4f}")
    print(f"Average score: {final_db.df['score'].mean():.4f}")

    # Show top 10 programs
    print("\n" + "="*70)
    print("Top 10 Programs:")
    print("="*70)
    top_10 = final_db.get_top_n(10)
    for rank, (pid, row) in enumerate(top_10.iterrows(), 1):
        print(f"\n{rank}. Score: {row['score']:.4f} | Origin: {row['origin']} | Gen: {row['generation']}")
        print("-" * 70)
        code_preview = row['code'][:400]
        print(code_preview + "..." if len(row['code']) > 400 else code_preview)

    # ===== SAVE RESULTS =====
    save_path = "task/tsp_construct/heuristics_crossover_evolved.json"

    evolved_programs = {}
    for pid, row in final_db.df.iterrows():
        evolved_programs[f"program_{pid}"] = row['code']

    with open(save_path, 'w') as f:
        json.dump(evolved_programs, f, indent=2)

    print(f"\n✓ Saved {len(evolved_programs)} programs to {save_path}")

    db_path = "program_database_crossover_evolved.parquet"
    final_db.to_disk(db_path)
    print(f"✓ Saved database to {db_path}")


if __name__ == "__main__":
    main()
