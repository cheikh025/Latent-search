"""
Gradient-based evolutionary search using trained Mapper, Normalizing Flow, and Score Predictor.

Implements gradient-based optimization in prior space:
1. Train score predictor R(u) to predict program scores in prior space
2. Use gradient ascent to find u* = argmax R(u)
3. Map back to latent space: z* = F^(-1)(u*)
4. Generate code using mapper + decoder
5. Evaluate and add successful programs to database
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from model_config import DEFAULT_ENCODER, DEFAULT_DECODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder
from programDB import ProgramDatabase
from task.tsp_construct.evaluation import TSPEvaluation
from mapper import Mapper
from normalizing_flow import NormalizingFlow, sample_from_flow
from score_predictor import ScorePredictor, train_score_predictor, adaptive_gradient_search
from utils import extract_python_code_robust, is_valid_python
from base.code import TextFunctionProgramConverter
from base.evaluate import SecureEvaluator
import os
import json


def load_trained_models(mapper_path, flow_path, device="cuda", decoder_name=None):
    """Load trained mapper and flow models, reading dimensions from checkpoints.

    Args:
        mapper_path: Path to trained mapper checkpoint.
        flow_path: Path to trained normalizing flow checkpoint.
        device: Device to use.
        decoder_name: Decoder model name. Defaults to DEFAULT_DECODER from model_config.py
    """
    if decoder_name is None:
        decoder_name = DEFAULT_DECODER

    # Load decoder
    print(f"Loading decoder model: {decoder_name}...")
    decoder_model = AutoModelForCausalLM.from_pretrained(
        decoder_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        decoder_name,
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


def gradient_evolution(
    program_db,
    flow_model,
    mapper_model,
    decoder_model,
    decoder_tokenizer,
    evaluator,
    skeleton_prompt,
    num_iterations=10,
    num_gradient_samples=10,
    gradient_steps=100,
    gradient_lr=0.01,
    retrain_predictor_every=3,
    predictor_hidden_dim=256,
    predictor_training_epochs=100,
    also_sample_random=True,
    num_random_samples=2,
    device="cuda"
):
    """
    Perform evolutionary search using gradient-based optimization only.

    Args:
        program_db: ProgramDatabase with initial programs
        flow_model: Trained normalizing flow
        mapper_model: Trained mapper
        decoder_model: Decoder LLM
        decoder_tokenizer: Tokenizer for decoder
        evaluator: Evaluation instance (e.g., TSPEvaluation)
        skeleton_prompt: Template prompt for generation
        num_iterations: Number of evolution iterations
        num_gradient_samples: Number of gradient ascent runs per iteration
        gradient_steps: Steps per gradient ascent
        gradient_lr: Learning rate for gradient ascent
        retrain_predictor_every: Retrain score predictor every N iterations
        predictor_hidden_dim: Hidden dimension for score predictor MLP
        predictor_training_epochs: Epochs to train score predictor
        also_sample_random: Also sample from prior for exploration
        num_random_samples: Number of random samples if enabled
        device: Device to use

    Returns:
        ProgramDatabase with all discovered programs, generation statistics
    """

    # Initialize score predictor
    score_predictor = ScorePredictor(
        input_dim=flow_model.dim,
        hidden_dim=predictor_hidden_dim
    ).to(device)
    print("✓ Score predictor initialized (2-layer MLP)")

    # Create secure evaluator wrapper
    secure_eval = SecureEvaluator(evaluator, debug_mode=False)
    print("✓ Secure evaluator initialized")

    # Train initial predictor
    print("\nTraining initial score predictor...")
    score_predictor = train_score_predictor(
        score_predictor,
        program_db,
        flow_model,
        epochs=predictor_training_epochs,
        batch_size=32,
        lr=1e-3,
        device=device,
        verbose=True
    )

    print(f"\nStarting gradient-based evolution with {len(program_db)} initial programs")
    print(f"Best initial score: {program_db.df['score'].max():.4f}")
    print(f"Average initial score: {program_db.df['score'].mean():.4f}")

    generation_stats = []

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        # Retrain predictor periodically
        if iteration > 0 and iteration % retrain_predictor_every == 0:
            print("\n📊 Retraining score predictor with updated database...")
            score_predictor = train_score_predictor(
                score_predictor,
                program_db,
                flow_model,
                epochs=50,  # Fewer epochs for retraining
                batch_size=32,
                lr=1e-3,
                device=device,
                verbose=True
            )

        candidates = []

        # ===== GRADIENT SEARCH: Optimize in prior space using score predictor =====
        print(f"\n1. Gradient-based search ({num_gradient_samples} runs, {gradient_steps} steps each)...")

        try:
            optimized_u, optimized_z = adaptive_gradient_search(
                score_predictor,
                flow_model,
                program_db,
                num_searches=num_gradient_samples,
                steps_per_search=gradient_steps,
                lr=gradient_lr,
                init_from_top_k=5,
                device=device,
                verbose=True
            )

            print(f"   Generated {len(optimized_z)} candidates from gradient search")

            for z in optimized_z:
                candidates.append({
                    'z': z.cpu().numpy(),
                    'origin': f'gradient_gen{iteration}',
                    'parent_pair': None,
                    'alpha': None
                })

        except Exception as e:
            print(f"   ⚠ Gradient search failed: {e}")

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

                # Evaluate using secure evaluator
                print("Evaluating...")
                try:
                    score = secure_eval.evaluate_program(clean_code)

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
    """Main gradient evolution workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Gradient-based evolutionary search")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER, help=f"Encoder model name (default: {DEFAULT_ENCODER})")
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help=f"Matryoshka embedding dimension (default: {DEFAULT_MATRYOSHKA_DIM or 'model native'})")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, help=f"Decoder model name (default: {DEFAULT_DECODER})")
    parser.add_argument("--mapper", type=str, default="Mapper_Checkpoints/Mapper.pth", help="Path to mapper checkpoint")
    parser.add_argument("--flow", type=str, default="Flow_Checkpoints/normalizing_flow_final.pth", help="Path to flow checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")

    # ===== LOAD TRAINED MODELS =====
    mapper_path = args.mapper
    flow_path = args.flow

    decoder_model, decoder_tokenizer, mapper_model, flow_model = load_trained_models(
        mapper_path, flow_path, device, decoder_name=args.decoder
    )

    # ===== LOAD EVALUATOR =====
    print("\nLoading evaluator...")
    evaluator = TSPEvaluation()

    # ===== LOAD OR CREATE PROGRAM DATABASE =====
    program_db = ProgramDatabase()

    if os.path.exists("task/tsp_construct/heuristics.json"):
        print(f"\nLoading initial programs from JSON...")
        print(f"Using encoder: {args.encoder}")
        encoder_model, embedding_dim = load_encoder(
            model_name=args.encoder,
            device=device,
            truncate_dim=getattr(args, 'embedding_dim', None)
        )
        print(f"Embedding dimension: {embedding_dim}")
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            args.encoder,
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

    # ===== RUN GRADIENT EVOLUTION =====
    print("\n" + "="*70)
    print("Starting Gradient-Based Evolutionary Search")
    print("="*70)

    final_db, stats = gradient_evolution(
        program_db=program_db,
        flow_model=flow_model,
        mapper_model=mapper_model,
        decoder_model=decoder_model,
        decoder_tokenizer=decoder_tokenizer,
        evaluator=evaluator,
        skeleton_prompt=skeleton_prompt,
        num_iterations=10,
        num_gradient_samples=10,
        gradient_steps=100,
        gradient_lr=0.01,
        retrain_predictor_every=3,
        predictor_hidden_dim=256,
        predictor_training_epochs=100,
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
    save_path = "task/tsp_construct/heuristics_gradient_evolved.json"

    evolved_programs = {}
    for pid, row in final_db.df.iterrows():
        evolved_programs[f"program_{pid}"] = row['code']

    with open(save_path, 'w') as f:
        json.dump(evolved_programs, f, indent=2)

    print(f"\n✓ Saved {len(evolved_programs)} programs to {save_path}")

    db_path = "program_database_gradient_evolved.parquet"
    final_db.to_disk(db_path)
    print(f"✓ Saved database to {db_path}")


if __name__ == "__main__":
    main()
