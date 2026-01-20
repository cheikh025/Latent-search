"""
Unified Multi-Task Mapper Training

Trains a single mapper on augmented heuristics from all combinatorial optimization tasks.
This enables the mapper to learn universal latent-to-prompt transformations across:
- TSP, CVRP, VRPTW, JSSP, Knapsack, Bin Packing, QAP, CFLP, Set Cover, Admissible Set
"""

import os
import json
import glob
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from mapper import Mapper, train_mapper
from utils import is_valid_python


# ============================================================================
# Task-Specific Skeleton Prompts
# ============================================================================

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

# ============================================================================
# Prompt Augmentation: Problem-Class and General Prompts
# ============================================================================

# Problem-class general prompts (20% of training samples)
PROBLEM_CLASS_PROMPTS = [
    "Write a Python function that implements a heuristic for a combinatorial optimization problem.",
    "Implement a greedy heuristic function in Python for an optimization problem.",
    "Write a Python function for solving a discrete optimization problem.",
    "Create a Python heuristic for a routing or scheduling problem.",
    "Write a constructive heuristic function in Python.",
    "Implement an optimization heuristic in Python.",
]

# Fully general prompts (20% of training samples)
GENERAL_PROMPTS = [
    "Write a Python function.",
    "Write a Python heuristic function.",
    "Implement the following algorithmic approach in Python.",
    "Write the following function in Python.",
    "Create a Python function.",
    "Implement this algorithm in Python.",
]


def sample_prompt_with_augmentation(
    task_name: str,
    task_specific_prob: float = 0.60,
    problem_class_prob: float = 0.20,
    general_prob: float = 0.20,
    rng: Optional[np.random.Generator] = None
) -> str:
    """
    Sample a prompt using augmentation strategy for robustness.

    Strategy 1: Prompt Augmentation
    - 60% task-specific (e.g., "Write TSP heuristic")
    - 20% problem-class general (e.g., "Write combinatorial optimization heuristic")
    - 20% fully general (e.g., "Write a Python function")

    This forces the mapper to rely on latent z for reconstruction, not the prompt.

    Args:
        task_name: Name of the task (e.g., 'tsp_construct')
        task_specific_prob: Probability of using task-specific prompt
        problem_class_prob: Probability of using problem-class prompt
        general_prob: Probability of using fully general prompt
        rng: Random number generator (for reproducibility)

    Returns:
        Sampled prompt string
    """
    if rng is None:
        rng = np.random.default_rng()

    # Normalize probabilities
    total = task_specific_prob + problem_class_prob + general_prob
    task_specific_prob /= total
    problem_class_prob /= total
    general_prob /= total

    # Sample prompt type
    choice = rng.choice(
        ['task_specific', 'problem_class', 'general'],
        p=[task_specific_prob, problem_class_prob, general_prob]
    )

    if choice == 'task_specific':
        return TASK_PROMPTS.get(task_name, PROBLEM_CLASS_PROMPTS[0])
    elif choice == 'problem_class':
        return rng.choice(PROBLEM_CLASS_PROMPTS)
    else:  # general
        return rng.choice(GENERAL_PROMPTS)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_augmented_heuristics(task_dir: str = "task") -> List[Tuple[str, str, str, str]]:
    """
    Load all augmented heuristics from all tasks.

    Args:
        task_dir: Root directory containing task subdirectories

    Returns:
        List of (code, task_name, skeleton_prompt, heuristic_name) tuples
    """
    all_heuristics = []
    augmented_files = glob.glob(os.path.join(task_dir, "*", "augmented.json"))

    print(f"\n{'='*70}")
    print(f"Loading Augmented Heuristics from All Tasks")
    print(f"{'='*70}")
    print(f"Found {len(augmented_files)} task directories\n")

    for aug_file in sorted(augmented_files):
        task_name = Path(aug_file).parent.name

        # Get task-specific prompt
        skeleton_prompt = TASK_PROMPTS.get(task_name)
        if skeleton_prompt is None:
            warnings.warn(f"No skeleton prompt defined for task '{task_name}', skipping...")
            continue

        # Load augmented heuristics JSON
        try:
            with open(aug_file, 'r', encoding='utf-8') as f:
                heuristics_dict = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load {aug_file}: {e}")
            continue

        # Check if empty
        if not heuristics_dict:
            warnings.warn(f"Empty augmented.json for task '{task_name}', skipping...")
            continue

        # Validate and add heuristics
        valid_count = 0
        for heuristic_name, code in heuristics_dict.items():
            # Basic validation
            if not code or not isinstance(code, str):
                continue

            # Optional: validate Python syntax
            if not is_valid_python(code):
                warnings.warn(f"Invalid Python syntax in {task_name}/{heuristic_name}, skipping...")
                continue

            all_heuristics.append((code, task_name, skeleton_prompt, heuristic_name))
            valid_count += 1

        print(f"  {task_name:25s}: {valid_count:3d} heuristics")

    print(f"\n{'='*70}")
    print(f"Total: {len(all_heuristics)} valid heuristics across {len(set(h[1] for h in all_heuristics))} tasks")
    print(f"{'='*70}\n")

    return all_heuristics


def encode_all_heuristics(
    heuristics_list: List[Tuple[str, str, str, str]],
    encoder_model: SentenceTransformer,
    encoder_tokenizer,
    device: str = "cuda",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Encode all heuristics using the code encoder.

    Args:
        heuristics_list: List of (code, task, prompt, name) tuples
        encoder_model: SentenceTransformer encoder model
        encoder_tokenizer: Encoder tokenizer
        device: Device for encoding
        batch_size: Batch size for encoding

    Returns:
        DataFrame with columns: ['code', 'z', 'task', 'skeleton_prompt', 'heuristic_name']
    """
    print(f"\n{'='*70}")
    print(f"Encoding All Heuristics")
    print(f"{'='*70}\n")

    codes = [h[0] for h in heuristics_list]
    tasks = [h[1] for h in heuristics_list]
    prompts = [h[2] for h in heuristics_list]
    names = [h[3] for h in heuristics_list]

    # Batch encode
    print(f"Encoding {len(codes)} programs in batches of {batch_size}...")
    embeddings = []

    encoder_model.eval()
    encoder_model.to(device)

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

    print(f"\n✓ Encoded all programs")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

    # Create DataFrame
    df = pd.DataFrame({
        'code': codes,
        'z': list(embeddings),  # Store as list of numpy arrays
        'task': tasks,
        'skeleton_prompt': prompts,
        'heuristic_name': names
    })

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"{'='*70}")
    task_counts = df['task'].value_counts().sort_index()
    for task, count in task_counts.items():
        print(f"  {task:25s}: {count:3d} programs")
    print(f"{'='*70}\n")

    return df


# ============================================================================
# Custom Collate Function for Multi-Task Training
# ============================================================================

def multi_task_collate_fn(batch):
    """
    Collate function that handles per-sample data with task names.

    Args:
        batch: List of (code, z, task_name) tuples

    Returns:
        codes: List of code strings
        zs: Tensor of embeddings [batch_size, embedding_dim]
        task_names: List of task name strings
    """
    codes, zs, task_names = zip(*batch)
    zs = torch.tensor(np.stack(zs), dtype=torch.float32)
    return list(codes), zs, list(task_names)


# ============================================================================
# Modified Training Function for Multi-Task Mapper
# ============================================================================

def train_unified_mapper(
    df: pd.DataFrame,
    mapper_model,
    optimizer,
    decoder_model,
    decoder_tokenizer,
    batch_size: int = 4,
    epochs: int = 30,
    accumulation_steps: int = 2,
    max_length: Optional[int] = 2048,
    verbose: bool = True,
    checkpoint_dir: str = "Mapper_Checkpoints",
    task_specific_prob: float = 0.60,
    problem_class_prob: float = 0.20,
    general_prob: float = 0.20,
    seed: int = 42
):
    """
    Train mapper on unified multi-task dataset with prompt augmentation.

    This uses Strategy 1: Prompt Augmentation for robustness.
    - 60% task-specific prompts (e.g., "Write TSP heuristic")
    - 20% problem-class prompts (e.g., "Write combinatorial optimization heuristic")
    - 20% general prompts (e.g., "Write a Python function")

    This forces the mapper to rely on latent z for reconstruction, not the prompt.

    Args:
        df: DataFrame with columns ['code', 'z', 'task', ...]
        mapper_model: Mapper model instance
        optimizer: Optimizer for mapper parameters
        decoder_model: Frozen decoder (Qwen2.5-Coder-7B)
        decoder_tokenizer: Decoder tokenizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        accumulation_steps: Gradient accumulation steps
        max_length: Max sequence length for tokenization
        verbose: Whether to print training progress
        checkpoint_dir: Directory to save checkpoints
        task_specific_prob: Probability of using task-specific prompt (default: 0.60)
        problem_class_prob: Probability of using problem-class prompt (default: 0.20)
        general_prob: Probability of using fully general prompt (default: 0.20)
        seed: Random seed for prompt sampling reproducibility
    """
    from torch.utils.data import DataLoader

    print(f"\n{'='*70}")
    print(f"Training Unified Multi-Task Mapper with Prompt Augmentation")
    print(f"{'='*70}\n")

    # Prepare dataset (use task names for on-the-fly prompt sampling)
    dataset = list(zip(
        df['code'].tolist(),
        df['z'].tolist(),
        df['task'].tolist()  # Store task names, not prompts!
    ))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multi_task_collate_fn
    )

    # Initialize RNG for prompt sampling
    rng = np.random.default_rng(seed)

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=verbose,
        min_lr=1e-7
    )

    # Freeze decoder and get device
    mapper_model.train()
    decoder_model.eval()
    for p in decoder_model.parameters():
        p.requires_grad = False
    decoder_model.config.use_cache = False

    embed_layer = decoder_model.get_input_embeddings()
    first_dev = embed_layer.weight.device
    embed_dtype = embed_layer.weight.dtype
    mapper_model = mapper_model.to(first_dev)

    pad_id = decoder_tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("decoder_tokenizer.pad_token_id must be set.")

    # Training loop
    print(f"Training Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Total programs: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"\nPrompt Augmentation (Strategy 1):")
    print(f"  Task-specific:   {task_specific_prob*100:.0f}%")
    print(f"  Problem-class:   {problem_class_prob*100:.0f}%")
    print(f"  General:         {general_prob*100:.0f}%")
    print(f"\nStarting training...\n")

    for epoch in range(epochs):
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, (code_batch, z_batch, task_batch) in enumerate(dataloader):
            # Move batch to device
            z_batch = z_batch.to(first_dev)
            B = len(code_batch)

            # Build per-sample prompts with augmentation
            code_texts = [f"```python\n{code}\n```" for code in code_batch]

            prompt_id_tensors = []
            target_id_tensors = []

            for task_name, code_text in zip(task_batch, code_texts):
                # Sample augmented prompt on-the-fly (CRITICAL: Prompt Augmentation!)
                skel_text = sample_prompt_with_augmentation(
                    task_name=task_name,
                    task_specific_prob=task_specific_prob,
                    problem_class_prob=problem_class_prob,
                    general_prob=general_prob,
                    rng=rng
                )

                prompt_messages = [{"role": "user", "content": skel_text}]
                full_messages = [
                    {"role": "user", "content": skel_text},
                    {"role": "assistant", "content": code_text},
                ]

                prompt_ids = decoder_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=max_length,
                )
                full_ids = decoder_tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=max_length,
                )

                if len(full_ids) < len(prompt_ids):
                    raise ValueError("chat template produced invalid prompt/full lengths.")

                target_ids = full_ids[len(prompt_ids):]
                prompt_id_tensors.append(torch.tensor(prompt_ids, dtype=torch.long))
                target_id_tensors.append(torch.tensor(target_ids, dtype=torch.long))

            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence
            prompt_ids = pad_sequence(prompt_id_tensors, batch_first=True, padding_value=pad_id).to(first_dev)
            target_ids = pad_sequence(target_id_tensors, batch_first=True, padding_value=pad_id).to(first_dev)

            prompt_embeds = embed_layer(prompt_ids)

            # Mapper generates soft prompts
            soft_prompt_embeds = mapper_model(z_batch).to(first_dev, dtype=embed_dtype)

            if soft_prompt_embeds.dim() != 3 or soft_prompt_embeds.size(0) != B:
                raise ValueError(f"mapper output must be [B, S_soft, D]; got {tuple(soft_prompt_embeds.shape)}")

            # Target embeddings for teacher forcing
            target_embeds = embed_layer(target_ids)

            # Concatenate: [soft_prompts, instruction, target]
            inputs_embeds = torch.cat([soft_prompt_embeds, prompt_embeds, target_embeds], dim=1)

            # Build labels (mask soft prompts and instruction)
            ignore_left = torch.full(
                (B, prompt_embeds.size(1) + soft_prompt_embeds.size(1)),
                -100,
                dtype=torch.long,
                device=first_dev,
            )
            tgt_labels = target_ids.masked_fill(target_ids == pad_id, -100)
            labels = torch.cat([ignore_left, tgt_labels], dim=1)

            # Attention mask
            left_mask = torch.ones(B, prompt_embeds.size(1) + soft_prompt_embeds.size(1), dtype=torch.long, device=first_dev)
            right_mask = (target_ids != pad_id).long().to(first_dev)
            att_mask = torch.cat([left_mask, right_mask], dim=1)

            # Forward pass
            out = decoder_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
            loss = out.loss / accumulation_steps

            loss.backward()
            running_loss += loss.item()
            total_steps += 1

            if total_steps % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Final gradient step
        if total_steps % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Compute average loss
        avg_loss = running_loss / max(1, len(dataloader))

        # Step scheduler
        scheduler.step(avg_loss)

        torch.cuda.empty_cache()

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unified_mapper_epoch{epoch+1}.pth")
            torch.save(mapper_model.state_dict(), checkpoint_path)
            if verbose:
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    torch.cuda.empty_cache()
    return mapper_model


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training pipeline for unified multi-task mapper."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # ========================================================================
    # Step 1: Load All Augmented Heuristics
    # ========================================================================

    heuristics_list = load_all_augmented_heuristics(task_dir="task")

    if len(heuristics_list) == 0:
        raise ValueError("No valid heuristics loaded! Check task directories.")

    # ========================================================================
    # Step 2: Load Encoder Model
    # ========================================================================

    print(f"{'='*70}")
    print("Loading Encoder Model")
    print(f"{'='*70}\n")

    encoder_model = SentenceTransformer(
        "BAAI/bge-code-v1",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    ).to(device)

    encoder_tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-code-v1",
        trust_remote_code=True
    )

    encoder_model.eval()
    print("✓ Encoder loaded\n")

    # ========================================================================
    # Step 3: Encode All Heuristics
    # ========================================================================

    unified_df = encode_all_heuristics(
        heuristics_list=heuristics_list,
        encoder_model=encoder_model,
        encoder_tokenizer=encoder_tokenizer,
        device=device,
        batch_size=32
    )

    # Free encoder memory
    del encoder_model
    torch.cuda.empty_cache()

    # ========================================================================
    # Step 4: Load Decoder Model (Qwen2.5-Coder)
    # ========================================================================

    print(f"{'='*70}")
    print("Loading Decoder Model")
    print(f"{'='*70}\n")

    decoder_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16,
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

    print("✓ Decoder loaded\n")

    # ========================================================================
    # Step 5: Initialize Mapper
    # ========================================================================

    print(f"{'='*70}")
    print("Initializing Mapper")
    print(f"{'='*70}\n")

    embeddings = np.stack(unified_df['z'].values)
    input_dim = embeddings.shape[1]
    output_dim = decoder_model.config.hidden_size
    num_tokens = 16

    mapper_model = Mapper(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens
    )

    num_params = sum(p.numel() for p in mapper_model.parameters())
    print(f"Mapper Architecture:")
    print(f"  Input: {input_dim}D (code embeddings)")
    print(f"  Output: {num_tokens} tokens × {output_dim}D (soft prompts)")
    print(f"  Parameters: {num_params:,}\n")

    # ========================================================================
    # Step 6: Setup Optimizer
    # ========================================================================

    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(mapper_model.parameters(), lr=learning_rate)

    # ========================================================================
    # Step 7: Create Checkpoint Directory
    # ========================================================================

    checkpoint_dir = "Mapper_Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================================
    # Step 8: Train Unified Mapper
    # ========================================================================

    trained_mapper = train_unified_mapper(
        df=unified_df,
        mapper_model=mapper_model,
        optimizer=optimizer,
        decoder_model=decoder_model,
        decoder_tokenizer=decoder_tokenizer,
        batch_size=4,
        epochs=30,
        accumulation_steps=2,
        max_length=2048,
        verbose=True,
        checkpoint_dir=checkpoint_dir
    )

    # ========================================================================
    # Step 9: Save Final Checkpoint with Metadata
    # ========================================================================

    print(f"\n{'='*70}")
    print("Saving Final Checkpoint")
    print(f"{'='*70}\n")

    task_counts = unified_df['task'].value_counts().to_dict()

    checkpoint_data = {
        'model_state_dict': trained_mapper.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_tokens': num_tokens,
        'tasks_trained': list(TASK_PROMPTS.keys()),
        'total_programs': len(unified_df),
        'programs_per_task': task_counts,
        'epoch': 30,
        'prompt_augmentation': {
            'strategy': 'Strategy 1: Task-Specific + Problem-Class + General',
            'task_specific_prob': 0.60,
            'problem_class_prob': 0.20,
            'general_prob': 0.20,
        }
    }

    final_path = os.path.join(checkpoint_dir, "unified_mapper.pth")
    torch.save(checkpoint_data, final_path)

    print(f"✓ Final model saved to: {final_path}")
    print(f"\nTraining Complete!")
    print(f"{'='*70}\n")

    # Print summary
    print("Training Summary:")
    print(f"  Total programs: {len(unified_df)}")
    print(f"  Tasks trained: {len(set(unified_df['task']))}")
    print(f"  Model parameters: {num_params:,}")
    print(f"  Prompt augmentation: 60% task-specific, 20% problem-class, 20% general")
    print(f"  Checkpoint: {final_path}")
    print()


if __name__ == "__main__":
    main()
