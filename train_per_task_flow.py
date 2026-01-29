"""
Per-Task Normalizing Flow Training

Trains a SEPARATE normalizing flow for EACH task.
Unlike train_unified_flow.py which trains a single shared flow,
this script creates individual flow models per task.

Each task gets its own:
- Flow model trained on its specific heuristics
- Checkpoint file saved in task-specific directory

Usage:
    python train_per_task_flow.py
    python train_per_task_flow.py --task tsp_construct
    python train_per_task_flow.py --resume task/tsp_construct/flow_checkpoint.pth --task tsp_construct
"""

import os
import json
import glob
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from normalizing_flow import NormalizingFlow, compute_flow_loss
from model_config import DEFAULT_ENCODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder
from utils import is_valid_python


# ============================================================================
# Data Loading Functions (Per-Task)
# ============================================================================

def get_available_tasks(task_dir: str = "task") -> List[str]:
    """Get list of available tasks with augmented.json files."""
    augmented_files = glob.glob(os.path.join(task_dir, "*", "augmented.json"))
    tasks = [Path(f).parent.name for f in sorted(augmented_files)]
    return tasks


def load_task_heuristics(task_name: str, task_dir: str = "task") -> List[str]:
    """
    Load heuristics for a specific task.

    Returns:
        List of code strings
    """
    aug_file = os.path.join(task_dir, task_name, "augmented.json")

    if not os.path.exists(aug_file):
        raise FileNotFoundError(f"No augmented.json found for task: {task_name}")

    print(f"\n{'='*70}")
    print(f"Loading Heuristics for Task: {task_name}")
    print(f"{'='*70}")

    with open(aug_file, 'r', encoding='utf-8') as f:
        heuristics_dict = json.load(f)

    if not heuristics_dict:
        raise ValueError(f"Empty augmented.json for task '{task_name}'")

    heuristics = []
    invalid_count = 0

    for heuristic_name, code in heuristics_dict.items():
        if not code or not isinstance(code, str):
            invalid_count += 1
            continue
        if not is_valid_python(code):
            warnings.warn(f"Invalid Python syntax in {heuristic_name}, skipping...")
            invalid_count += 1
            continue
        heuristics.append(code)

    print(f"  Valid heuristics: {len(heuristics)}")
    if invalid_count > 0:
        print(f"  Skipped (invalid): {invalid_count}")
    print(f"{'='*70}\n")

    return heuristics


def encode_heuristics(
    heuristics: List[str],
    encoder_model: SentenceTransformer,
    device: str = "cuda",
    batch_size: int = 64
) -> np.ndarray:
    """
    Encode heuristics using the code encoder.

    Returns:
        embeddings: [num_programs, embedding_dim]
    """
    print(f"Encoding {len(heuristics)} programs...")

    encoder_model.eval()
    encoder_model.to(device)

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(heuristics), batch_size), desc="Encoding"):
            batch_codes = heuristics[i:i+batch_size]
            batch_embeddings = encoder_model.encode(
                batch_codes,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings).astype(np.float32)
    print(f"Embedding shape: {embeddings.shape}\n")

    return embeddings


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint_for_resume(
    checkpoint_path: str,
    flow_model: NormalizingFlow,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[int, Dict]:
    """Load checkpoint for resuming training."""
    print(f"\n{'='*70}")
    print(f"Loading Checkpoint for Resume Training")
    print(f"{'='*70}\n")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model state from: {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded scheduler state")

    start_epoch = checkpoint.get('epoch', 0)

    print(f"\nCheckpoint Metadata:")
    print(f"  Epoch: {start_epoch}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"  Task: {checkpoint.get('task', 'N/A')}")
    print(f"  Programs: {checkpoint.get('num_programs', 'N/A')}")
    print(f"{'='*70}\n")

    return start_epoch, checkpoint


def save_checkpoint(
    epoch: int,
    flow_model: NormalizingFlow,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: float,
    checkpoint_path: str,
    metadata: Optional[Dict] = None
):
    """Save training checkpoint with full state and metadata."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': flow_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'dim': flow_model.dim,
        'num_layers': flow_model.num_layers,
    }

    if metadata:
        checkpoint_data.update(metadata)

    torch.save(checkpoint_data, checkpoint_path)


# ============================================================================
# Training Function
# ============================================================================

def train_task_flow(
    flow_model: NormalizingFlow,
    z_embeddings: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    task_name: str,
    batch_size: int = 64,
    epochs: int = 200,
    holdout_ratio: float = 0.1,
    device: str = 'cuda',
    verbose: bool = True,
    checkpoint_dir: str = None,
    start_epoch: int = 0,
    save_every: int = 20,
    metadata: Optional[Dict] = None
) -> NormalizingFlow:
    """
    Train normalizing flow on embeddings from a single task.
    """
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*70}")
    print(f"Training Flow for Task: {task_name}")
    print(f"{'='*70}\n")

    # Prepare data with train/val split
    z_tensor = torch.tensor(z_embeddings, dtype=torch.float32)

    n_total = len(z_tensor)
    n_val = max(1, int(n_total * holdout_ratio))
    n_train = n_total - n_val

    # Shuffle indices for random split
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    z_train = z_tensor[train_indices]
    z_val = z_tensor[val_indices]

    print(f"Data Split:")
    print(f"  Total samples: {n_total}")
    print(f"  Training: {n_train} ({(1-holdout_ratio)*100:.1f}%)")
    print(f"  Validation: {n_val} ({holdout_ratio*100:.1f}%)")

    # Adjust batch size for small datasets
    actual_batch_size = min(batch_size, n_train // 2) if n_train > 2 else 1
    if actual_batch_size != batch_size:
        print(f"  Adjusted batch size: {actual_batch_size} (dataset too small)")
    print()

    # Training DataLoader
    train_dataset = TensorDataset(z_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=len(z_train) > actual_batch_size
    )

    # Validation DataLoader
    val_dataset = TensorDataset(z_val)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # Move model to device
    flow_model = flow_model.to(device)
    flow_model.train()

    print(f"Training Configuration:")
    print(f"  Batch size: {actual_batch_size}")
    print(f"  Epochs: {start_epoch} -> {epochs}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Batches per epoch: {len(train_dataloader)} (train), {len(val_dataloader)} (val)")
    print(f"  Device: {device}")

    if start_epoch > 0:
        print(f"\n  Resuming from epoch {start_epoch}")

    print(f"\nFlow Model Architecture:")
    print(f"  Dimension: {flow_model.dim}")
    print(f"  Layers: {flow_model.num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    print(f"\nStarting training...\n")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        # ===== Training Phase =====
        flow_model.train()
        running_train_loss = torch.tensor(0.0, device=device)
        num_train_batches = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for (z_batch,) in pbar:
            z_batch = z_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_flow_loss(flow_model, z_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=5.0)
            optimizer.step()

            running_train_loss += loss.detach()
            num_train_batches += 1

            if num_train_batches % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_train_loss.item() / max(num_train_batches, 1)

        # ===== Validation Phase =====
        flow_model.eval()
        running_val_loss = torch.tensor(0.0, device=device)
        num_val_batches = 0

        with torch.no_grad():
            for (z_batch,) in val_dataloader:
                z_batch = z_batch.to(device, non_blocking=True)
                loss = compute_flow_loss(flow_model, z_batch)
                running_val_loss += loss.detach()
                num_val_batches += 1

        avg_val_loss = running_val_loss.item() / max(num_val_batches, 1)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Best: {best_val_loss:.4f} | LR: {current_lr:.2e}")

        # Save checkpoint periodically
        if checkpoint_dir and (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"flow_epoch{epoch+1}.pth")
            checkpoint_metadata = metadata.copy() if metadata else {}
            checkpoint_metadata.update({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            })
            save_checkpoint(
                epoch=epoch + 1,
                flow_model=flow_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=avg_val_loss,
                checkpoint_path=checkpoint_path,
                metadata=checkpoint_metadata
            )
            if verbose:
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model separately
        if checkpoint_dir and avg_val_loss == best_val_loss:
            best_checkpoint_path = os.path.join(checkpoint_dir, "flow_best.pth")
            checkpoint_metadata = metadata.copy() if metadata else {}
            checkpoint_metadata.update({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'is_best': True
            })
            save_checkpoint(
                epoch=epoch + 1,
                flow_model=flow_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=avg_val_loss,
                checkpoint_path=best_checkpoint_path,
                metadata=checkpoint_metadata
            )

    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Training Complete for {task_name}!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    return flow_model


# ============================================================================
# Validation Function
# ============================================================================

def validate_flow(flow_model: NormalizingFlow, z_embeddings: np.ndarray, task_name: str, device: str = 'cuda'):
    """Validate the trained flow model."""
    print(f"\n{'='*70}")
    print(f"Validating Flow for Task: {task_name}")
    print(f"{'='*70}\n")

    flow_model.eval()
    flow_model = flow_model.to(device)

    sample_size = min(1000, len(z_embeddings))
    indices = np.random.choice(len(z_embeddings), sample_size, replace=False)
    z_sample = torch.tensor(z_embeddings[indices], dtype=torch.float32, device=device)

    with torch.no_grad():
        u, log_det = flow_model(z_sample)
        z_reconstructed = flow_model.inverse(u)

        reconstruction_error = torch.mean((z_sample - z_reconstructed) ** 2).item()
        u_mean = u.mean(dim=0).mean().item()
        u_std = u.std(dim=0).mean().item()
        log_det_mean = log_det.mean().item()
        log_det_std = log_det.std().item()

    print(f"Validation Results (on {sample_size} samples):")
    print(f"{'='*70}")
    print(f"  Reconstruction Error: {reconstruction_error:.6f} {'OK' if reconstruction_error < 1e-4 else 'HIGH'}")
    print(f"  Prior Mean: {u_mean:.4f} (target: 0.0) {'OK' if abs(u_mean) < 0.1 else 'HIGH'}")
    print(f"  Prior Std: {u_std:.4f} (target: 1.0) {'OK' if abs(u_std - 1.0) < 0.2 else 'HIGH'}")
    print(f"  Log Det Mean: {log_det_mean:.4f}")
    print(f"  Log Det Std: {log_det_std:.4f}")
    print(f"{'='*70}\n")

    if reconstruction_error < 1e-4 and abs(u_mean) < 0.1 and abs(u_std - 1.0) < 0.2:
        print("Flow model validation PASSED!")
    else:
        print("Flow model validation shows potential issues. Consider training longer.")
    print()


# ============================================================================
# Single Task Training
# ============================================================================

def train_single_task(
    task_name: str,
    encoder_model: SentenceTransformer,
    device: str = "cuda",
    epochs: int = 200,
    holdout_ratio: float = 0.1,
    dropout: float = 0.1,
    resume_checkpoint: Optional[str] = None,
    task_dir: str = "task"
):
    """Train flow for a single task."""
    print(f"\n{'#'*70}")
    print(f"# Training Flow for: {task_name}")
    print(f"{'#'*70}\n")

    # Load and encode heuristics
    heuristics = load_task_heuristics(task_name, task_dir=task_dir)

    if len(heuristics) < 5:
        print(f"Skipping {task_name}: only {len(heuristics)} heuristics (need at least 5)")
        return None

    z_embeddings = encode_heuristics(heuristics, encoder_model, device=device)

    # Initialize flow model
    embedding_dim = z_embeddings.shape[1]
    num_layers = 4
    hidden_dim = 128

    flow_model = NormalizingFlow(
        dim=embedding_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    # Setup optimizer and scheduler
    learning_rate = 1e-3
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch, _ = load_checkpoint_for_resume(
            checkpoint_path=resume_checkpoint,
            flow_model=flow_model,
            optimizer=optimizer,
            scheduler=scheduler
        )

    # Create checkpoint directory inside task folder
    checkpoint_dir = os.path.join(task_dir, task_name, "flow_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training metadata
    training_metadata = {
        'task': task_name,
        'num_programs': len(heuristics),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'encoder': DEFAULT_ENCODER,
        'optimizer': 'AdamW',
        'initial_lr': learning_rate,
    }

    # Train
    trained_flow = train_task_flow(
        flow_model=flow_model,
        z_embeddings=z_embeddings,
        optimizer=optimizer,
        scheduler=scheduler,
        task_name=task_name,
        batch_size=64,
        epochs=epochs,
        holdout_ratio=holdout_ratio,
        device=device,
        verbose=True,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        save_every=20,
        metadata=training_metadata
    )

    # Validate
    validate_flow(trained_flow, z_embeddings, task_name, device=device)

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "flow_final.pth")
    final_checkpoint = {
        'model_state_dict': trained_flow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epochs,
        'dim': embedding_dim,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'task': task_name,
        'num_programs': len(heuristics),
        'embedding_dim': embedding_dim,
        'encoder': DEFAULT_ENCODER,
        'architecture': 'RealNVP with ActNorm + Dropout',
    }
    torch.save(final_checkpoint, final_path)
    print(f"Final model saved to: {final_path}")

    return trained_flow


# ============================================================================
# Main Training Script
# ============================================================================

def main(
    task: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    holdout_ratio: float = 0.1,
    dropout: float = 0.1,
    epochs: int = 200,
    encoder_name: str = None,
    embedding_dim: int = None
):
    """Main training pipeline for per-task normalizing flows.

    Args:
        task: Specific task to train (if None, trains all tasks).
        resume_checkpoint: Path to checkpoint file to resume training from.
        holdout_ratio: Fraction of data to hold out for validation.
        dropout: Dropout probability for regularization.
        epochs: Number of training epochs.
        encoder_name: Encoder model name.
        embedding_dim: Matryoshka embedding dimension.
    """
    if encoder_name is None:
        encoder_name = DEFAULT_ENCODER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.set_float32_matmul_precision('high')
        print("TF32 matmul precision: enabled\n")

    # Load encoder (shared across all tasks)
    print(f"{'='*70}")
    print("Loading Encoder Model")
    print(f"{'='*70}\n")

    encoder_model, actual_embedding_dim = load_encoder(
        model_name=encoder_name,
        device=device,
        truncate_dim=embedding_dim
    )
    print(f"Encoder loaded ({encoder_name})")
    print(f"Embedding dimension: {actual_embedding_dim}\n")

    # Get tasks to train
    if task is not None:
        tasks = [task]
    else:
        tasks = get_available_tasks()

    print(f"{'='*70}")
    print(f"Tasks to Train: {len(tasks)}")
    print(f"{'='*70}")
    for t in tasks:
        print(f"  - {t}")
    print()

    # Train each task
    results = {}
    for task_name in tasks:
        try:
            # Only use resume checkpoint if training specific task
            task_resume = resume_checkpoint if (task is not None) else None

            trained_flow = train_single_task(
                task_name=task_name,
                encoder_model=encoder_model,
                device=device,
                epochs=epochs,
                holdout_ratio=holdout_ratio,
                dropout=dropout,
                resume_checkpoint=task_resume
            )
            results[task_name] = "SUCCESS" if trained_flow is not None else "SKIPPED"
        except Exception as e:
            print(f"Error training {task_name}: {e}")
            results[task_name] = f"FAILED: {e}"

    # Summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    for task_name, status in results.items():
        print(f"  {task_name:25s}: {status}")
    print(f"{'='*70}\n")

    # Free encoder memory
    del encoder_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    print("Per-task flow training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-task normalizing flow training"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to train (if not specified, trains all tasks)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (only for single task)"
    )
    parser.add_argument(
        "--holdout_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for validation (default: 0.1)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for regularization (default: 0.1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_ENCODER,
        help=f"Encoder model name (default: {DEFAULT_ENCODER})"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help=f"Matryoshka embedding dimension (default: {DEFAULT_MATRYOSHKA_DIM or 'model native'})"
    )
    args = parser.parse_args()

    main(
        task=args.task,
        resume_checkpoint=args.resume,
        holdout_ratio=args.holdout_ratio,
        dropout=args.dropout,
        epochs=args.epochs,
        encoder_name=args.encoder,
        embedding_dim=getattr(args, 'embedding_dim', None)
    )
