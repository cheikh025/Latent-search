"""
Conditional Multi-Task Normalizing Flow Training

Trains a SINGLE normalizing flow with TASK-CONDITIONAL priors on embeddings from ALL tasks.
Based on the ACL 2023 paper "Controllable Text Generation via Probability Density Estimation
in the Latent Space" (Gu et al., 2023).

Key difference from unconditional flow (train_unified_flow.py):
- Instead of mapping all z to a single N(0,I), we learn per-task Gaussian priors π(u|τ) = N(μ_τ, σ²_τ)
- The loss becomes: L = -∑_(z,τ) [log π(F(z)|τ) + log|det dF(z)/dz|]
- This allows task-specific control in prior space while sharing the flow transformation

Training Objective (Equation 3 from paper):
    L = -∑_(x,a) [log π(F_θ(x)|a) + log |det dF_θ(x)/dx|]

Usage:
    python train_conditional_flow.py
    python train_conditional_flow.py --resume Flow_Checkpoints/conditional_flow_epoch100.pth
"""

import os
import json
import glob
import math
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from normalizing_flow import NormalizingFlow
from utils import is_valid_python
from model_config import DEFAULT_ENCODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder


LOG_2PI = math.log(2 * math.pi)


# ============================================================================
# Task-Conditional Prior Module (from paper Section 3.2)
# ============================================================================

class TaskConditionalPrior(nn.Module):
    """
    Learnable task-conditional Gaussian prior π(u|τ) = N(μ_τ, σ²_τ).

    Each task τ has its own learnable mean μ_τ and log-std log(σ_τ).
    The covariance is diagonal: Σ_τ = diag(σ²_τ).

    From the paper (Section 3.2):
    "For the convenience of control, we set covariance matrices Σ ∈ R^{n×n}
    of prior distributions as diagonal matrices σ² = σσ^T I, where π(z|a) = N(μ_a, σ²_a)."
    """

    def __init__(self, num_tasks: int, dim: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim

        # Learnable per-task mean and log-std
        self.mu = nn.Embedding(num_tasks, dim)
        self.log_sigma = nn.Embedding(num_tasks, dim)

        # Initialize: μ=0, σ=1 (log σ = 0)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.log_sigma.weight)

    def log_prob(self, u: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute log π(u|τ) = log N(u; μ_τ, diag(σ²_τ))

        Args:
            u: [B, D] - samples in prior space
            task_ids: [B] - task indices

        Returns:
            log_prob: [B] - log probability for each sample
        """
        mu = self.mu(task_ids)                    # [B, D]
        log_sigma = self.log_sigma(task_ids)      # [B, D]

        # log N(u; μ, σ²) = -0.5 * [((u-μ)/σ)² + 2*log(σ) + log(2π)]
        # Summed over all dimensions
        inv_sigma = torch.exp(-log_sigma)
        quad = ((u - mu) * inv_sigma) ** 2
        log_prob = -0.5 * (quad + 2 * log_sigma + LOG_2PI).sum(dim=1)  # [B]

        return log_prob

    def get_task_params(self, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get μ and σ for a specific task."""
        task_tensor = torch.tensor([task_id], device=self.mu.weight.device)
        mu = self.mu(task_tensor).squeeze(0)
        sigma = torch.exp(self.log_sigma(task_tensor)).squeeze(0)
        return mu, sigma

    def sample(self, task_id: int, num_samples: int = 1,
               device: str = 'cuda', lambda_scale: float = 1.0) -> torch.Tensor:
        """
        Sample from the task-conditional prior π(u|τ).

        Args:
            task_id: Task index
            num_samples: Number of samples
            device: Device for sampling
            lambda_scale: Scale factor for sampling variance (paper's λ parameter)

        Returns:
            samples: [num_samples, D]
        """
        mu, sigma = self.get_task_params(task_id)
        eps = torch.randn(num_samples, self.dim, device=device)
        return mu + sigma * lambda_scale * eps


# ============================================================================
# Conditional Flow Loss Function (Equation 3 from paper)
# ============================================================================

def compute_conditional_flow_loss(
    flow_model: NormalizingFlow,
    prior_model: TaskConditionalPrior,
    z_batch: torch.Tensor,
    task_ids: torch.Tensor
) -> torch.Tensor:
    """
    Compute the conditional flow loss from Equation 3 of the paper:

    L = -∑_(x,a) [log π(F_θ(x)|a) + log |det dF_θ(x)/dx|]

    Args:
        flow_model: The normalizing flow F_θ
        prior_model: Task-conditional prior π(·|a)
        z_batch: [B, D] - input embeddings
        task_ids: [B] - task indices

    Returns:
        loss: Scalar - mean negative log-likelihood
    """
    # Forward pass through flow: z → u
    u, log_det = flow_model(z_batch)  # u: [B, D], log_det: [B] or [B, 1]

    if log_det.ndim > 1:
        log_det = log_det.squeeze(-1)

    # Compute log probability under task-conditional prior
    log_pu = prior_model.log_prob(u, task_ids)  # [B]

    # Negative log-likelihood: -(log π(u|task) + log|det|)
    nll = -(log_pu + log_det)

    return nll.mean()


# ============================================================================
# Data Loading Functions (Same as Unified Mapper/Flow)
# ============================================================================

def load_all_augmented_heuristics(task_dir: str = "task") -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Load all augmented heuristics from all tasks.

    Returns:
        List of (code, task_name) tuples
        Dictionary of task counts
    """
    all_heuristics = []
    augmented_files = glob.glob(os.path.join(task_dir, "*", "augmented.json"))

    print(f"\n{'='*70}")
    print(f"Loading Augmented Heuristics from All Tasks")
    print(f"{'='*70}")
    print(f"Found {len(augmented_files)} task directories\n")

    task_counts = {}

    for aug_file in sorted(augmented_files):
        task_name = Path(aug_file).parent.name

        try:
            with open(aug_file, 'r', encoding='utf-8') as f:
                heuristics_dict = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load {aug_file}: {e}")
            continue

        if not heuristics_dict:
            warnings.warn(f"Empty augmented.json for task '{task_name}', skipping...")
            continue

        valid_count = 0
        for heuristic_name, code in heuristics_dict.items():
            if not code or not isinstance(code, str):
                continue
            if not is_valid_python(code):
                warnings.warn(f"Invalid Python syntax in {task_name}/{heuristic_name}, skipping...")
                continue
            all_heuristics.append((code, task_name))
            valid_count += 1

        task_counts[task_name] = valid_count
        print(f"  {task_name:25s}: {valid_count:3d} heuristics")

    print(f"\n{'='*70}")
    print(f"Total: {len(all_heuristics)} valid heuristics across {len(task_counts)} tasks")
    print(f"{'='*70}\n")

    return all_heuristics, task_counts


def encode_all_heuristics(
    heuristics_list: List[Tuple[str, str]],
    encoder_model: SentenceTransformer,
    device: str = "cuda",
    batch_size: int = 64
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Encode all heuristics using the code encoder.

    Returns:
        z_combined: Combined embeddings [total_programs, embedding_dim]
        metadata_df: DataFrame with task names and codes
    """
    print(f"\n{'='*70}")
    print(f"Encoding All Heuristics")
    print(f"{'='*70}\n")

    codes = [h[0] for h in heuristics_list]
    tasks = [h[1] for h in heuristics_list]

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

    embeddings = np.vstack(embeddings).astype(np.float32)

    print(f"\n✓ Encoded all programs")
    print(f"  Embedding shape: {embeddings.shape}")

    metadata_df = pd.DataFrame({
        'task': tasks,
        'code': codes
    })

    print(f"\nDataset Statistics:")
    print(f"{'='*70}")
    task_counts = metadata_df['task'].value_counts().sort_index()
    for task, count in task_counts.items():
        print(f"  {task:25s}: {count:3d} programs")
    print(f"{'='*70}\n")

    return embeddings, metadata_df


def create_task_mapping(metadata_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    """
    Create task ID mappings.

    Returns:
        task_to_id: Dict mapping task name to integer ID
        id_to_task: Dict mapping integer ID to task name
        task_ids: Array of task IDs aligned with the data
    """
    task_names = metadata_df["task"].tolist()
    unique_tasks = sorted(set(task_names))

    task_to_id = {t: i for i, t in enumerate(unique_tasks)}
    id_to_task = {i: t for t, i in task_to_id.items()}

    task_ids = np.array([task_to_id[t] for t in task_names], dtype=np.int64)

    print(f"\nTask ID Mapping:")
    print(f"{'='*70}")
    for task, tid in task_to_id.items():
        count = (task_ids == tid).sum()
        print(f"  {tid}: {task:25s} ({count} samples)")
    print(f"{'='*70}\n")

    return task_to_id, id_to_task, task_ids


def create_balanced_sampler(task_ids: np.ndarray, task_counts: Dict[str, int]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to balance tasks per epoch.

    From the paper (Section 3.2):
    "It's worth noting that the amount of training data for different attributes
    should be consistent as possible to ensure the balance of the transformation."
    """
    # Compute sample weights: inverse of class frequency
    unique_tasks, counts = np.unique(task_ids, return_counts=True)
    total_samples = len(task_ids)

    # Weight for each class = total / (num_classes * class_count)
    class_weights = total_samples / (len(unique_tasks) * counts)

    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[t] for t in task_ids])

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(task_ids),
        replacement=True
    )

    print(f"Balanced Sampling Weights:")
    for tid, weight in enumerate(class_weights):
        print(f"  Task {tid}: weight = {weight:.4f}")

    return sampler


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(
    epoch: int,
    flow_model: NormalizingFlow,
    prior_model: TaskConditionalPrior,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: float,
    checkpoint_path: str,
    metadata: Optional[Dict] = None
):
    """Save training checkpoint with full state and metadata."""
    checkpoint_data = {
        'epoch': epoch,
        'flow_state_dict': flow_model.state_dict(),
        'prior_state_dict': prior_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'flow_dim': flow_model.dim,
        'flow_num_layers': flow_model.num_layers,
        'prior_num_tasks': prior_model.num_tasks,
        'prior_dim': prior_model.dim,
    }

    if metadata:
        checkpoint_data.update(metadata)

    torch.save(checkpoint_data, checkpoint_path)


def load_checkpoint_for_resume(
    checkpoint_path: str,
    flow_model: NormalizingFlow,
    prior_model: TaskConditionalPrior,
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

    flow_model.load_state_dict(checkpoint['flow_state_dict'])
    print(f"✓ Loaded flow model state")

    prior_model.load_state_dict(checkpoint['prior_state_dict'])
    print(f"✓ Loaded prior model state")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded optimizer state")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Loaded scheduler state")

    start_epoch = checkpoint.get('epoch', 0)

    print(f"\nCheckpoint Metadata:")
    print(f"  Epoch: {start_epoch}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"  Total programs: {checkpoint.get('total_programs', 'N/A')}")
    print(f"  Tasks: {checkpoint.get('num_tasks', 'N/A')}")
    print(f"{'='*70}\n")

    return start_epoch, checkpoint


# ============================================================================
# Training Function
# ============================================================================

def train_conditional_flow(
    flow_model: NormalizingFlow,
    prior_model: TaskConditionalPrior,
    z_combined: np.ndarray,
    task_ids: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    batch_size: int = 128,
    epochs: int = 200,
    holdout_ratio: float = 0.1,
    device: str = 'cuda',
    verbose: bool = True,
    checkpoint_dir: str = "Flow_Checkpoints",
    start_epoch: int = 0,
    save_every: int = 20,
    use_balanced_sampling: bool = True,
    metadata: Optional[Dict] = None
) -> Tuple[NormalizingFlow, TaskConditionalPrior]:
    """
    Train conditional normalizing flow on combined embeddings from all tasks.

    The flow learns a shared transformation F_θ while the prior learns
    task-specific Gaussian parameters (μ_τ, σ_τ).
    """
    print(f"\n{'='*70}")
    print(f"Training: Conditional Multi-Task Normalizing Flow")
    print(f"{'='*70}\n")

    # Prepare tensors
    z_tensor = torch.tensor(z_combined, dtype=torch.float32)
    t_tensor = torch.tensor(task_ids, dtype=torch.long)

    # Split into train and validation sets (stratified by task)
    n_total = len(z_tensor)
    n_val = int(n_total * holdout_ratio)
    n_train = n_total - n_val

    # Shuffle indices for random split
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    z_train, t_train = z_tensor[train_indices], t_tensor[train_indices]
    z_val, t_val = z_tensor[val_indices], t_tensor[val_indices]

    print(f"Data Split:")
    print(f"  Total samples: {n_total}")
    print(f"  Training: {n_train} ({(1-holdout_ratio)*100:.1f}%)")
    print(f"  Validation: {n_val} ({holdout_ratio*100:.1f}%)")
    print()

    # Create datasets
    train_dataset = TensorDataset(z_train, t_train)
    val_dataset = TensorDataset(z_val, t_val)

    # Create balanced sampler for training (paper requirement)
    if use_balanced_sampling:
        print("Using balanced sampling across tasks...")
        train_task_ids_np = t_train.numpy()
        unique_tasks, counts = np.unique(train_task_ids_np, return_counts=True)
        total_train = len(train_task_ids_np)
        class_weights = total_train / (len(unique_tasks) * counts)
        sample_weights = np.array([class_weights[t] for t in train_task_ids_np])

        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_task_ids_np),
            replacement=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # Training DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Validation DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # Move models to device
    flow_model = flow_model.to(device)
    prior_model = prior_model.to(device)

    print(f"Training Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {start_epoch} → {epochs}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Balanced sampling: {use_balanced_sampling}")
    print(f"  Batches per epoch: {len(train_dataloader)} (train), {len(val_dataloader)} (val)")
    print(f"  Device: {device}")

    print(f"\nFlow Model Architecture:")
    print(f"  Dimension: {flow_model.dim}")
    print(f"  Layers: {flow_model.num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in flow_model.parameters()):,}")

    print(f"\nPrior Model:")
    print(f"  Number of tasks: {prior_model.num_tasks}")
    print(f"  Dimension: {prior_model.dim}")
    print(f"  Parameters: {sum(p.numel() for p in prior_model.parameters()):,}")

    print(f"\nStarting training...\n")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        # ===== Training Phase =====
        flow_model.train()
        prior_model.train()

        running_train_loss = torch.tensor(0.0, device=device)
        num_train_batches = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for z_batch, t_batch in pbar:
            z_batch = z_batch.to(device, non_blocking=True)
            t_batch = t_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Compute conditional flow loss (Equation 3 from paper)
            loss = compute_conditional_flow_loss(flow_model, prior_model, z_batch, t_batch)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(prior_model.parameters(), max_norm=5.0)

            optimizer.step()

            running_train_loss += loss.detach()
            num_train_batches += 1

            if num_train_batches % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_train_loss.item() / max(num_train_batches, 1)

        # ===== Validation Phase =====
        flow_model.eval()
        prior_model.eval()

        running_val_loss = torch.tensor(0.0, device=device)
        num_val_batches = 0

        with torch.no_grad():
            for z_batch, t_batch in val_dataloader:
                z_batch = z_batch.to(device, non_blocking=True)
                t_batch = t_batch.to(device, non_blocking=True)

                loss = compute_conditional_flow_loss(flow_model, prior_model, z_batch, t_batch)
                running_val_loss += loss.detach()
                num_val_batches += 1

        avg_val_loss = running_val_loss.item() / max(num_val_batches, 1)

        # Update scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Track best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Best Val: {best_val_loss:.4f} | LR: {current_lr:.2e}")

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"conditional_flow_epoch{epoch+1}.pth")
            checkpoint_metadata = metadata.copy() if metadata else {}
            checkpoint_metadata.update({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            })
            save_checkpoint(
                epoch=epoch + 1,
                flow_model=flow_model,
                prior_model=prior_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=avg_val_loss,
                checkpoint_path=checkpoint_path,
                metadata=checkpoint_metadata
            )
            if verbose:
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Save best model separately
        if avg_val_loss == best_val_loss:
            best_checkpoint_path = os.path.join(checkpoint_dir, "conditional_flow_best.pth")
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
                prior_model=prior_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=avg_val_loss,
                checkpoint_path=best_checkpoint_path,
                metadata=checkpoint_metadata
            )
            if verbose:
                print(f"  ✓ Best model saved: {best_checkpoint_path}")

    # Clear cache at end of training
    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    return flow_model, prior_model


# ============================================================================
# Validation Functions
# ============================================================================

def validate_conditional_flow(
    flow_model: NormalizingFlow,
    prior_model: TaskConditionalPrior,
    z_combined: np.ndarray,
    task_ids: np.ndarray,
    id_to_task: Dict[int, str],
    device: str = 'cuda'
):
    """
    Validate the trained conditional flow model.

    For conditional flows, we check that standardized samples per task look normal:
    ũ = (u - μ_τ) / σ_τ ≈ N(0, I)

    This is different from unconditional validation which checks global u_mean ≈ 0, u_std ≈ 1.
    """
    print(f"\n{'='*70}")
    print(f"Validating Conditional Flow Model (Per-Task)")
    print(f"{'='*70}\n")

    flow_model.eval()
    prior_model.eval()
    flow_model = flow_model.to(device)
    prior_model = prior_model.to(device)

    # Sample for validation
    sample_size = min(10000, len(z_combined))
    indices = np.random.choice(len(z_combined), sample_size, replace=False)
    z_sample = torch.tensor(z_combined[indices], dtype=torch.float32, device=device)
    t_sample = torch.tensor(task_ids[indices], dtype=torch.long, device=device)

    with torch.no_grad():
        # Forward pass
        u, log_det = flow_model(z_sample)

        # Inverse pass for reconstruction check
        z_reconstructed = flow_model.inverse(u)

        # Compute global metrics
        reconstruction_error = torch.mean((z_sample - z_reconstructed) ** 2).item()
        log_det_mean = log_det.mean().item()
        log_det_std = log_det.std().item()

    print(f"Global Metrics (on {sample_size} samples):")
    print(f"{'='*70}")
    print(f"  Reconstruction Error: {reconstruction_error:.6f} {'✓' if reconstruction_error < 1e-4 else '✗'}")
    print(f"  Log Det Mean: {log_det_mean:.4f}")
    print(f"  Log Det Std: {log_det_std:.4f}")
    print()

    # Per-task validation
    print(f"Per-Task Prior Statistics:")
    print(f"{'='*70}")
    print(f"{'Task':<25} {'μ mean':>10} {'μ std':>10} {'σ mean':>10} {'σ std':>10} {'ũ mean':>10} {'ũ std':>10}")
    print(f"{'-'*95}")

    unique_tasks = np.unique(task_ids[indices])
    all_passed = True

    for tid in unique_tasks:
        task_name = id_to_task[tid]
        task_mask = t_sample == tid
        u_task = u[task_mask]

        # Get learned prior parameters
        mu, sigma = prior_model.get_task_params(tid)
        mu = mu.cpu()
        sigma = sigma.cpu()

        # Compute standardized samples: ũ = (u - μ) / σ
        u_standardized = (u_task.cpu() - mu) / sigma

        # Check if standardized samples look like N(0, I)
        u_std_mean = u_standardized.mean(dim=0).mean().item()
        u_std_std = u_standardized.std(dim=0).mean().item()

        # Check conditions
        mean_ok = abs(u_std_mean) < 0.2
        std_ok = abs(u_std_std - 1.0) < 0.3

        if not (mean_ok and std_ok):
            all_passed = False

        status = '✓' if (mean_ok and std_ok) else '✗'

        print(f"{task_name:<25} {mu.mean().item():>10.4f} {mu.std().item():>10.4f} "
              f"{sigma.mean().item():>10.4f} {sigma.std().item():>10.4f} "
              f"{u_std_mean:>10.4f} {u_std_std:>10.4f} {status}")

    print(f"{'='*70}")

    if reconstruction_error < 1e-4 and all_passed:
        print("\n✓ Conditional flow model validation PASSED!")
    else:
        print("\n⚠ Conditional flow model validation shows potential issues.")
        if reconstruction_error >= 1e-4:
            print("  - Reconstruction error too high")
        if not all_passed:
            print("  - Some tasks have non-standard prior distributions")
    print()


def print_learned_priors(
    prior_model: TaskConditionalPrior,
    id_to_task: Dict[int, str],
    device: str = 'cuda'
):
    """Print the learned prior parameters for each task."""
    print(f"\n{'='*70}")
    print(f"Learned Task-Conditional Prior Parameters")
    print(f"{'='*70}\n")

    prior_model = prior_model.to(device)
    prior_model.eval()

    for tid, task_name in id_to_task.items():
        mu, sigma = prior_model.get_task_params(tid)
        mu = mu.cpu()
        sigma = sigma.cpu()

        print(f"Task {tid}: {task_name}")
        print(f"  μ: mean={mu.mean().item():.4f}, std={mu.std().item():.4f}, "
              f"min={mu.min().item():.4f}, max={mu.max().item():.4f}")
        print(f"  σ: mean={sigma.mean().item():.4f}, std={sigma.std().item():.4f}, "
              f"min={sigma.min().item():.4f}, max={sigma.max().item():.4f}")
        print()


# ============================================================================
# Main Training Script
# ============================================================================

def main(
    resume_checkpoint: Optional[str] = None,
    holdout_ratio: float = 0.1,
    dropout: float = 0.1,
    use_balanced_sampling: bool = True,
    encoder_name: str = None,
    embedding_dim: int = None
):
    """Main training pipeline for conditional normalizing flow.

    Args:
        resume_checkpoint: Path to checkpoint file to resume training from.
        holdout_ratio: Fraction of data to hold out for validation.
        dropout: Dropout probability for regularization.
        use_balanced_sampling: Whether to use balanced sampling across tasks.
        encoder_name: Encoder model name. Defaults to DEFAULT_ENCODER.
        embedding_dim: Matryoshka embedding dimension. If None, uses DEFAULT_MATRYOSHKA_DIM.
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

    # ========================================================================
    # Step 1: Load All Augmented Heuristics
    # ========================================================================

    heuristics_list, task_counts = load_all_augmented_heuristics(task_dir="task")

    if len(heuristics_list) == 0:
        raise ValueError("No valid heuristics loaded! Check task directories.")

    # ========================================================================
    # Step 2: Load Encoder Model
    # ========================================================================

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

    # ========================================================================
    # Step 3: Encode All Heuristics
    # ========================================================================

    z_combined, metadata_df = encode_all_heuristics(
        heuristics_list=heuristics_list,
        encoder_model=encoder_model,
        device=device,
        batch_size=64
    )

    # Free encoder memory
    del encoder_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ========================================================================
    # Step 4: Create Task ID Mapping
    # ========================================================================

    task_to_id, id_to_task, task_ids = create_task_mapping(metadata_df)
    num_tasks = len(task_to_id)

    # ========================================================================
    # Step 5: Initialize Models
    # ========================================================================

    print(f"{'='*70}")
    print("Initializing Models")
    print(f"{'='*70}\n")

    embedding_dim = z_combined.shape[1]
    num_layers = 4
    hidden_dim = 128

    # Initialize flow model (same architecture as unconditional)
    flow_model = NormalizingFlow(
        dim=embedding_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    # Initialize task-conditional prior
    prior_model = TaskConditionalPrior(
        num_tasks=num_tasks,
        dim=embedding_dim
    )

    flow_params = sum(p.numel() for p in flow_model.parameters())
    prior_params = sum(p.numel() for p in prior_model.parameters())

    print(f"Flow Model:")
    print(f"  Dimension: {embedding_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Parameters: {flow_params:,}")
    print()
    print(f"Prior Model:")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Dimension: {embedding_dim}")
    print(f"  Parameters: {prior_params:,}")
    print()
    print(f"Total Parameters: {flow_params + prior_params:,}")
    print()

    # ========================================================================
    # Step 6: Setup Optimizer and Scheduler
    # ========================================================================

    learning_rate = 1e-3

    # Joint optimizer for both flow and prior parameters
    optimizer = torch.optim.AdamW(
        list(flow_model.parameters()) + list(prior_model.parameters()),
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

    # ========================================================================
    # Step 7: Load Checkpoint if Resuming
    # ========================================================================

    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch, _ = load_checkpoint_for_resume(
            checkpoint_path=resume_checkpoint,
            flow_model=flow_model,
            prior_model=prior_model,
            optimizer=optimizer,
            scheduler=scheduler
        )

    # ========================================================================
    # Step 8: Create Checkpoint Directory
    # ========================================================================

    checkpoint_dir = "Flow_Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================================
    # Step 9: Prepare Metadata for Checkpoints
    # ========================================================================

    training_metadata = {
        'tasks_trained': list(task_counts.keys()),
        'task_to_id': task_to_id,
        'id_to_task': id_to_task,
        'num_tasks': num_tasks,
        'total_programs': len(z_combined),
        'programs_per_task': task_counts,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'encoder': 'BAAI/bge-code-v1',
        'optimizer': 'AdamW',
        'initial_lr': learning_rate,
        'scheduler': 'ReduceLROnPlateau',
        'balanced_sampling': use_balanced_sampling,
        'flow_type': 'conditional',
    }

    # ========================================================================
    # Step 10: Train Conditional Flow
    # ========================================================================

    trained_flow, trained_prior = train_conditional_flow(
        flow_model=flow_model,
        prior_model=prior_model,
        z_combined=z_combined,
        task_ids=task_ids,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=128,
        epochs=200,
        holdout_ratio=holdout_ratio,
        device=device,
        verbose=True,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        save_every=20,
        use_balanced_sampling=use_balanced_sampling,
        metadata=training_metadata
    )

    # ========================================================================
    # Step 11: Validate Trained Flow
    # ========================================================================

    validate_conditional_flow(
        trained_flow, trained_prior, z_combined, task_ids, id_to_task, device=device
    )

    # Print learned prior parameters
    print_learned_priors(trained_prior, id_to_task, device=device)

    # ========================================================================
    # Step 12: Save Final Checkpoint
    # ========================================================================

    print(f"{'='*70}")
    print("Saving Final Checkpoint")
    print(f"{'='*70}\n")

    final_checkpoint = {
        'flow_state_dict': trained_flow.state_dict(),
        'prior_state_dict': trained_prior.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': 200,
        'flow_dim': embedding_dim,
        'flow_num_layers': num_layers,
        'flow_hidden_dim': hidden_dim,
        'flow_dropout': dropout,
        'prior_num_tasks': num_tasks,
        'prior_dim': embedding_dim,
        'task_to_id': task_to_id,
        'id_to_task': id_to_task,
        'tasks_trained': list(task_counts.keys()),
        'total_programs': len(z_combined),
        'programs_per_task': task_counts,
        'encoder': 'BAAI/bge-code-v1',
        'architecture': 'Conditional RealNVP with Task-Specific Gaussian Priors',
        'training': {
            'optimizer': 'AdamW',
            'initial_lr': learning_rate,
            'weight_decay': 1e-5,
            'scheduler': 'ReduceLROnPlateau',
            'batch_size': 128,
            'epochs': 200,
            'holdout_ratio': holdout_ratio,
            'dropout': dropout,
            'gradient_clipping': 5.0,
            'balanced_sampling': use_balanced_sampling,
        }
    }

    final_path = os.path.join(checkpoint_dir, "conditional_flow_final.pth")
    torch.save(final_checkpoint, final_path)

    print(f"✓ Final model saved to: {final_path}")
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    print(f"  Total programs: {len(z_combined):,}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Training split: {int((1-holdout_ratio)*len(z_combined))} train, {int(holdout_ratio*len(z_combined))} val")
    print(f"  Flow parameters: {flow_params:,}")
    print(f"  Prior parameters: {prior_params:,}")
    print(f"  Architecture: Conditional RealNVP with Task-Specific Gaussian Priors")
    print(f"  Balanced sampling: {use_balanced_sampling}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Best checkpoint: {os.path.join(checkpoint_dir, 'conditional_flow_best.pth')}")
    print(f"{'='*70}\n")

    print("✓ Conditional normalizing flow training complete!")
    print("\nYou can now use this flow for:")
    print("  1. Task-specific sampling from π(u|τ) = N(μ_τ, σ²_τ)")
    print("  2. Crossover evolution with task-aware interpolation")
    print("  3. Control strength adjustment (paper Section 3.3.2)")
    print("  4. Multi-task generation with task conditioning")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conditional multi-task normalizing flow training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
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
        "--no_balanced_sampling",
        action="store_true",
        help="Disable balanced sampling across tasks"
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
        resume_checkpoint=args.resume,
        holdout_ratio=args.holdout_ratio,
        dropout=args.dropout,
        use_balanced_sampling=not args.no_balanced_sampling,
        encoder_name=args.encoder,
        embedding_dim=getattr(args, 'embedding_dim', None)
    )
