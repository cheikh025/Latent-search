"""
Ranking-based score predictor operating directly on z-space (CodeBERT embeddings).

No normalizing flow required - learns to rank programs from their embeddings directly.
From n programs, creates n*(n-1)/2 pairs for training.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm


class PairwiseDataset(Dataset):
    """Dataset of (z_better, z_worse, score_diff) pairs for ranking."""

    def __init__(self, z_better: torch.Tensor, z_worse: torch.Tensor, score_diffs: torch.Tensor):
        self.z_better = z_better
        self.z_worse = z_worse
        self.score_diffs = score_diffs

    def __len__(self):
        return len(self.z_better)

    def __getitem__(self, idx):
        return self.z_better[idx], self.z_worse[idx], self.score_diffs[idx]


class RankingScorePredictor(nn.Module):
    """
    MLP that predicts score from z-space vector (BAAI/bge-code-v1 embedding).
    Trained with ranking loss for better generalization on small datasets.

    R: R^d -> R
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict score from z-space vector."""
        return self.network(z)


def load_heuristics(task_name: str) -> Dict[str, str]:
    """Load heuristics from JSON file for a given task."""
    heuristics_path = Path(f"task/{task_name}/heuristics.json")

    if not heuristics_path.exists():
        raise FileNotFoundError(f"Heuristics file not found: {heuristics_path}")

    with open(heuristics_path, 'r') as f:
        programs = json.load(f)

    return programs


def get_evaluator(task_name: str):
    """Get the evaluator class for a given task."""
    task_evaluators = {
        'tsp_construct': ('task.tsp_construct.evaluation', 'TSPEvaluation'),
        'cvrp_construct': ('task.cvrp_construct.evaluation', 'CVRPEvaluation'),
        'vrptw_construct': ('task.vrptw_construct.evaluation', 'VRPTWEvaluation'),
        'jssp_construct': ('task.jssp_construct.evaluation', 'JSSPEvaluation'),
        'knapsack_construct': ('task.knapsack_construct.evaluation', 'KnapsackEvaluation'),
        'online_bin_packing': ('task.online_bin_packing.evaluation', 'BinPackingEvaluation'),
        'qap_construct': ('task.qap_construct.evaluation', 'QAPEvaluation'),
        'set_cover_construct': ('task.set_cover_construct.evaluation', 'SetCoverEvaluation'),
        'cflp_construct': ('task.cflp_construct.evaluation', 'CFLPEvaluation'),
        'admissible_set': ('task.admissible_set.evaluation', 'ASPEvaluation'),
    }

    if task_name not in task_evaluators:
        raise ValueError(f"Unknown task: {task_name}")

    module_path, class_name = task_evaluators[task_name]
    module = __import__(module_path, fromlist=[class_name])
    EvaluatorClass = getattr(module, class_name)
    return EvaluatorClass()


def evaluate_programs(task_name: str, programs: Dict[str, str], use_secure: bool = True) -> pd.DataFrame:
    """Evaluate all programs for a task and return DataFrame with scores."""
    evaluator = get_evaluator(task_name)

    if use_secure:
        from base.evaluate import SecureEvaluator
        evaluator = SecureEvaluator(evaluator, debug_mode=False)

    results = []
    for name, code in tqdm(programs.items(), desc=f"Evaluating {task_name}"):
        try:
            score = evaluator.evaluate_program(code)
            if score is not None and np.isfinite(score):
                results.append({'name': name, 'code': code, 'score': score})
        except Exception:
            pass

    return pd.DataFrame(results)


def get_encoder_model(device: str = 'cuda'):
    """
    Load the same encoder model used in unified training and programDB.
    Uses BAAI/bge-code-v1 SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformer

    encoder_model = SentenceTransformer(
        "BAAI/bge-code-v1",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    ).to(device)

    encoder_model.eval()
    return encoder_model


def encode_programs(codes: List[str], encoder_model=None, device: str = 'cuda', batch_size: int = 32) -> torch.Tensor:
    """
    Encode programs using BAAI/bge-code-v1 SentenceTransformer.
    Same encoding as used in unified mapper training and programDB.

    Args:
        codes: List of code strings
        encoder_model: Optional pre-loaded encoder (to avoid reloading)
        device: Device for encoding
        batch_size: Batch size for encoding

    Returns:
        Tensor of embeddings [n, 1024]
    """
    # Load encoder if not provided
    if encoder_model is None:
        encoder_model = get_encoder_model(device)

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Encoding programs"):
            batch_codes = codes[i:i+batch_size]
            batch_embeddings = encoder_model.encode(
                batch_codes,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    return torch.tensor(np.vstack(embeddings), dtype=torch.float32, device=device)


def create_pairwise_data(
    scores: np.ndarray,
    embeddings: torch.Tensor,
    min_score_diff: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create pairwise ranking data from evaluated programs.

    Args:
        scores: Array of program scores
        embeddings: Tensor of program embeddings (z-space) [n, dim]
        min_score_diff: Minimum score difference to exclude ties

    Returns:
        z_better: Z-space vectors of better programs
        z_worse: Z-space vectors of worse programs
        score_diffs: Score differences (always positive)
    """
    n = len(scores)
    device = embeddings.device

    # Create pairs
    better_indices = []
    worse_indices = []
    diffs = []

    for i, j in combinations(range(n), 2):
        diff = scores[i] - scores[j]

        if abs(diff) <= min_score_diff:
            continue

        if diff > 0:
            better_indices.append(i)
            worse_indices.append(j)
            diffs.append(diff)
        else:
            better_indices.append(j)
            worse_indices.append(i)
            diffs.append(-diff)

    z_better = embeddings[better_indices]
    z_worse = embeddings[worse_indices]
    score_diffs = torch.tensor(diffs, dtype=torch.float32, device=device)

    return z_better, z_worse, score_diffs


def create_dataset_from_task(
    task_name: str,
    min_score_diff: float = 0.0,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    encoder_model=None
) -> Tuple[PairwiseDataset, pd.DataFrame]:
    """
    Create pairwise dataset from a task's heuristics.

    Args:
        task_name: Name of the task
        min_score_diff: Minimum score difference for pairs
        device: Device to use
        cache_dir: Optional directory to cache evaluated results

    Returns:
        dataset: PairwiseDataset for training
        df: DataFrame with evaluated programs and embeddings
    """
    cache_path = Path(cache_dir) / f"{task_name}_evaluated.parquet" if cache_dir else None

    # Try to load from cache
    if cache_path and cache_path.exists():
        print(f"Loading cached evaluation from {cache_path}")
        df = pd.read_parquet(cache_path)
        # Re-encode if embeddings not in cache
        if 'z' not in df.columns:
            codes = df['code'].tolist()
            embeddings = encode_programs(codes, encoder_model=encoder_model, device=device)
            df['z'] = list(embeddings.cpu().numpy())
    else:
        # Load and evaluate
        programs = load_heuristics(task_name)
        print(f"Loaded {len(programs)} programs from {task_name}")

        df = evaluate_programs(task_name, programs)
        print(f"Successfully evaluated {len(df)} programs")

        if len(df) < 2:
            raise ValueError(f"Need at least 2 programs, got {len(df)}")

        # Encode programs
        codes = df['code'].tolist()
        embeddings = encode_programs(codes, encoder_model=encoder_model, device=device)
        df['z'] = list(embeddings.cpu().numpy())

        # Cache results
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            print(f"Cached to {cache_path}")

    # Get embeddings tensor from dataframe
    z_array = np.stack(df['z'].values)
    embeddings = torch.tensor(z_array, dtype=torch.float32, device=device)

    # Create pairwise data
    scores = df['score'].values
    z_better, z_worse, score_diffs = create_pairwise_data(scores, embeddings, min_score_diff)

    print(f"Created {len(z_better)} pairs from {len(df)} programs")
    print(f"  Amplification: {len(z_better) / len(df):.1f}x")

    dataset = PairwiseDataset(z_better, z_worse, score_diffs)

    return dataset, df


class RankingLoss(nn.Module):
    """
    Soft ranking loss using sigmoid + BCE.
    Provides smooth gradients for optimization.

    P(i > j) = sigmoid(tau * (R(z_i) - R(z_j)))
    loss = -log(P(i > j))
    """

    def __init__(self, tau: float = 1.0, margin: float = 0.0):
        super().__init__()
        self.tau = tau
        self.margin = margin

    def forward(self, score_better: torch.Tensor, score_worse: torch.Tensor) -> torch.Tensor:
        diff = score_better - score_worse - self.margin
        prob = torch.sigmoid(self.tau * diff)
        loss = -torch.log(prob + 1e-8).mean()
        return loss


class MarginRankingLossWrapper(nn.Module):
    """Wrapper around PyTorch's MarginRankingLoss."""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, score_better: torch.Tensor, score_worse: torch.Tensor) -> torch.Tensor:
        target = torch.ones_like(score_better)
        return self.loss_fn(score_better, score_worse, target)


def train_ranking_predictor(
    predictor: RankingScorePredictor,
    dataset: PairwiseDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_type: str = 'soft',
    tau: float = 1.0,
    margin: float = 0.1,
    val_split: float = 0.1,
    device: str = 'cuda',
    verbose: bool = True,
    patience: int = 20
) -> Tuple[RankingScorePredictor, Dict]:
    """
    Train ranking score predictor on pairwise data.

    Args:
        predictor: RankingScorePredictor model
        dataset: PairwiseDataset
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization
        loss_type: 'soft' (sigmoid+BCE) or 'margin' (hinge)
        tau: Temperature for soft ranking loss
        margin: Margin for ranking losses
        val_split: Fraction for validation
        device: Device to train on
        verbose: Print progress
        patience: Early stopping patience

    Returns:
        Trained predictor and training history
    """
    # Split into train/val
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training on {n_train} pairs, validating on {n_val} pairs")

    # Loss function
    if loss_type == 'soft':
        criterion = RankingLoss(tau=tau, margin=margin)
    elif loss_type == 'margin':
        criterion = MarginRankingLossWrapper(margin=margin)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Move to device
    predictor = predictor.to(device)
    predictor.train()

    # Optimizer
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        predictor.train()
        train_loss = torch.tensor(0.0, device=device)
        n_batches = 0

        for z_better, z_worse, _ in train_loader:
            z_better = z_better.to(device)
            z_worse = z_worse.to(device)

            optimizer.zero_grad()

            score_better = predictor(z_better).squeeze()
            score_worse = predictor(z_worse).squeeze()

            loss = criterion(score_better, score_worse)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate as tensor to avoid CPU-GPU sync
            train_loss += loss.detach()
            n_batches += 1

        avg_train_loss = train_loss.item() / max(n_batches, 1)

        # Validation
        predictor.eval()
        val_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0

        with torch.no_grad():
            for z_better, z_worse, _ in val_loader:
                z_better = z_better.to(device)
                z_worse = z_worse.to(device)

                score_better = predictor(z_better).squeeze()
                score_worse = predictor(z_worse).squeeze()

                loss = criterion(score_better, score_worse)
                val_loss += loss

                # Pairwise accuracy
                correct += (score_better > score_worse).sum()
                total += len(score_better)

        avg_val_loss = val_loss.item() / max(len(val_loader), 1)
        val_accuracy = correct.item() / max(total, 1)

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = predictor.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"Acc: {val_accuracy:.2%} | "
                  f"LR: {current_lr:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        predictor.load_state_dict(best_state)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.2%}")

    return predictor, history


def save_ranking_predictor(
    predictor: RankingScorePredictor,
    path: str,
    history: Optional[Dict] = None,
    extra_info: Optional[Dict] = None
):
    """Save ranking predictor to disk."""
    checkpoint = {
        'model_state_dict': predictor.state_dict(),
        'input_dim': predictor.input_dim,
        'hidden_dim': predictor.hidden_dim,
        'num_layers': predictor.num_layers,
        'space': 'z',  # Indicates this model operates on z-space
        'encoder': 'BAAI/bge-code-v1',  # Encoder used for embeddings
    }

    if history is not None:
        checkpoint['history'] = history

    if extra_info is not None:
        checkpoint.update(extra_info)

    torch.save(checkpoint, path)
    print(f"Saved ranking predictor to {path}")


def load_ranking_predictor(path: str, device: str = 'cuda') -> Tuple[RankingScorePredictor, Dict]:
    """Load ranking predictor from disk."""
    checkpoint = torch.load(path, map_location=device)

    predictor = RankingScorePredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint.get('num_layers', 2)
    )
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.to(device)

    print(f"Loaded ranking predictor from {path}")
    print(f"  Space: {checkpoint.get('space', 'z')}")
    print(f"  Encoder: {checkpoint.get('encoder', 'BAAI/bge-code-v1')}")
    print(f"  Input dim: {checkpoint['input_dim']}")

    return predictor, checkpoint


# ============================================================================
# Gradient search in z-space (for use with this predictor)
# ============================================================================

def gradient_ascent_in_z(
    predictor: RankingScorePredictor,
    num_starts: int = 10,
    steps: int = 100,
    lr: float = 0.01,
    init_z: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Perform gradient ascent in z-space to find high-scoring regions.

    Args:
        predictor: Trained ranking score predictor R(z)
        num_starts: Number of random starting points
        steps: Number of gradient ascent steps
        lr: Learning rate for gradient ascent
        init_z: Optional initial z values [num_starts, dim]
        device: Device to use
        verbose: Print progress

    Returns:
        best_z: Optimized z-space vectors [num_starts, dim]
    """
    predictor.eval()

    # Initialize starting points
    if init_z is None:
        # Sample from standard Gaussian (approximate z distribution)
        z = torch.randn(num_starts, predictor.input_dim, device=device, requires_grad=True)
    else:
        z = init_z.clone().to(device).requires_grad_(True)
        num_starts = z.shape[0]

    if verbose:
        print(f"\nGradient ascent with {num_starts} starting points for {steps} steps...")

    # Gradient ascent
    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()

        # Predict score
        pred_scores = predictor(z)

        # Maximize score (gradient ascent)
        loss = -pred_scores.mean()
        loss.backward()

        # Update z
        with torch.no_grad():
            z += lr * z.grad

        if verbose and (step + 1) % 20 == 0:
            avg_score = -loss.item()
            print(f"  Step {step+1}/{steps} | Avg predicted score: {avg_score:.4f}")

    # Final scores
    with torch.no_grad():
        final_scores = predictor(z)

    if verbose:
        print(f"\nFinal predicted scores:")
        print(f"  Min: {final_scores.min().item():.4f}")
        print(f"  Max: {final_scores.max().item():.4f}")
        print(f"  Mean: {final_scores.mean().item():.4f}")

    return z.detach()


def adaptive_gradient_search_z(
    predictor: RankingScorePredictor,
    df: pd.DataFrame,
    num_searches: int = 5,
    steps_per_search: int = 100,
    lr: float = 0.01,
    init_from_top_k: int = 5,
    device: str = 'cuda',
    verbose: bool = True
) -> torch.Tensor:
    """
    Adaptive gradient search: start from top-k programs and optimize in z-space.

    Args:
        predictor: Trained ranking score predictor
        df: DataFrame with 'z' and 'score' columns
        num_searches: Number of optimization runs
        steps_per_search: Gradient steps per run
        lr: Learning rate
        init_from_top_k: Initialize from top-k programs
        device: Device to use
        verbose: Print progress

    Returns:
        optimized_z: Optimized z-space vectors
    """
    # Get top-k programs
    top_k = df.nlargest(init_from_top_k, 'score')

    if len(top_k) == 0:
        raise ValueError("No valid programs in dataframe")

    # Get their z embeddings
    z_top = torch.tensor(
        np.stack(top_k['z'].values),
        dtype=torch.float32,
        device=device
    )

    if verbose:
        print(f"\nAdaptive gradient search starting from top {len(top_k)} programs")
        print(f"Top scores: {top_k['score'].values}")

    # Sample starting points from top programs
    init_indices = np.random.choice(len(z_top), size=num_searches, replace=True)
    z_init = z_top[init_indices]

    # Add small noise for exploration
    z_init = z_init + torch.randn_like(z_init) * 0.1

    # Perform gradient ascent
    optimized_z = gradient_ascent_in_z(
        predictor,
        num_starts=num_searches,
        steps=steps_per_search,
        lr=lr,
        init_z=z_init,
        device=device,
        verbose=verbose
    )

    return optimized_z


# ============================================================================
# Main training script
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train ranking score predictor (z-space)')
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--output', type=str, default='ranking_predictor_z.pth', help='Output path')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--loss_type', type=str, default='soft', choices=['soft', 'margin'])
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature for soft loss')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for ranking loss')
    parser.add_argument('--min_score_diff', type=float, default=0.0, help='Min score diff for pairs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')

    args = parser.parse_args()

    print("="*70)
    print("Ranking Score Predictor Training (Z-Space)")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Loss type: {args.loss_type}")
    print(f"Device: {args.device}")
    print()

    # Load encoder model (BAAI/bge-code-v1)
    print("Loading encoder model (BAAI/bge-code-v1)...")
    encoder_model = get_encoder_model(args.device)
    print("Encoder loaded.\n")

    # Create dataset (no flow model needed!)
    dataset, df = create_dataset_from_task(
        task_name=args.task,
        min_score_diff=args.min_score_diff,
        device=args.device,
        cache_dir=args.cache_dir,
        encoder_model=encoder_model
    )

    # Free encoder memory after encoding
    del encoder_model
    torch.cuda.empty_cache()

    print(f"\nDataset size: {len(dataset)} pairs")

    # Create predictor - get input_dim from actual embeddings
    input_dim = df['z'].iloc[0].shape[0]  # Get dim from encoded embeddings
    print(f"Embedding dimension: {input_dim}")
    predictor = RankingScorePredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    print(f"Created predictor: {sum(p.numel() for p in predictor.parameters()):,} parameters")

    # Train
    predictor, history = train_ranking_predictor(
        predictor=predictor,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss_type,
        tau=args.tau,
        margin=args.margin,
        device=args.device,
        verbose=True
    )

    # Save
    save_ranking_predictor(
        predictor=predictor,
        path=args.output,
        history=history,
        extra_info={
            'task': args.task,
            'loss_type': args.loss_type,
            'tau': args.tau,
            'margin': args.margin,
            'n_pairs': len(dataset),
            'n_programs': len(df)
        }
    )

    print(f"\nDone! Model saved to {args.output}")


if __name__ == '__main__':
    main()
