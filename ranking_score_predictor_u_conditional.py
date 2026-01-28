"""
Ranking-based score predictor for gradient-based optimization in prior space (u-space)
using a TASK-CONDITIONAL normalizing flow.

Uses pairwise ranking loss instead of MSE regression to handle small datasets.
From n programs, creates n*(n-1)/2 pairs for training.

This version is designed to work with gradient_search_u_conditional.py which uses
a task-conditional flow. The u-space vectors will be consistent between training
and search.

Encoder: BAAI/bge-code-v1 (same as unified mapper training)
Pipeline: code -> z (encoder) -> u (conditional flow) -> score (predictor)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm


class PairwiseDataset(Dataset):
    """Dataset of (u_better, u_worse, score_diff) pairs for ranking."""

    def __init__(self, u_better: torch.Tensor, u_worse: torch.Tensor, score_diffs: torch.Tensor):
        self.u_better = u_better
        self.u_worse = u_worse
        self.score_diffs = score_diffs

    def __len__(self):
        return len(self.u_better)

    def __getitem__(self, idx):
        return self.u_better[idx], self.u_worse[idx], self.score_diffs[idx]


class RankingScorePredictor(nn.Module):
    """
    MLP that predicts score from prior-space vector u.
    Trained with ranking loss for better generalization on small datasets.

    Uses BAAI/bge-code-v1 for z embeddings, then conditional normalizing flow for z -> u.

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

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Predict score from prior-space vector."""
        return self.network(u)


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


def evaluate_programs(task_name: str, programs: Dict[str, str], use_secure: bool = True, num_workers: int = 4) -> pd.DataFrame:
    """
    Evaluate all programs for a task and return DataFrame with scores.

    Args:
        task_name: Name of the task
        programs: Dictionary of {name: code} pairs
        use_secure: Whether to use SecureEvaluator wrapper
        num_workers: Number of parallel workers for evaluation

    Returns:
        DataFrame with columns ['name', 'code', 'score']
    """
    evaluator = get_evaluator(task_name)

    if use_secure:
        from base.evaluate import SecureEvaluator
        evaluator = SecureEvaluator(evaluator, debug_mode=False)

    def eval_single(name_code):
        """Evaluate a single program."""
        name, code = name_code
        try:
            score = evaluator.evaluate_program(code)
            if score is not None and np.isfinite(score):
                return {'name': name, 'code': code, 'score': score}
        except Exception:
            pass
        return None

    results = []

    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_single, (name, code)): name
                   for name, code in programs.items()}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Evaluating {task_name} ({num_workers} workers)"):
            result = future.result()
            if result is not None:
                results.append(result)

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


def load_conditional_flow(flow_path: str, device: str = "cuda"):
    """
    Load trained conditional normalizing flow (and task-conditional prior if present).

    Same loading logic as gradient_search_u_conditional.py for consistency.
    """
    from normalizing_flow import NormalizingFlow
    from train_conditional_flow import TaskConditionalPrior

    print(f"Loading conditional flow from {flow_path}...")

    checkpoint = torch.load(flow_path, map_location=device)

    # Extract architecture parameters (conditional checkpoints use flow_* keys)
    dim = checkpoint.get('flow_dim', checkpoint.get('dim', checkpoint.get('embedding_dim', 768)))
    num_layers = checkpoint.get('flow_num_layers', checkpoint.get('num_layers', 4))
    hidden_dim = checkpoint.get('flow_hidden_dim', checkpoint.get('hidden_dim', 128))
    dropout = checkpoint.get('flow_dropout', checkpoint.get('dropout', 0.0))

    print(f"  Dimension: {dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout} (disabled in eval mode)")

    # Create flow model with dropout (disabled in eval mode)
    flow_model = NormalizingFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)

    flow_state = checkpoint.get('flow_state_dict', checkpoint.get('model_state_dict'))
    if flow_state is None:
        raise KeyError("Conditional flow checkpoint missing flow_state_dict/model_state_dict")

    flow_model.load_state_dict(flow_state)
    flow_model.to(device)
    flow_model.eval()

    prior_model = None
    if 'prior_state_dict' in checkpoint:
        num_tasks = checkpoint.get('prior_num_tasks') or checkpoint.get('num_tasks')
        if num_tasks is None:
            task_map = checkpoint.get('task_to_id', {})
            num_tasks = len(task_map) if task_map else None
        prior_dim = checkpoint.get('prior_dim', dim)

        if num_tasks is not None:
            prior_model = TaskConditionalPrior(num_tasks=num_tasks, dim=prior_dim)
            prior_model.load_state_dict(checkpoint['prior_state_dict'])
            prior_model.to(device)
            prior_model.eval()

    task_to_id = checkpoint.get('task_to_id')
    id_to_task = checkpoint.get('id_to_task')

    print(f"  Flow loaded successfully")
    if prior_model is not None:
        print(f"  Task-conditional prior loaded")

    return flow_model, prior_model, task_to_id, id_to_task, dim


def create_pairwise_data(
    scores: np.ndarray,
    embeddings: torch.Tensor,
    flow_model,
    min_score_diff: float = 0.0,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create pairwise ranking data from evaluated programs.

    Args:
        scores: Array of program scores
        embeddings: Tensor of program embeddings (z-space) [n, dim]
        flow_model: Normalizing flow to map z -> u
        min_score_diff: Minimum score difference to exclude ties
        device: Device to use

    Returns:
        u_better: Prior-space vectors of better programs
        u_worse: Prior-space vectors of worse programs
        score_diffs: Score differences (always positive)
    """
    n = len(scores)

    # Map z -> u using flow
    flow_model.eval()
    with torch.no_grad():
        u_all, _ = flow_model(embeddings.to(device))

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

    u_better = u_all[better_indices]
    u_worse = u_all[worse_indices]
    score_diffs = torch.tensor(diffs, dtype=torch.float32, device=device)

    return u_better, u_worse, score_diffs


def create_dataset_from_task(
    task_name: str,
    flow_model,
    min_score_diff: float = 0.0,
    device: str = 'cuda',
    encoder_model=None,
    num_workers: int = 4
) -> Tuple[PairwiseDataset, pd.DataFrame]:
    """
    Create pairwise dataset from a task's heuristics.

    Args:
        task_name: Name of the task
        flow_model: Trained conditional normalizing flow
        min_score_diff: Minimum score difference for pairs
        device: Device to use
        encoder_model: Optional pre-loaded encoder model
        num_workers: Number of parallel workers for evaluation

    Returns:
        dataset: PairwiseDataset for training
        df: DataFrame with evaluated programs
    """
    # Load and evaluate
    programs = load_heuristics(task_name)
    print(f"Loaded {len(programs)} programs from {task_name}")

    df = evaluate_programs(task_name, programs, num_workers=num_workers)
    print(f"Successfully evaluated {len(df)} programs")

    if len(df) < 2:
        raise ValueError(f"Need at least 2 programs, got {len(df)}")

    # Encode programs
    codes = df['code'].tolist()
    embeddings = encode_programs(codes, encoder_model=encoder_model, device=device)

    # Store embeddings in dataframe
    df['z'] = list(embeddings.cpu().numpy())

    # Create pairwise data
    scores = df['score'].values
    u_better, u_worse, score_diffs = create_pairwise_data(
        scores, embeddings, flow_model, min_score_diff, device
    )

    print(f"Created {len(u_better)} pairs from {len(df)} programs")
    print(f"  Amplification: {len(u_better) / len(df):.1f}x")

    dataset = PairwiseDataset(u_better, u_worse, score_diffs)

    return dataset, df


class RankingLoss(nn.Module):
    """
    Soft ranking loss using sigmoid + BCE.
    Provides smooth gradients for optimization.

    P(i > j) = sigmoid(tau * (R(u_i) - R(u_j)))
    loss = -log(P(i > j))

    Uses numerically stable logsigmoid implementation.
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, score_better: torch.Tensor, score_worse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_better: Predicted scores for better programs [batch]
            score_worse: Predicted scores for worse programs [batch]

        Returns:
            loss: Scalar loss value
        """
        diff = score_better - score_worse
        # Use numerically stable logsigmoid instead of log(sigmoid())
        loss = -F.logsigmoid(self.tau * diff).mean()
        return loss


def train_ranking_predictor(
    predictor: RankingScorePredictor,
    dataset: PairwiseDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    tau: float = 1.0,
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
        tau: Temperature for soft ranking loss (higher = sharper gradients)
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

    # Loss function (soft ranking loss with numerically stable logsigmoid)
    criterion = RankingLoss(tau=tau)

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
        train_loss = 0.0
        n_batches = 0

        for u_better, u_worse, _ in train_loader:
            u_better = u_better.to(device)
            u_worse = u_worse.to(device)

            optimizer.zero_grad()

            score_better = predictor(u_better).squeeze()
            score_worse = predictor(u_worse).squeeze()

            loss = criterion(score_better, score_worse)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        predictor.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for u_better, u_worse, _ in val_loader:
                u_better = u_better.to(device)
                u_worse = u_worse.to(device)

                score_better = predictor(u_better).squeeze()
                score_worse = predictor(u_worse).squeeze()

                loss = criterion(score_better, score_worse)
                val_loss += loss.item()

                # Pairwise accuracy
                correct += (score_better > score_worse).sum().item()
                total += len(score_better)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_accuracy = correct / max(total, 1)

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
        'space': 'u_conditional',  # Indicates this model operates on conditional u-space
        'encoder': 'BAAI/bge-code-v1',  # Encoder used for z embeddings
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
    print(f"  Space: {checkpoint.get('space', 'u_conditional')}")
    print(f"  Encoder: {checkpoint.get('encoder', 'BAAI/bge-code-v1')}")
    print(f"  Input dim: {checkpoint['input_dim']}")

    return predictor, checkpoint


# ============================================================================
# Main training script
# ============================================================================

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train ranking score predictor (U-Space with Conditional Flow)')
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--flow_path', type=str, default='Flow_Checkpoints/conditional_flow_final.pth', help='Path to trained conditional flow model')
    parser.add_argument('--output_dir', type=str, default='Predictor_Checkpoints', help='Output directory for saved models')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: ranking_predictor_u_conditional_{task}.pth)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature for soft ranking loss (higher = sharper gradients)')
    parser.add_argument('--min_score_diff', type=float, default=0.0, help='Min score diff for pairs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for evaluation')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set default output filename if not provided
    if args.output is None:
        args.output = f"ranking_predictor_u_conditional_{args.task}.pth"

    # Full output path
    output_path = os.path.join(args.output_dir, args.output)

    print("="*70)
    print("Ranking Score Predictor Training (U-Space with Conditional Flow)")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Flow model: {args.flow_path} (conditional)")
    print(f"Output: {output_path}")
    print(f"Loss: Soft ranking (tau={args.tau})")
    print(f"Device: {args.device}")
    print(f"Parallel workers: {args.num_workers}")
    print()

    # Load encoder model (BAAI/bge-code-v1)
    print("Loading encoder model (BAAI/bge-code-v1)...")
    encoder_model = get_encoder_model(args.device)
    print("Encoder loaded.\n")

    # Load conditional flow model
    flow_model, prior_model, task_to_id, id_to_task, flow_dim = load_conditional_flow(
        args.flow_path, args.device
    )

    # Log task mapping if available
    if task_to_id is not None and args.task in task_to_id:
        print(f"  Task '{args.task}' has ID: {task_to_id[args.task]}")
    print()

    # Create dataset
    dataset, df = create_dataset_from_task(
        task_name=args.task,
        flow_model=flow_model,
        min_score_diff=args.min_score_diff,
        device=args.device,
        encoder_model=encoder_model,
        num_workers=args.num_workers
    )

    # Free encoder memory after encoding
    del encoder_model
    torch.cuda.empty_cache()

    print(f"\nDataset size: {len(dataset)} pairs")

    # Create predictor - input_dim comes from flow (u-space has same dim as z-space)
    input_dim = flow_dim
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
        tau=args.tau,
        device=args.device,
        verbose=True
    )

    # Save with extra info about the conditional flow
    extra_info = {
        'task': args.task,
        'loss_type': 'soft',
        'tau': args.tau,
        'n_pairs': len(dataset),
        'n_programs': len(df),
        'flow_path': args.flow_path,
        'flow_type': 'conditional',
    }

    # Include task mapping if available
    if task_to_id is not None:
        extra_info['task_to_id'] = task_to_id
    if id_to_task is not None:
        extra_info['id_to_task'] = id_to_task

    save_ranking_predictor(
        predictor=predictor,
        path=output_path,
        history=history,
        extra_info=extra_info
    )

    print(f"\nDone! Model saved to {output_path}")


if __name__ == '__main__':
    main()
