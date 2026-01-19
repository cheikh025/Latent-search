"""
Score predictor for gradient-based optimization in prior space.

Train a regressor R(u) that predicts score from prior-space vectors,
then use gradient ascent to find high-scoring regions.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


class ScorePredictor(nn.Module):
    """
    Simple 2-layer MLP that predicts score from prior-space vector u.

    R: R^d -> R
    score = R(u)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Args:
            input_dim: Dimension of prior space (same as latent dim)
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super().__init__()

        # Simple 2-layer architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Predict score from prior-space vector.

        Args:
            u: Prior-space vectors [batch_size, dim]

        Returns:
            scores: Predicted scores [batch_size, 1]
        """
        return self.network(u)


def train_score_predictor(
    predictor: ScorePredictor,
    program_db,
    flow_model,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    verbose: bool = True
) -> ScorePredictor:
    """
    Train score predictor on current program database.

    Args:
        predictor: ScorePredictor model
        program_db: ProgramDatabase with (z, score) pairs
        flow_model: Trained normalizing flow to map z -> u
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        verbose: Print training progress

    Returns:
        Trained predictor
    """

    # Extract data from database
    if len(program_db) == 0:
        raise ValueError("ProgramDatabase is empty")

    # Filter out invalid scores
    valid_df = program_db.df[program_db.df['score'].notna() & np.isfinite(program_db.df['score'])]

    if len(valid_df) == 0:
        raise ValueError("No valid scores in database")

    print(f"Training on {len(valid_df)} programs with valid scores")

    # Get z embeddings and scores
    z_list = valid_df['z'].tolist()
    z_array = np.stack(z_list).astype(np.float32)
    z_tensor = torch.tensor(z_array, dtype=torch.float32, device=device)

    scores = valid_df['score'].values.astype(np.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device).unsqueeze(1)

    # Map z -> u using flow
    print("Mapping z to prior space u...")
    flow_model.eval()
    with torch.no_grad():
        u_tensor, _ = flow_model(z_tensor)

    print(f"Prior space u: mean={u_tensor.mean().item():.4f}, std={u_tensor.std().item():.4f}")
    print(f"Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

    # Normalize scores for better training (optional but recommended)
    score_mean = scores_tensor.mean()
    score_std = scores_tensor.std() + 1e-8
    scores_normalized = (scores_tensor - score_mean) / score_std

    # Create dataloader
    dataset = TensorDataset(u_tensor, scores_normalized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move predictor to device and set to training mode
    predictor = predictor.to(device)
    predictor.train()

    # Optimizer
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
    )

    # Loss function (MSE)
    criterion = nn.MSELoss()

    # Training loop
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for u_batch, score_batch in dataloader:
            optimizer.zero_grad()

            # Predict
            pred_scores = predictor(u_batch)

            # Compute loss
            loss = criterion(pred_scores, score_batch)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)

            # Update
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.6f}")

    # Store normalization parameters
    predictor.score_mean = score_mean
    predictor.score_std = score_std

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

    return predictor


def gradient_ascent_in_prior(
    predictor: ScorePredictor,
    flow_model,
    num_starts: int = 10,
    steps: int = 100,
    lr: float = 0.01,
    init_method: str = 'random',
    init_u: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform gradient ascent in prior space to find high-scoring regions.

    Args:
        predictor: Trained score predictor R(u)
        flow_model: Normalizing flow for mapping u -> z
        num_starts: Number of random starting points
        steps: Number of gradient ascent steps
        lr: Learning rate for gradient ascent
        init_method: 'random' (sample from N(0,I)) or 'provided' (use init_u)
        init_u: Optional initial u values [num_starts, dim]
        device: Device to use
        verbose: Print progress

    Returns:
        best_u: Optimized prior-space vectors [num_starts, dim]
        best_z: Corresponding latent vectors [num_starts, dim]
    """

    predictor.eval()
    flow_model.eval()

    # Initialize starting points
    if init_method == 'random':
        # Sample from standard Gaussian N(0, I)
        u = torch.randn(num_starts, flow_model.dim, device=device, requires_grad=True)
    elif init_method == 'provided':
        if init_u is None:
            raise ValueError("init_u must be provided when init_method='provided'")
        u = init_u.clone().to(device).requires_grad_(True)
        num_starts = u.shape[0]
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    if verbose:
        print(f"\nGradient ascent with {num_starts} starting points for {steps} steps...")

    # Gradient ascent
    for step in range(steps):
        if u.grad is not None:
            u.grad.zero_()

        # Predict score
        pred_scores = predictor(u)

        # Denormalize
        if hasattr(predictor, 'score_mean') and hasattr(predictor, 'score_std'):
            pred_scores = pred_scores * predictor.score_std + predictor.score_mean

        # Maximize score (gradient ascent)
        loss = -pred_scores.mean()  # Negative because we want to maximize
        loss.backward()

        # Update u
        with torch.no_grad():
            u += lr * u.grad

        if verbose and (step + 1) % 20 == 0:
            avg_score = -loss.item()
            print(f"  Step {step+1}/{steps} | Avg predicted score: {avg_score:.4f}")

    # Final scores
    with torch.no_grad():
        final_scores = predictor(u)
        if hasattr(predictor, 'score_mean') and hasattr(predictor, 'score_std'):
            final_scores = final_scores * predictor.score_std + predictor.score_mean

        # Map back to latent space
        best_z = flow_model.inverse(u)

    if verbose:
        print(f"\nFinal predicted scores:")
        print(f"  Min: {final_scores.min().item():.4f}")
        print(f"  Max: {final_scores.max().item():.4f}")
        print(f"  Mean: {final_scores.mean().item():.4f}")

    return u.detach(), best_z.detach()


def adaptive_gradient_search(
    predictor: ScorePredictor,
    flow_model,
    program_db,
    num_searches: int = 5,
    steps_per_search: int = 100,
    lr: float = 0.01,
    init_from_top_k: int = 5,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive gradient search: start from top-k programs and optimize.

    Args:
        predictor: Trained score predictor
        flow_model: Normalizing flow
        program_db: ProgramDatabase
        num_searches: Number of optimization runs
        steps_per_search: Gradient steps per run
        lr: Learning rate
        init_from_top_k: Initialize from top-k programs in database
        device: Device to use
        verbose: Print progress

    Returns:
        optimized_u: Optimized prior-space vectors
        optimized_z: Corresponding latent vectors
    """

    # Get top-k programs
    top_k = program_db.get_top_n(init_from_top_k)

    if len(top_k) == 0:
        raise ValueError("No valid programs in database")

    # Get their z embeddings
    z_top = torch.tensor(
        np.stack(top_k['z'].values),
        dtype=torch.float32,
        device=device
    )

    # Map to prior space
    flow_model.eval()
    with torch.no_grad():
        u_top, _ = flow_model(z_top)

    if verbose:
        print(f"\nAdaptive gradient search starting from top {len(top_k)} programs")
        print(f"Top scores: {top_k['score'].values}")

    # Sample starting points from top programs
    init_indices = np.random.choice(len(u_top), size=num_searches, replace=True)
    u_init = u_top[init_indices]

    # Add small noise for exploration
    u_init = u_init + torch.randn_like(u_init) * 0.1

    # Perform gradient ascent
    optimized_u, optimized_z = gradient_ascent_in_prior(
        predictor,
        flow_model,
        num_starts=num_searches,
        steps=steps_per_search,
        lr=lr,
        init_method='provided',
        init_u=u_init,
        device=device,
        verbose=verbose
    )

    return optimized_u, optimized_z


if __name__ == '__main__':
    # Test with dummy data
    print("Testing Score Predictor")
    print("=" * 70)

    dim = 768
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create predictor
    predictor = ScorePredictor(input_dim=dim, hidden_dim=256)
    print(f"Created predictor with {sum(p.numel() for p in predictor.parameters()):,} parameters")

    # Test forward pass
    u = torch.randn(10, dim)
    scores = predictor(u)
    print(f"Input shape: {u.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Predicted scores: {scores.squeeze()}")
