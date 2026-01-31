"""
Visualize both z-space (original latent) and u-space (normalized prior) embeddings using t-SNE.

Creates two side-by-side scatter plots:
- Left: t-SNE of z embeddings (CodeBERT/encoder embeddings)
- Right: t-SNE of u embeddings (after normalizing flow transformation)

Both plots use viridis colormap and are colored by program scores.

Usage:
    # Use default flow checkpoint
    python visualize_z_and_u_space.py --task tsp_construct

    # Specify custom flow checkpoint
    python visualize_z_and_u_space.py --task tsp_construct --flow my_flow.pth --perplexity 30 --show_top_k 10
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm

from normalizing_flow import NormalizingFlow
from model_config import DEFAULT_ENCODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder


def load_heuristics(task_name: str) -> dict:
    """Load heuristics from JSON file."""
    augmented_path = Path(f"task/{task_name}/augmented.json")
    heuristics_path = Path(f"task/{task_name}/heuristics.json")

    if augmented_path.exists():
        path = augmented_path
    elif heuristics_path.exists():
        path = heuristics_path
    else:
        raise FileNotFoundError(f"Heuristics not found in {augmented_path} or {heuristics_path}")

    with open(path, "r") as f:
        return json.load(f)


def encode_programs(
    codes: list,
    encoder_model,
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    """Encode programs to z-space embeddings."""
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Encoding programs"):
            batch_codes = codes[i:i + batch_size]
            batch_embeddings = encoder_model.encode(
                batch_codes,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


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


def get_scores(task_name: str, codes: list, num_workers: int = 4) -> np.ndarray:
    """Evaluate programs in parallel to get scores for coloring."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from base.evaluate import SecureEvaluator

    evaluator = get_evaluator(task_name)
    secure_evaluator = SecureEvaluator(evaluator, debug_mode=False)

    scores = [None] * len(codes)

    def eval_single(idx_code):
        """Evaluate a single program."""
        idx, code = idx_code
        try:
            score = secure_evaluator.evaluate_program(code)
            if score is not None and np.isfinite(score):
                return idx, score
        except Exception:
            pass
        return idx, None

    # Parallel evaluation with progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_single, (i, code)): i
                   for i, code in enumerate(codes)}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Evaluating {task_name} ({num_workers} workers)"):
            idx, score = future.result()
            scores[idx] = score

    # Replace None with median for visualization
    valid_scores = [s for s in scores if s is not None]
    if valid_scores:
        median = np.median(valid_scores)
        print(f"Successfully evaluated {len(valid_scores)}/{len(codes)} programs")
        print(f"Score range: [{min(valid_scores):.4f}, {max(valid_scores):.4f}]")
    else:
        median = 0.0
        print(f"Warning: No programs evaluated successfully!")

    scores = [s if s is not None else median for s in scores]

    return np.array(scores, dtype=float)


def load_flow(flow_path: str, device: str = "cuda") -> NormalizingFlow:
    """Load trained normalizing flow model."""
    print(f"Loading normalizing flow from {flow_path}...")

    checkpoint = torch.load(flow_path, map_location=device, weights_only=False)

    # Extract architecture parameters
    dim = checkpoint.get('dim', checkpoint.get('embedding_dim', 768))
    num_layers = checkpoint.get('num_layers', 4)  # Default to 4 for small datasets
    hidden_dim = checkpoint.get('hidden_dim', 128)  # Default to 128 for small datasets
    dropout = checkpoint.get('dropout', 0.0)  # Dropout will be disabled in eval mode

    print(f"  Dimension: {dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout} (disabled in eval mode)")

    # Create flow model with dropout (will be disabled in eval mode)
    flow_model = NormalizingFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    flow_model.to(device)
    flow_model.eval()  # This disables dropout

    print(f"  ✓ Flow loaded")
    return flow_model


def transform_z_to_u(z_embeddings: np.ndarray, flow_model: NormalizingFlow, device: str = "cuda") -> np.ndarray:
    """Transform z embeddings to u (prior space) using normalizing flow."""
    print("Transforming z → u using normalizing flow...")

    z_tensor = torch.from_numpy(z_embeddings).float().to(device)

    with torch.no_grad():
        u_tensor, _ = flow_model(z_tensor)

    u_embeddings = u_tensor.cpu().numpy()

    # Check if u is approximately N(0, I)
    u_mean = np.mean(u_embeddings, axis=0)
    u_std = np.std(u_embeddings, axis=0)
    print(f"  U-space statistics:")
    print(f"    Mean: {np.mean(u_mean):.6f} ± {np.std(u_mean):.6f} (should be ~0)")
    print(f"    Std:  {np.mean(u_std):.6f} ± {np.std(u_std):.6f} (should be ~1)")

    return u_embeddings


def plot_scatter(ax, x, y, scores, title, cmap='viridis', show_top_k=None, alpha=0.6, point_size=50):
    """Scatter plot showing individual heuristics colored by score."""

    # Plot all points
    scatter = ax.scatter(
        x, y,
        c=scores,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors='black',
        linewidths=0.3
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Score", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Highlight top-k performers
    if show_top_k is not None and show_top_k > 0:
        top_k_indices = np.argsort(scores)[-show_top_k:]
        ax.scatter(
            x[top_k_indices],
            y[top_k_indices],
            s=200,
            facecolors='none',
            edgecolors='red',
            linewidths=2.5,
            label=f'Top {show_top_k}',
            zorder=10
        )
        ax.legend(loc='upper right', fontsize=10)

    # Axis labels
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)


def visualize_z_and_u_space(
    task_name: str,
    flow_path: str,
    perplexity: int = 30,
    num_workers: int = 4,
    device: str = "cuda",
    save_path: str = None,
    encoder_name: str = None,
    truncate_dim: int = None,
    cmap: str = 'viridis',
    show_top_k: int = None,
    scatter_alpha: float = 0.6,
    point_size: int = 50,
):
    """
    Create side-by-side t-SNE visualizations of z-space and u-space.

    Args:
        task_name: Name of the task (e.g., "tsp_construct")
        flow_path: Path to trained normalizing flow checkpoint
        perplexity: t-SNE perplexity parameter
        num_workers: Number of parallel workers for evaluation
        device: Device to use for encoding and flow
        save_path: Path to save the plot (default: auto-generated)
        encoder_name: Encoder model name (default: from model_config.py)
        truncate_dim: Matryoshka embedding dimension (default: from model_config.py)
        cmap: Colormap name (default: 'viridis')
        show_top_k: Highlight top-k performers (e.g., 5 or 10)
        scatter_alpha: Transparency for scatter plots (0-1)
        point_size: Size of scatter points

    Returns:
        Tuple of (z_2d, u_2d, scores, names)
    """

    # Load heuristics
    heuristics = load_heuristics(task_name)
    names = list(heuristics.keys())
    codes = list(heuristics.values())
    print(f"Loaded {len(codes)} programs from {task_name}")

    if len(codes) < 3:
        raise ValueError("Need at least 3 programs to run t-SNE.")

    # Load encoder
    print(f"\nLoading encoder model: {encoder_name or DEFAULT_ENCODER}")
    if truncate_dim is not None:
        print(f"  Truncating to {truncate_dim} dimensions (Matryoshka)")
    elif DEFAULT_MATRYOSHKA_DIM is not None:
        print(f"  Using default Matryoshka dimension: {DEFAULT_MATRYOSHKA_DIM}")

    encoder_model, embedding_dim = load_encoder(
        model_name=encoder_name,
        device=device,
        truncate_dim=truncate_dim
    )
    print(f"  Actual embedding dimension: {embedding_dim}")

    # Encode programs to z-space
    print()
    z_embeddings = encode_programs(codes, encoder_model, device)

    # Free encoder memory
    del encoder_model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load normalizing flow
    print()
    flow_model = load_flow(flow_path, device)

    # Transform z → u
    print()
    u_embeddings = transform_z_to_u(z_embeddings, flow_model, device)

    # Free flow memory
    del flow_model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get scores for coloring
    print()
    scores = get_scores(task_name, codes, num_workers=num_workers)

    # Run t-SNE on z-space
    effective_perplexity = min(perplexity, len(codes) - 1)
    if effective_perplexity < 2:
        effective_perplexity = 2

    print(f"\nRunning t-SNE on z-space (perplexity={effective_perplexity})...")
    tsne_z = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        verbose=0
    )
    z_2d = tsne_z.fit_transform(z_embeddings)

    # Run t-SNE on u-space
    print(f"Running t-SNE on u-space (perplexity={effective_perplexity})...")
    tsne_u = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        verbose=0
    )
    u_2d = tsne_u.fit_transform(u_embeddings)

    # Create side-by-side plots
    print("\nCreating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Plot z-space
    plot_scatter(
        ax1,
        z_2d[:, 0], z_2d[:, 1],
        scores,
        title="Z-Space (Original Latent Embeddings)",
        cmap=cmap,
        show_top_k=show_top_k,
        alpha=scatter_alpha,
        point_size=point_size
    )

    # Plot u-space
    plot_scatter(
        ax2,
        u_2d[:, 0], u_2d[:, 1],
        scores,
        title="U-Space (Normalized Prior Space)",
        cmap=cmap,
        show_top_k=show_top_k,
        alpha=scatter_alpha,
        point_size=point_size
    )

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = f"{task_name}_z_and_u_space_comparison.png"

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
    print(f"\nSaved visualization to: {save_path}")

    plt.show()

    return z_2d, u_2d, scores, names


def main():
    parser = argparse.ArgumentParser(
        description="Visualize both z-space and u-space embeddings using t-SNE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Task and flow arguments
    parser.add_argument("--task", type=str, default="tsp_construct",
                        help="Task name (e.g., tsp_construct, cvrp_construct)")
    parser.add_argument("--flow", type=str, default="Flow_Checkpoints/unified_flow_final.pth",
                        help="Path to trained normalizing flow checkpoint")

    # Model configuration
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                        help=f"Encoder model name")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_MATRYOSHKA_DIM,
                        help=f"Matryoshka embedding dimension")

    # Visualization parameters
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for evaluation")

    # Plot style
    parser.add_argument("--show_top_k", type=int, default=None,
                        help="Highlight top-k performers with special markers (e.g., 5 or 10)")
    parser.add_argument("--scatter_alpha", type=float, default=0.6,
                        help="Point transparency for scatter plots (0-1)")
    parser.add_argument("--point_size", type=int, default=50,
                        help="Size of scatter points (default: 50)")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap name")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (default: auto-generated)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for encoder and flow")

    args = parser.parse_args()

    print("="*70)
    print("Z-Space vs U-Space Visualization")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Flow: {args.flow}")
    print(f"Encoder: {args.encoder}")
    if args.embedding_dim is not None:
        print(f"Embedding dimension: {args.embedding_dim} (Matryoshka)")
    print(f"Colormap: {args.cmap}")
    if args.show_top_k:
        print(f"Highlighting top-{args.show_top_k} performers")
    print("="*70)
    print()

    visualize_z_and_u_space(
        task_name=args.task,
        flow_path=args.flow,
        perplexity=args.perplexity,
        num_workers=args.num_workers,
        device=args.device,
        save_path=args.output,
        encoder_name=args.encoder,
        truncate_dim=args.embedding_dim,
        cmap=args.cmap,
        show_top_k=args.show_top_k,
        scatter_alpha=args.scatter_alpha,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
