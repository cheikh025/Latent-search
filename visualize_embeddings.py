"""
Visualize heuristic embeddings using t-SNE as a HEATMAP (no points).

Usage:
    python visualize_embeddings.py --task tsp_construct
    python visualize_embeddings.py --task tsp_construct --perplexity 30 --with_scores
    python visualize_embeddings.py --task tsp_construct --plot hexbin
    python visualize_embeddings.py --task tsp_construct --plot hist2d --with_scores
    python visualize_embeddings.py --task tsp_construct --embedding-dim 512
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm

from model_config import DEFAULT_ENCODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder


def load_heuristics(task_name: str) -> dict:
    """Load heuristics from JSON file."""
    # Try augmented.json first, fall back to heuristics.json
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
    """
    Encode programs to z-space embeddings using the project's standard encoder.

    Args:
        codes: List of code strings
        encoder_model: Pre-loaded encoder model from load_encoder()
        device: Device for encoding
        batch_size: Batch size for encoding

    Returns:
        Array of embeddings [n, embedding_dim]
    """
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
    """Get the evaluator class for a given task (matches ranking_score_predictor_z.py)."""
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
    """
    Evaluate programs in parallel to get scores for coloring.

    Uses the same parallel evaluation pattern as ranking_score_predictor_z.py

    Args:
        task_name: Name of the task
        codes: List of code strings
        num_workers: Number of parallel workers

    Returns:
        Array of scores (None values replaced with median)
    """
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


def plot_heatmap_hexbin(ax, x, y, scores=None, gridsize=70, cmap=None):
    """
    Hexbin heatmap with improved styling.
      - If scores is None: density heatmap (log counts).
      - If scores is not None: mean score per bin.
    """
    # Better default colormaps
    if cmap is None:
        cmap = "YlOrRd" if scores is None else "RdYlGn"

    if scores is None:
        hb = ax.hexbin(
            x, y,
            gridsize=gridsize,
            bins="log",   # log10(count)
            mincnt=1,     # hide empty bins
            cmap=cmap,
            edgecolors='none',
            linewidths=0.2
        )
        cbar = plt.colorbar(hb, ax=ax, pad=0.02)
        cbar.set_label("Density (log₁₀ count)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)
    else:
        hb = ax.hexbin(
            x, y,
            C=scores,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            mincnt=1,
            cmap=cmap,
            edgecolors='none',
            linewidths=0.2
        )
        cbar = plt.colorbar(hb, ax=ax, pad=0.02)
        cbar.set_label("Mean Score", fontsize=11)
        cbar.ax.tick_params(labelsize=9)


def plot_heatmap_hist2d(ax, x, y, scores=None, bins=120, cmap=None):
    """
    Square-bin heatmap using histogram2d + imshow with improved styling.
      - If scores is None: density heatmap (log scale).
      - If scores is not None: mean score per bin.
    """
    # Better default colormaps
    if cmap is None:
        cmap = "YlOrRd" if scores is None else "RdYlGn"

    if scores is None:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        # Avoid log(0): only show bins >= 1
        H_masked = np.ma.masked_where(H <= 0, H)

        im = ax.imshow(
            H_masked.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            norm=LogNorm(vmin=1),
            cmap=cmap,
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Density (log scale)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)
    else:
        sumw, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=scores)
        cnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])

        mean = np.divide(sumw, cnt, out=np.full_like(sumw, np.nan), where=cnt > 0)
        im = ax.imshow(
            mean.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap=cmap,
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Mean Score", fontsize=11)
        cbar.ax.tick_params(labelsize=9)


def visualize_tsne(
    task_name: str,
    perplexity: int = 30,
    with_scores: bool = False,
    num_workers: int = 4,
    device: str = "cuda",
    save_path: str = None,
    plot_kind: str = "hexbin",
    gridsize: int = 70,
    hist_bins: int = 120,
    encoder_name: str = None,
    truncate_dim: int = None,
    cmap: str = None,
):
    """
    Main visualization function using t-SNE heatmaps.

    Args:
        task_name: Name of the task (e.g., "tsp_construct")
        perplexity: t-SNE perplexity parameter
        with_scores: Whether to evaluate programs and color by score
        num_workers: Number of parallel workers for evaluation
        device: Device to use for encoding
        save_path: Path to save the plot (default: auto-generated)
        plot_kind: "hexbin" or "hist2d"
        gridsize: Hexbin grid resolution (higher = finer)
        hist_bins: Number of bins for hist2d
        encoder_name: Encoder model name (default: from model_config.py)
        truncate_dim: Matryoshka embedding dimension (default: from model_config.py)
        cmap: Custom colormap name

    Returns:
        Tuple of (z_2d, scores, names)
    """

    # Load heuristics
    heuristics = load_heuristics(task_name)
    names = list(heuristics.keys())
    codes = list(heuristics.values())
    print(f"Loaded {len(codes)} programs from {task_name}")

    if len(codes) < 3:
        raise ValueError("Need at least 3 programs to run t-SNE and make a heatmap.")

    # Load encoder using project's standard loader
    print(f"Loading encoder model: {encoder_name or DEFAULT_ENCODER}")
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

    # Encode programs
    z_embeddings = encode_programs(codes, encoder_model, device)

    # Free encoder memory
    del encoder_model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get scores if requested
    scores = None
    if with_scores:
        scores = get_scores(task_name, codes, num_workers=num_workers)

    # t-SNE
    effective_perplexity = min(perplexity, len(codes) - 1)
    if effective_perplexity < 2:
        effective_perplexity = 2

    print(f"Running t-SNE (perplexity={effective_perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        learning_rate="auto",
        init="pca",
        n_iter=1000,
        verbose=0
    )
    z_2d = tsne.fit_transform(z_embeddings)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    x, y = z_2d[:, 0], z_2d[:, 1]

    if plot_kind == "hexbin":
        plot_heatmap_hexbin(ax, x, y, scores=scores, gridsize=gridsize, cmap=cmap)
    elif plot_kind == "hist2d":
        plot_heatmap_hist2d(ax, x, y, scores=scores, bins=hist_bins, cmap=cmap)
    else:
        raise ValueError(f"Unknown plot_kind: {plot_kind} (use 'hexbin' or 'hist2d')")

    # Cleaner axis labels
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.tick_params(labelsize=10)

    # Clean title without dim/datasize
    task_display = task_name.replace("_", " ").title()
    subtitle = "Density" if scores is None else "Mean Score"
    ax.set_title(
        f"{task_display} Heuristic Embeddings - {subtitle}",
        fontsize=14,
        pad=15
    )

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save or show
    if save_path is None:
        score_suffix = "_scores" if with_scores else ""
        plot_suffix = "hexbin" if plot_kind == "hexbin" else "hist2d"
        save_path = f"{task_name}_tsne_{plot_suffix}{score_suffix}.png"

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
    print(f"\nSaved visualization to: {save_path}")

    plt.show()

    return z_2d, scores, names


def main():
    parser = argparse.ArgumentParser(
        description="Visualize heuristic embeddings with t-SNE heatmap (no points)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Task and model configuration
    parser.add_argument("--task", type=str, default="tsp_construct",
                        help="Task name (e.g., tsp_construct, cvrp_construct)")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                        help=f"Encoder model name (default: {DEFAULT_ENCODER})")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_MATRYOSHKA_DIM,
                        help=f"Matryoshka embedding dimension (default: {DEFAULT_MATRYOSHKA_DIM or 'model native'})")

    # Visualization parameters
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter")
    parser.add_argument("--with_scores", action="store_true",
                        help="Evaluate programs and color heatmap by score")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for evaluation (if --with_scores)")

    # Plot style
    parser.add_argument("--plot", type=str, default="hexbin", choices=["hexbin", "hist2d"],
                        help="Heatmap style")
    parser.add_argument("--gridsize", type=int, default=70,
                        help="Hexbin grid resolution (higher = finer)")
    parser.add_argument("--hist_bins", type=int, default=120,
                        help="Number of bins per axis for hist2d")
    parser.add_argument("--cmap", type=str, default=None,
                        help="Matplotlib colormap name (e.g., viridis, plasma, RdYlGn)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (default: auto-generated)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for encoder")

    args = parser.parse_args()

    print("="*70)
    print("Heuristic Embedding Visualization (t-SNE Heatmap)")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Encoder: {args.encoder}")
    if args.embedding_dim is not None:
        print(f"Embedding dimension: {args.embedding_dim} (Matryoshka)")
    print(f"Plot style: {args.plot}")
    if args.with_scores:
        print(f"Scoring: Enabled ({args.num_workers} workers)")
    else:
        print(f"Scoring: Disabled (density heatmap)")
    print("="*70)
    print()

    visualize_tsne(
        task_name=args.task,
        perplexity=args.perplexity,
        with_scores=args.with_scores,
        num_workers=args.num_workers,
        device=args.device,
        save_path=args.output,
        plot_kind=args.plot,
        gridsize=args.gridsize,
        hist_bins=args.hist_bins,
        encoder_name=args.encoder,
        truncate_dim=args.embedding_dim,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
