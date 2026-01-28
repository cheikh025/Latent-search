"""
Visualize heuristic embeddings using t-SNE with score heatmap.

Usage:
    python visualize_embeddings.py --task tsp_construct
    python visualize_embeddings.py --task tsp_construct --perplexity 30
    python visualize_embeddings.py --task cvrp_construct --cmap plasma
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_heuristics(task_name: str) -> dict:
    """Load heuristics from JSON file."""
    path = Path(f"task/{task_name}/heuristics.json")
    if not path.exists():
        raise FileNotFoundError(f"Heuristics not found: {path}")

    with open(path, 'r') as f:
        return json.load(f)


def get_encoder(device: str = 'cuda'):
    """Load the encoder model."""
    print("Loading encoder (BAAI/bge-code-v1)...")
    encoder = SentenceTransformer(
        "BAAI/bge-code-v1",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    ).to(device)
    encoder.eval()
    return encoder


def encode_programs(codes: list, encoder, device: str = 'cuda', batch_size: int = 32) -> np.ndarray:
    """Encode programs to z-space embeddings."""
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Encoding"):
            batch = codes[i:i+batch_size]
            batch_emb = encoder.encode(
                batch,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            embeddings.append(batch_emb.cpu().numpy())

    return np.vstack(embeddings)


def get_scores(task_name: str, codes: list, num_workers: int = 4) -> np.ndarray:
    """Evaluate programs to get scores for heatmap coloring."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ranking_score_predictor_z import get_evaluator
    from base.evaluate import SecureEvaluator

    print(f"Evaluating {len(codes)} programs...")
    evaluator = SecureEvaluator(get_evaluator(task_name), debug_mode=False)

    scores = [None] * len(codes)

    def eval_one(idx_code):
        idx, code = idx_code
        try:
            score = evaluator.evaluate_program(code)
            return idx, score if score is not None and np.isfinite(score) else None
        except:
            return idx, None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(eval_one, (i, c)) for i, c in enumerate(codes)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            idx, score = f.result()
            scores[idx] = score

    # Replace None with min score for visualization (failed = worst)
    valid_scores = [s for s in scores if s is not None]
    min_score = min(valid_scores) if valid_scores else 0
    scores = [s if s is not None else min_score for s in scores]

    return np.array(scores)


def visualize_tsne(
    task_name: str,
    perplexity: int = 30,
    device: str = 'cuda',
    save_path: str = None,
    cmap: str = 'RdYlGn',
    num_workers: int = 4
):
    """Main visualization function with score heatmap."""

    # Load heuristics
    heuristics = load_heuristics(task_name)
    names = list(heuristics.keys())
    codes = list(heuristics.values())
    print(f"Loaded {len(codes)} heuristics from {task_name}")

    # Encode
    encoder = get_encoder(device)
    z_embeddings = encode_programs(codes, encoder, device)
    print(f"Embeddings shape: {z_embeddings.shape}")

    # Free encoder memory
    del encoder
    torch.cuda.empty_cache()

    # Get scores for heatmap
    scores = get_scores(task_name, codes, num_workers)
    print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")

    # t-SNE
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(codes) - 1),
        random_state=42,
        n_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    z_2d = tsne.fit_transform(z_embeddings)

    # Plot with heatmap
    fig, ax = plt.subplots(figsize=(12, 9))

    # Normalize scores for better color distribution
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Sort by score so best points are drawn on top
    sort_idx = np.argsort(scores)
    z_2d_sorted = z_2d[sort_idx]
    scores_sorted = scores[sort_idx]

    # Scatter plot with heatmap colors
    scatter = ax.scatter(
        z_2d_sorted[:, 0], z_2d_sorted[:, 1],
        c=scores_sorted,
        cmap=cmap,
        alpha=0.8,
        s=80,
        edgecolors='white',
        linewidths=0.5
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Score (higher = better)', fontsize=12)

    # Mark best and worst
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)

    ax.scatter(z_2d[best_idx, 0], z_2d[best_idx, 1],
               c='none', s=300, edgecolors='gold', linewidths=3,
               label=f'Best: {scores[best_idx]:.4f}')
    ax.scatter(z_2d[worst_idx, 0], z_2d[worst_idx, 1],
               c='none', s=300, edgecolors='red', linewidths=3,
               label=f'Worst: {scores[worst_idx]:.4f}')

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f't-SNE of {task_name} Heuristics\n{len(codes)} programs | z-dim={z_embeddings.shape[1]}',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)

    # Stats text box
    stats_text = f"Score Stats:\n  Mean: {scores.mean():.4f}\n  Std: {scores.std():.4f}\n  Min: {scores.min():.4f}\n  Max: {scores.max():.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = f"{task_name}_tsne_heatmap.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")

    plt.show()

    return z_2d, scores, names


def main():
    parser = argparse.ArgumentParser(description='Visualize heuristic embeddings with t-SNE heatmap')
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output', type=str, default=None, help='Output path for plot')
    parser.add_argument('--cmap', type=str, default='RdYlGn', help='Colormap (RdYlGn, viridis, plasma, coolwarm)')
    parser.add_argument('--num_workers', type=int, default=4, help='Parallel workers for evaluation')

    args = parser.parse_args()

    visualize_tsne(
        task_name=args.task,
        perplexity=args.perplexity,
        device=args.device,
        save_path=args.output,
        cmap=args.cmap,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
