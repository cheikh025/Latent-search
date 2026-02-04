"""
Visualize heuristic embeddings with clustering in z-space and u-space (prior space).

This simplified script:
1. Loads heuristics and encodes them to z (encoder embeddings)
2. Performs clustering on z-space
3. Plots t-SNE of z colored by cluster
4. Transforms z -> u using a trained normalizing flow
5. Plots t-SNE of u colored by cluster

Usage:
    python visualize_clusters.py --task tsp_construct
    python visualize_clusters.py --task tsp_construct --n_clusters 5
    python visualize_clusters.py --task tsp_construct --flow_path task/tsp_construct/flow_checkpoints/flow_final.pth
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm

from model_config import DEFAULT_ENCODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder
from normalizing_flow import NormalizingFlow


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


def encode_programs(codes: list, encoder_model, device: str = "cuda", batch_size: int = 32) -> np.ndarray:
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


def load_flow(flow_path: str, device: str = "cuda") -> NormalizingFlow:
    """Load trained normalizing flow model."""
    print(f"Loading normalizing flow from {flow_path}...")

    checkpoint = torch.load(flow_path, map_location=device)

    # Extract architecture parameters
    dim = checkpoint.get('dim', checkpoint.get('embedding_dim', 768))
    num_layers = checkpoint.get('num_layers', 4)
    hidden_dim = checkpoint.get('hidden_dim', 128)
    dropout = checkpoint.get('dropout', 0.0)

    print(f"  Dimension: {dim}, Layers: {num_layers}, Hidden: {hidden_dim}")

    flow_model = NormalizingFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    flow_model.to(device)
    flow_model.eval()

    return flow_model


def transform_z_to_u(z_embeddings: np.ndarray, flow_model: NormalizingFlow, device: str = "cuda") -> np.ndarray:
    """Transform z embeddings to u (prior space) using the flow."""
    z_tensor = torch.tensor(z_embeddings, dtype=torch.float32, device=device)

    with torch.no_grad():
        u, _ = flow_model(z_tensor)

    return u.cpu().numpy()


def visualize_clusters(
    task_name: str,
    n_clusters: int = 5,
    perplexity: int = 30,
    flow_path: str = None,
    device: str = "cuda",
    save_path: str = None,
    encoder_name: str = None,
    truncate_dim: int = None,
):
    """
    Visualize z-space and u-space embeddings colored by cluster.

    Args:
        task_name: Name of the task (e.g., "tsp_construct")
        n_clusters: Number of clusters for KMeans
        perplexity: t-SNE perplexity parameter
        flow_path: Path to trained normalizing flow (optional)
        device: Device for encoding
        save_path: Path to save the plot
        encoder_name: Encoder model name
        truncate_dim: Matryoshka embedding dimension
    """
    # Load heuristics
    heuristics = load_heuristics(task_name)
    names = list(heuristics.keys())
    codes = list(heuristics.values())
    print(f"Loaded {len(codes)} programs from {task_name}")

    if len(codes) < 3:
        raise ValueError("Need at least 3 programs to run t-SNE.")

    # Load encoder
    print(f"Loading encoder: {encoder_name or DEFAULT_ENCODER}")
    encoder_model, embedding_dim = load_encoder(
        model_name=encoder_name,
        device=device,
        truncate_dim=truncate_dim
    )
    print(f"  Embedding dimension: {embedding_dim}")

    # Encode programs to z-space
    z_embeddings = encode_programs(codes, encoder_model, device)

    # Free encoder memory
    del encoder_model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clustering on z-space
    print(f"\nClustering with KMeans (n_clusters={n_clusters})...")
    n_clusters = min(n_clusters, len(codes))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z_embeddings)
    print(f"  Cluster distribution: {np.bincount(cluster_labels)}")

    # t-SNE for z-space
    effective_perplexity = min(perplexity, len(codes) - 1)
    if effective_perplexity < 2:
        effective_perplexity = 2

    print(f"\nRunning t-SNE on z-space (perplexity={effective_perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        learning_rate="auto",
        init="pca",
        max_iter=1000
    )
    z_2d = tsne.fit_transform(z_embeddings)

    # Check if flow is available
    u_2d = None
    if flow_path is not None and Path(flow_path).exists():
        # Load flow and transform z -> u
        flow_model = load_flow(flow_path, device)

        # Check dimension compatibility
        if flow_model.dim != embedding_dim:
            print(f"  Warning: Flow dimension ({flow_model.dim}) != embedding dimension ({embedding_dim})")
            print(f"  Skipping u-space visualization")
        else:
            u_embeddings = transform_z_to_u(z_embeddings, flow_model, device)

            # Verify u is roughly N(0, I)
            print(f"\nPrior space (u) statistics:")
            print(f"  Mean: {u_embeddings.mean():.4f} (target: 0)")
            print(f"  Std: {u_embeddings.std():.4f} (target: 1)")

            # t-SNE for u-space
            print(f"Running t-SNE on u-space...")
            tsne_u = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                random_state=42,
                learning_rate="auto",
                init="pca",
                max_iter=1000
            )
            u_2d = tsne_u.fit_transform(u_embeddings)

            del flow_model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    else:
        if flow_path is not None:
            print(f"\nWarning: Flow path not found: {flow_path}")
        print("Skipping u-space visualization (no flow model provided)")

    # Create plot
    if u_2d is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]

    # Color map for clusters
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    colors = [cmap(label) for label in cluster_labels]

    # Plot z-space
    ax_z = axes[0]
    scatter_z = ax_z.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=cluster_labels,
        cmap='tab10',
        s=80,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    ax_z.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax_z.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax_z.set_title(f"z-space (Encoder Output) - {task_name}", fontsize=14)
    ax_z.grid(True, alpha=0.3, linestyle='--')

    # Add cluster legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i),
                          markersize=10, label=f'Cluster {i}') for i in range(n_clusters)]
    ax_z.legend(handles=handles, loc='upper right', fontsize=9)

    # Plot u-space if available
    if u_2d is not None:
        ax_u = axes[1]
        scatter_u = ax_u.scatter(
            u_2d[:, 0], u_2d[:, 1],
            c=cluster_labels,
            cmap='tab10',
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        ax_u.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax_u.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax_u.set_title(f"u-space (Prior/Gaussian) - {task_name}", fontsize=14)
        ax_u.grid(True, alpha=0.3, linestyle='--')
        ax_u.legend(handles=handles, loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save
    if save_path is None:
        if u_2d is not None:
            save_path = f"{task_name}_clusters_z_u.png"
        else:
            save_path = f"{task_name}_clusters_z.png"

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
    print(f"\nSaved visualization to: {save_path}")

    plt.show()

    return z_2d, u_2d, cluster_labels, names


def main():
    parser = argparse.ArgumentParser(
        description="Visualize heuristic embeddings with clustering in z-space and u-space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--task", type=str, default="tsp_construct",
                        help="Task name (e.g., tsp_construct, cvrp_construct)")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters for KMeans")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter")
    parser.add_argument("--flow_path", type=str, default=None,
                        help="Path to trained normalizing flow (for u-space visualization)")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                        help="Encoder model name")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_MATRYOSHKA_DIM,
                        help="Matryoshka embedding dimension")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for encoding")

    args = parser.parse_args()

    # Auto-detect flow path if not provided
    flow_path = args.flow_path
    if flow_path is None:
        # Try task-specific flow first
        task_flow = Path(f"task/{args.task}/flow_checkpoints/flow_final.pth")
        unified_flow = Path("Flow_Checkpoints/unified_flow_final.pth")

        if task_flow.exists():
            flow_path = str(task_flow)
            print(f"Auto-detected task-specific flow: {flow_path}")
        elif unified_flow.exists():
            flow_path = str(unified_flow)
            print(f"Auto-detected unified flow: {flow_path}")
        else:
            print("No flow model found. Will only visualize z-space.")

    print("=" * 70)
    print("Cluster Visualization: z-space and u-space")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Clusters: {args.n_clusters}")
    print(f"Encoder: {args.encoder}")
    if args.embedding_dim:
        print(f"Embedding dimension: {args.embedding_dim}")
    if flow_path:
        print(f"Flow: {flow_path}")
    print("=" * 70)
    print()

    visualize_clusters(
        task_name=args.task,
        n_clusters=args.n_clusters,
        perplexity=args.perplexity,
        flow_path=flow_path,
        device=args.device,
        save_path=args.output,
        encoder_name=args.encoder,
        truncate_dim=args.embedding_dim,
    )


if __name__ == "__main__":
    main()
