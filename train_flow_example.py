"""
Example script demonstrating how to train the normalizing flow
on program embeddings from the ProgramDatabase.

This implements Step 3 from the paper:
"Flow fitting: minimize L_flow(φ) on {z_j}^M_{j=1}"
"""

import torch
import numpy as np
from programDB import ProgramDatabase
from normalizing_flow import NormalizingFlow, train_flow, sample_from_flow
import json
import os


def example_train_flow_on_database():
    """
    Example: Train normalizing flow on existing program database
    """
    print("="*70)
    print("Training Normalizing Flow on Program Database")
    print("="*70)

    # ============================================================
    # Step 1: Load or create a ProgramDatabase with embeddings
    # ============================================================
    db = ProgramDatabase()

    # Option A: Load from disk if you have a saved database
    db_path = "program_database.parquet"
    if os.path.exists(db_path):
        print(f"\nLoading existing database from {db_path}...")
        db.load_from_disk(db_path)
        print(f"Loaded {len(db)} programs from database")
    else:
        # Option B: Initialize from JSON (if you have initial programs)
        print("\nNo existing database found.")
        print("You need to initialize the database with programs first.")
        print("Example: db.load_func_from_json('initial_programs.json', encoder, tokenizer, evaluator)")
        return

    # Check if database has embeddings
    if len(db) == 0:
        print("Database is empty! Please populate it with programs first.")
        return

    # Get embedding dimension from first program
    first_z = db.df['z'].iloc[0]
    embedding_dim = first_z.shape[0]
    print(f"\nEmbedding dimension: {embedding_dim}")
    print(f"Number of programs: {len(db)}")

    # ============================================================
    # Step 2: Create Normalizing Flow model
    # ============================================================
    print("\n" + "="*70)
    print("Creating Normalizing Flow Model")
    print("="*70)

    flow_model = NormalizingFlow(
        dim=embedding_dim,
        num_layers=8,         # Number of coupling layers
        hidden_dim=512        # Hidden dimension for coupling networks
    )

    num_params = sum(p.numel() for p in flow_model.parameters())
    print(f"Flow model created with {num_params:,} parameters")

    # ============================================================
    # Step 3: Train the flow model
    # ============================================================
    print("\n" + "="*70)
    print("Training Flow Model")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    trained_flow = train_flow(
        flow_model=flow_model,
        program_db=db,
        batch_size=64,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-5,
        device=device,
        verbose=True,
        checkpoint_path="flow_checkpoint.pth"
    )

    # ============================================================
    # Step 4: Test the trained flow
    # ============================================================
    print("\n" + "="*70)
    print("Testing Trained Flow")
    print("="*70)

    trained_flow.eval()

    # Test on a random program from database
    test_idx = np.random.randint(0, len(db))
    test_program = db.get_by_id(test_idx)
    test_z = torch.tensor(test_program['z'], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward: z -> u (should be ~ N(0, I))
        u, log_det = trained_flow(test_z)

        # Inverse: u -> z (should reconstruct)
        z_recon = trained_flow.inverse(u)

        print(f"\nTest on program {test_idx}:")
        print(f"Original z norm: {torch.norm(test_z).item():.4f}")
        print(f"Mapped u norm: {torch.norm(u).item():.4f} (should be ~ sqrt(dim) = {np.sqrt(embedding_dim):.2f})")
        print(f"Reconstruction error: {torch.norm(test_z - z_recon).item():.6f}")
        print(f"Log determinant: {log_det.item():.4f}")

    # ============================================================
    # Step 5: Sample new latent codes from the flow
    # ============================================================
    print("\n" + "="*70)
    print("Sampling New Latent Codes from Flow")
    print("="*70)

    num_samples = 10
    new_z_samples = sample_from_flow(trained_flow, num_samples=num_samples, device=device)

    print(f"\nGenerated {num_samples} new latent codes")
    print(f"Sample z shapes: {new_z_samples.shape}")
    print(f"Mean z norm: {torch.norm(new_z_samples, dim=1).mean().item():.4f}")
    print(f"Std z norm: {torch.norm(new_z_samples, dim=1).std().item():.4f}")

    # Compare with original distribution
    original_z = torch.tensor(np.stack(db.df['z'].tolist()), dtype=torch.float32)
    original_norms = torch.norm(original_z, dim=1)
    print(f"\nOriginal programs z norm - Mean: {original_norms.mean().item():.4f}, Std: {original_norms.std().item():.4f}")

    # ============================================================
    # Step 6: Save the trained flow
    # ============================================================
    print("\n" + "="*70)
    print("Saving Trained Flow")
    print("="*70)

    flow_save_path = "normalizing_flow_final.pth"
    torch.save({
        'model_state_dict': trained_flow.state_dict(),
        'dim': embedding_dim,
        'num_layers': 8,
        'hidden_dim': 512,
    }, flow_save_path)
    print(f"Flow model saved to {flow_save_path}")

    return trained_flow, db


def example_load_and_use_flow():
    """
    Example: Load a pre-trained flow and use it for inference
    """
    print("\n" + "="*70)
    print("Loading Pre-trained Flow")
    print("="*70)

    flow_path = "normalizing_flow_final.pth"

    if not os.path.exists(flow_path):
        print(f"Flow checkpoint not found at {flow_path}")
        print("Train the flow first using example_train_flow_on_database()")
        return

    # Load checkpoint
    checkpoint = torch.load(flow_path, map_location='cpu')

    # Recreate model
    flow_model = NormalizingFlow(
        dim=checkpoint['dim'],
        num_layers=checkpoint['num_layers'],
        hidden_dim=checkpoint['hidden_dim']
    )
    flow_model.load_state_dict(checkpoint['model_state_dict'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flow_model = flow_model.to(device)
    flow_model.eval()

    print(f"Loaded flow model from {flow_path}")
    print(f"Dimension: {checkpoint['dim']}")

    # Sample some latent codes
    num_samples = 5
    z_samples = sample_from_flow(flow_model, num_samples, device)

    print(f"\nSampled {num_samples} latent codes:")
    for i, z in enumerate(z_samples):
        print(f"  Sample {i+1}: norm={torch.norm(z).item():.4f}")

    return flow_model, z_samples


def visualize_flow_distribution(flow_model, db, device='cuda', num_samples=1000):
    """
    Visualize how well the flow maps z to standard Gaussian
    """
    print("\n" + "="*70)
    print("Analyzing Flow Distribution")
    print("="*70)

    flow_model.eval()

    # Get all z from database
    z_array = np.stack(db.df['z'].tolist()).astype(np.float32)
    z_tensor = torch.tensor(z_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Map through flow
        u, log_det = flow_model(z_tensor)

        u_cpu = u.cpu().numpy()

        # Compute statistics
        u_mean = u_cpu.mean(axis=0)
        u_std = u_cpu.std(axis=0)

        print(f"\nMapped to base distribution (u):")
        print(f"  Mean of means: {u_mean.mean():.6f} (target: 0.0)")
        print(f"  Std of means: {u_mean.std():.6f}")
        print(f"  Mean of stds: {u_std.mean():.6f} (target: 1.0)")
        print(f"  Std of stds: {u_std.std():.6f}")

        # Check Gaussianity with simple test
        u_flat = u_cpu.flatten()
        print(f"\n  Flattened u statistics:")
        print(f"    Mean: {u_flat.mean():.6f}")
        print(f"    Std: {u_flat.std():.6f}")
        print(f"    Min: {u_flat.min():.4f}")
        print(f"    Max: {u_flat.max():.4f}")

        # Mean log determinant
        print(f"\n  Mean log |det J|: {log_det.mean().item():.4f}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Normalizing Flow Training Example")
    print("="*70)

    # Check if database exists
    if os.path.exists("program_database.parquet"):
        # Train flow on existing database
        trained_flow, db = example_train_flow_on_database()

        # Analyze the distribution
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        visualize_flow_distribution(trained_flow, db, device=device)

    else:
        print("\nNo program database found.")
        print("\nTo use this script:")
        print("1. First populate a ProgramDatabase with programs and embeddings")
        print("2. Save it using: db.to_disk('program_database.parquet')")
        print("3. Then run this script to train the normalizing flow")
        print("\nExample initialization:")
        print("  from programDB import ProgramDatabase")
        print("  db = ProgramDatabase()")
        print("  # Add programs with embeddings...")
        print("  db.add_program(code, z, score)")
        print("  db.to_disk('program_database.parquet')")
