from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from programDB import  ProgramDatabase
from task.tsp_construct.evaluation import  TSPEvaluation
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mapper import Mapper, train_mapper
from normalizing_flow import NormalizingFlow, train_flow, sample_from_flow
from score_predictor import ScorePredictor, train_score_predictor, gradient_ascent_in_prior, adaptive_gradient_search
import os


device = "cuda"
eval = TSPEvaluation()


# Load the model, optionally in float16 precision for faster inference
encoder_model = SentenceTransformer(
    "BAAI/bge-code-v1",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16},
).to(device)
encoder_tokenizer = AutoTokenizer.from_pretrained(    "BAAI/bge-code-v1",
    trust_remote_code=True)
encoder_model.eval()

program_db = ProgramDatabase()

program_db.load_func_from_json("task/tsp_construct/heuristics.json", encoder_model, encoder_tokenizer, eval, device)

# Extract embeddings and scores from the database
embeddings = np.stack(program_db.df['z'].values)  # Shape: (n_programs, embedding_dim)
scores = program_db.df['score'].values  # Shape: (n_programs,)

print(f"Number of programs: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

# Apply t-SNE to reduce to 2D
print("\nApplying t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
embeddings_2d = tsne.fit_transform(embeddings)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 10))

# Create scatter plot colored by performance score
scatter = ax.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=scores,
    cmap='viridis',  # Yellow (high) to purple (low)
    s=100,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar to show score scale
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Performance Score', rotation=270, labelpad=20, fontsize=12)

# Annotate points with their index (optional: you can use function names)
for idx, (x, y) in enumerate(embeddings_2d):
    ax.annotate(
        f"{idx}",
        (x, y),
        fontsize=8,
        alpha=0.6,
        xytext=(3, 3),
        textcoords='offset points'
    )

ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title('Heuristic Embeddings (t-SNE) Colored by Performance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heuristics_tsne_visualization.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to 'heuristics_tsne_visualization.png'")
plt.show()

# Print statistics about clustering
print("\nTop 5 performing heuristics:")
top_5_indices = np.argsort(scores)[-5:][::-1]
for idx in top_5_indices:
    print(f"  Index {idx}: Score = {scores[idx]:.4f}, Position = ({embeddings_2d[idx, 0]:.2f}, {embeddings_2d[idx, 1]:.2f})")

print("\nBottom 5 performing heuristics:")
bottom_5_indices = np.argsort(scores)[:5]
for idx in bottom_5_indices:
    print(f"  Index {idx}: Score = {scores[idx]:.4f}, Position = ({embeddings_2d[idx, 0]:.2f}, {embeddings_2d[idx, 1]:.2f})")

# ============================================================================
# Train Mapper with QWEN2.5 Coder
# ============================================================================
print("\n" + "="*70)
print("Training Mapper with QWEN2.5 Coder")
print("="*70)

# Load QWEN2.5 Coder model
print("\nLoading QWEN2.5-Coder-7B-Instruct...")
decoder_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

decoder_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True
)

# Set padding token if not set
if decoder_tokenizer.pad_token is None:
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

print(f"Model loaded on: {decoder_model.device}")

# Initialize Mapper
print("\nInitializing Mapper...")
input_dim = embeddings.shape[1]  # Embedding dimension from encoder (768 for bge-code-v1)
output_dim = decoder_model.config.hidden_size  # QWEN2.5 hidden size (typically 4096 for 7B)
num_tokens = 16  # Number of soft prompt tokens

mapper_model = Mapper(
    input_dim=input_dim,
    output_dim=output_dim,
    num_tokens=num_tokens
)

print(f"Mapper: {input_dim}D -> {num_tokens} tokens x {output_dim}D")

# Set up optimizer
learning_rate = 1e-4
optimizer = torch.optim.AdamW(mapper_model.parameters(), lr=learning_rate)

# Define skeleton prompt for instruction
skeleton_prompt = "Write a Python function that implements a heuristic for the Traveling Salesman Problem."

# Create checkpoint directory
os.makedirs("Mapper_Checkpoints", exist_ok=True)

# Train the mapper
print("\nStarting training...")
trained_mapper = train_mapper(
    df=program_db.df,
    mapper_model=mapper_model,
    optimizer=optimizer,
    decoder_model=decoder_model,
    decoder_tokenizer=decoder_tokenizer,
    skeleton_prompt=skeleton_prompt,
    batch_size=2,  # Adjust based on GPU memory
    epochs=20,
    accumulation_steps=2,
    max_length=2048,
    verbose=True
)

print("\nTraining complete! Mapper saved to 'Mapper_Checkpoints/Mapper.pth'")

# ============================================================================
# Test Mapper: Reconstruct code from latent vector
# ============================================================================
print("\n" + "="*70)
print("Testing Mapper Reconstruction")
print("="*70)

test_index = 1  # Change this to test different heuristics

# Get the original code and embedding at the test index
original_code = program_db.df.iloc[test_index]['code']
z_vector = program_db.df.iloc[test_index]['z']
original_score = program_db.df.iloc[test_index]['score']

print(f"\nTesting with heuristic at index {test_index}")
print(f"Original score: {original_score:.4f}")
print("\n" + "-"*70)
print("ORIGINAL CODE:")
print("-"*70)
print(original_code)
print("-"*70)

# Prepare the mapper for inference
trained_mapper.eval()

# Get the first device (where embeddings layer is)
embed_layer = decoder_model.get_input_embeddings()
first_dev = embed_layer.weight.device
embed_dtype = embed_layer.weight.dtype

# Convert z_vector to tensor and move to correct device
z_tensor = torch.tensor(z_vector, dtype=torch.float32).unsqueeze(0).to(first_dev)  # [1, input_dim]

# Generate soft prompts from z
with torch.no_grad():
    soft_prompt_embeds = trained_mapper(z_tensor).to(first_dev, dtype=embed_dtype)  # [1, num_tokens, output_dim]

# Build the instruction prompt
instruction_text = skeleton_prompt
instruction_messages = [{"role": "user", "content": instruction_text}]
instruction_ids = decoder_tokenizer.apply_chat_template(
    instruction_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(first_dev)

# Get instruction embeddings
instruction_embeds = embed_layer(instruction_ids)  # [1, seq_len, hidden_dim]

# Concatenate soft prompts with instruction embeddings
inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)  # [1, num_tokens + seq_len, hidden_dim]

# Generate code from the combined embeddings
print("\n" + "-"*70)
print("RECONSTRUCTED CODE:")
print("-"*70)

with torch.no_grad():
    generated_ids = decoder_model.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=decoder_tokenizer.pad_token_id,
        eos_token_id=decoder_tokenizer.eos_token_id
    )

# Decode the generated code
reconstructed_code = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Extract only the code part using robust extraction (removes trailing comments/examples)
from utils import extract_python_code_robust
reconstructed_code = extract_python_code_robust(reconstructed_code, include_preface=True)

print(reconstructed_code)
print("-"*70)

# Compare lengths
print(f"\nOriginal code length: {len(original_code)} characters")
print(f"Reconstructed code length: {len(reconstructed_code)} characters")

# ============================================================================
# Train Normalizing Flow
# ============================================================================
print("\n" + "="*70)
print("Training Normalizing Flow")
print("="*70)

# Create normalizing flow model
embedding_dim = embeddings.shape[1]
flow_model = NormalizingFlow(
    dim=embedding_dim,
    num_layers=8,
    hidden_dim=512
)

num_params = sum(p.numel() for p in flow_model.parameters())
print(f"\nFlow model created with {num_params:,} parameters")
print(f"Input dimension: {embedding_dim}")
print(f"Number of coupling layers: 8")

# Create checkpoint directory
os.makedirs("Flow_Checkpoints", exist_ok=True)

# Train the flow
print("\nStarting flow training...")
trained_flow = train_flow(
    flow_model=flow_model,
    program_db=program_db,
    batch_size=8,  # Small batch for small dataset
    epochs=200,
    lr=1e-3,
    weight_decay=1e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True,
    checkpoint_path="Flow_Checkpoints/flow_checkpoint.pth"
)

print("\nFlow training complete!")

# ============================================================================
# Test Normalizing Flow: Forward and Inverse
# ============================================================================
print("\n" + "="*70)
print("Testing Normalizing Flow - Forward & Inverse")
print("="*70)

trained_flow.eval()
flow_device = "cuda" if torch.cuda.is_available() else "cpu"

# Test on multiple samples from the database
test_indices = [0, 5, 10, 15] if len(program_db.df) > 15 else list(range(min(4, len(program_db.df))))

print("\nTesting forward (z -> u) and inverse (u -> z) transformations:")
print("-"*70)

for test_idx in test_indices:
    # Get original z from database
    original_z = torch.tensor(
        program_db.df.iloc[test_idx]['z'],
        dtype=torch.float32
    ).unsqueeze(0).to(flow_device)

    with torch.no_grad():
        # Forward: z -> u (should map to N(0, I))
        u, log_det = trained_flow(original_z)

        # Inverse: u -> z (should reconstruct original)
        z_reconstructed = trained_flow.inverse(u)

        # Compute reconstruction error
        recon_error = torch.norm(original_z - z_reconstructed).item()

        # Original z statistics
        z_norm = torch.norm(original_z).item()

        # u statistics (should be ~ N(0, I))
        u_norm = torch.norm(u).item()
        u_mean = u.mean().item()
        u_std = u.std().item()

        print(f"\nProgram {test_idx}:")
        print(f"  Original z: norm={z_norm:.4f}")
        print(f"  Mapped to u: norm={u_norm:.4f}, mean={u_mean:.4f}, std={u_std:.4f}")
        print(f"  Expected u: norm≈{np.sqrt(embedding_dim):.2f}, mean≈0.0, std≈1.0")
        print(f"  Reconstruction error: {recon_error:.8f}")
        print(f"  Log |det J|: {log_det.item():.4f}")

# Overall distribution check
print("\n" + "-"*70)
print("Overall Distribution Analysis:")
print("-"*70)

all_z = torch.tensor(np.stack(program_db.df['z'].values), dtype=torch.float32).to(flow_device)

with torch.no_grad():
    all_u, all_log_det = trained_flow(all_z)
    all_z_recon = trained_flow.inverse(all_u)

    # Reconstruction errors
    recon_errors = torch.norm(all_z - all_z_recon, dim=1)

    # u statistics
    u_means = all_u.mean(dim=0)
    u_stds = all_u.std(dim=0)

    print(f"\nAll {len(all_z)} programs:")
    print(f"  Mean reconstruction error: {recon_errors.mean().item():.8f}")
    print(f"  Max reconstruction error: {recon_errors.max().item():.8f}")
    print(f"\nMapped u statistics (should be N(0,I)):")
    print(f"  Mean across all dims: {u_means.mean().item():.6f} (target: 0.0)")
    print(f"  Std of means: {u_means.std().item():.6f}")
    print(f"  Mean of stds: {u_stds.mean().item():.6f} (target: 1.0)")
    print(f"  Std of stds: {u_stds.std().item():.6f}")

# ============================================================================
# Sample New Latent Codes from Flow
# ============================================================================
print("\n" + "="*70)
print("Sampling New Latent Codes from Flow")
print("="*70)

num_samples = 5
print(f"\nSampling {num_samples} new latent codes from N(0, I)...")

sampled_z = sample_from_flow(trained_flow, num_samples=num_samples, device=flow_device)

print(f"\nGenerated samples:")
for i, z in enumerate(sampled_z):
    z_norm = torch.norm(z).item()
    print(f"  Sample {i+1}: norm={z_norm:.4f}, shape={z.shape}")

# Compare with original distribution
original_norms = torch.norm(all_z, dim=1)
sampled_norms = torch.norm(sampled_z, dim=1)

print(f"\nDistribution comparison:")
print(f"  Original z - Mean norm: {original_norms.mean().item():.4f}, Std: {original_norms.std().item():.4f}")
print(f"  Sampled z  - Mean norm: {sampled_norms.mean().item():.4f}, Std: {sampled_norms.std().item():.4f}")

# Save the trained flow
flow_save_path = "Flow_Checkpoints/normalizing_flow_final.pth"
torch.save({
    'model_state_dict': trained_flow.state_dict(),
    'dim': embedding_dim,
    'num_layers': 8,
    'hidden_dim': 512,
}, flow_save_path)
print(f"\nFlow model saved to {flow_save_path}")

# ============================================================================
# Train Score Predictor for Gradient-Based Search
# ============================================================================
print("\n" + "="*70)
print("Training Score Predictor R(u)")
print("="*70)

# Create score predictor (simple 2-layer MLP)
predictor = ScorePredictor(
    input_dim=embedding_dim,
    hidden_dim=256
)

num_predictor_params = sum(p.numel() for p in predictor.parameters())
print(f"\nScore predictor created with {num_predictor_params:,} parameters")
print(f"Architecture: {embedding_dim} -> 256 -> 1 (2-layer MLP)")

# Create checkpoint directory
os.makedirs("Predictor_Checkpoints", exist_ok=True)

# Train the predictor
print("\nStarting predictor training...")
trained_predictor = train_score_predictor(
    predictor=predictor,
    program_db=program_db,
    flow_model=trained_flow,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    device=device,
    verbose=True
)

print("\nPredictor training complete!")

# ============================================================================
# Test Score Predictor: Predict Scores
# ============================================================================
print("\n" + "="*70)
print("Testing Score Predictor")
print("="*70)

trained_predictor.eval()

# Test on known programs
print("\nPredicting scores for known programs:")
print("-"*70)

test_indices = [0, 5, 10, 15, 20] if len(program_db.df) > 20 else list(range(min(5, len(program_db.df))))

for test_idx in test_indices:
    # Get z and true score
    z_test = torch.tensor(
        program_db.df.iloc[test_idx]['z'],
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    true_score = program_db.df.iloc[test_idx]['score']

    with torch.no_grad():
        # Map z to u
        u_test, _ = trained_flow(z_test)

        # Predict score
        pred_score = trained_predictor(u_test)

        # Denormalize if needed
        if hasattr(trained_predictor, 'score_mean') and hasattr(trained_predictor, 'score_std'):
            pred_score = pred_score * trained_predictor.score_std + trained_predictor.score_mean

        pred_score = pred_score.item()

    error = abs(pred_score - true_score)
    print(f"\nProgram {test_idx}:")
    print(f"  True score: {true_score:.4f}")
    print(f"  Predicted:  {pred_score:.4f}")
    print(f"  Error:      {error:.4f}")

# Overall accuracy
print("\n" + "-"*70)
print("Overall Prediction Accuracy:")
print("-"*70)

all_z = torch.tensor(np.stack(program_db.df['z'].values), dtype=torch.float32).to(device)
all_true_scores = program_db.df['score'].values

with torch.no_grad():
    all_u, _ = trained_flow(all_z)
    all_pred_scores = trained_predictor(all_u)

    if hasattr(trained_predictor, 'score_mean') and hasattr(trained_predictor, 'score_std'):
        all_pred_scores = all_pred_scores * trained_predictor.score_std + trained_predictor.score_mean

    all_pred_scores = all_pred_scores.squeeze().cpu().numpy()

mae = np.mean(np.abs(all_pred_scores - all_true_scores))
rmse = np.sqrt(np.mean((all_pred_scores - all_true_scores)**2))
correlation = np.corrcoef(all_true_scores, all_pred_scores)[0, 1]

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Correlation: {correlation:.4f}")

# ============================================================================
# Gradient Ascent in Prior Space
# ============================================================================
print("\n" + "="*70)
print("Gradient Ascent to Find High-Scoring Regions")
print("="*70)

# Random initialization
print("\n1. Random initialization (sample from N(0, I)):")
print("-"*70)

optimized_u_random, optimized_z_random = gradient_ascent_in_prior(
    predictor=trained_predictor,
    flow_model=trained_flow,
    num_starts=5,
    steps=100,
    lr=0.01,
    init_method='random',
    device=device,
    verbose=True
)

# Adaptive initialization (from top programs)
print("\n" + "="*70)
print("2. Adaptive gradient search (initialize from top programs):")
print("-"*70)

optimized_u_adaptive, optimized_z_adaptive = adaptive_gradient_search(
    predictor=trained_predictor,
    flow_model=trained_flow,
    program_db=program_db,
    num_searches=5,
    steps_per_search=100,
    lr=0.01,
    init_from_top_k=5,
    device=device,
    verbose=True
)

# Compare strategies
print("\n" + "="*70)
print("Comparison: Random vs Adaptive Initialization")
print("="*70)

# Get predicted scores
with torch.no_grad():
    pred_random = trained_predictor(optimized_u_random)
    pred_adaptive = trained_predictor(optimized_u_adaptive)

    if hasattr(trained_predictor, 'score_mean') and hasattr(trained_predictor, 'score_std'):
        pred_random = pred_random * trained_predictor.score_std + trained_predictor.score_mean
        pred_adaptive = pred_adaptive * trained_predictor.score_std + trained_predictor.score_mean

    pred_random = pred_random.squeeze().cpu().numpy()
    pred_adaptive = pred_adaptive.squeeze().cpu().numpy()

print("\nRandom initialization:")
print(f"  Predicted scores: {pred_random}")
print(f"  Mean: {pred_random.mean():.4f}, Max: {pred_random.max():.4f}")

print("\nAdaptive initialization:")
print(f"  Predicted scores: {pred_adaptive}")
print(f"  Mean: {pred_adaptive.mean():.4f}, Max: {pred_adaptive.max():.4f}")

print("\nTop 3 existing programs:")
top_3 = program_db.get_top_n(3)
print(f"  Scores: {top_3['score'].values}")

# Save optimized z vectors for code generation
print("\n" + "-"*70)
print("Saving optimized latent vectors...")
print("-"*70)

torch.save({
    'z_random': optimized_z_random.cpu(),
    'z_adaptive': optimized_z_adaptive.cpu(),
    'pred_scores_random': pred_random,
    'pred_scores_adaptive': pred_adaptive
}, "Predictor_Checkpoints/optimized_latents.pth")

print("✓ Saved to 'Predictor_Checkpoints/optimized_latents.pth'")
print("\nThese optimized z vectors can now be used to generate new candidate programs!")

# Save the trained predictor
predictor_save_path = "Predictor_Checkpoints/score_predictor.pth"
torch.save({
    'model_state_dict': trained_predictor.state_dict(),
    'input_dim': embedding_dim,
    'hidden_dim': 256,
    'score_mean': trained_predictor.score_mean if hasattr(trained_predictor, 'score_mean') else None,
    'score_std': trained_predictor.score_std if hasattr(trained_predictor, 'score_std') else None,
}, predictor_save_path)
print(f"\n✓ Predictor saved to {predictor_save_path}")