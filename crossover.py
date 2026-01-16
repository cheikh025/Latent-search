"""
Prior-space crossover via interpolation using normalizing flow.

Implements Step 4 from the paper:
"Prior-space crossover: pick two strong parents pA, pB using true evaluated scores;
compute u_A = F_φ(E(p_A)), u_B = F_φ(E(p_B)), form u(α) = (1-α)u_A + αu_B,
map back z(α) = F_φ^(-1)(u(α)), and decode p̂(α)."
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from programDB import ProgramDatabase


class PriorSpaceCrossover:
    """
    Perform prior-space crossover using normalizing flow.

    This enables safe latent recombination by:
    1. Encoding parent programs: z_A = E(p_A), z_B = E(p_B)
    2. Mapping to prior space: u_A = F_φ(z_A), u_B = F_φ(z_B)
    3. Interpolating: u(α) = (1-α)u_A + αu_B
    4. Mapping back: z(α) = F_φ^(-1)(u(α))
    5. Decoding to program: p̂(α) = Decoder(z(α))
    """

    def __init__(self, flow_model, device='cuda'):
        """
        Args:
            flow_model: Trained normalizing flow (F_φ)
            device: Device for computation
        """
        self.flow = flow_model
        self.device = device
        self.flow.eval()

    def select_parents(
        self,
        program_db: ProgramDatabase,
        top_q_percent: float = 0.1,
        num_parents: int = 2,
        require_valid: bool = True,
        min_score: Optional[float] = None
    ) -> List[int]:
        """
        Select high-performing parent programs from database.

        Args:
            program_db: ProgramDatabase instance
            top_q_percent: Select from top q% by score (e.g., 0.1 = top 10%)
            num_parents: Number of parents to select
            require_valid: Only select programs with finite scores
            min_score: Minimum score threshold (optional)

        Returns:
            List of program IDs for selected parents
        """
        df = program_db.df.copy()

        # Filter valid programs
        if require_valid:
            df = df[df['score'].notna() & np.isfinite(df['score'])]

        # Filter by minimum score
        if min_score is not None:
            df = df[df['score'] >= min_score]

        if len(df) < num_parents:
            raise ValueError(f"Not enough valid programs. Found {len(df)}, need {num_parents}")

        # Sort by score (higher is better, since s = -y)
        df_sorted = df.sort_values('score', ascending=False)

        # Select from top q%
        top_k = max(num_parents, int(len(df_sorted) * top_q_percent))
        top_programs = df_sorted.head(top_k)

        # Randomly sample num_parents from top programs
        selected_ids = np.random.choice(
            top_programs.index.tolist(),
            size=num_parents,
            replace=False
        ).tolist()

        return selected_ids

    def crossover_two_parents(
        self,
        z_A: torch.Tensor,
        z_B: torch.Tensor,
        num_offspring: int = 5,
        alpha_range: Tuple[float, float] = (0.0, 1.0)
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Perform crossover between two parent latent codes.

        Args:
            z_A: First parent embedding [dim]
            z_B: Second parent embedding [dim]
            num_offspring: Number of interpolated offspring to generate
            alpha_range: Range for interpolation parameter (min, max)

        Returns:
            z_offspring: Offspring embeddings [num_offspring, dim]
            alphas: Interpolation parameters used [num_offspring]
        """
        self.flow.eval()

        with torch.no_grad():
            # Ensure inputs are on correct device
            z_A = z_A.to(self.device).unsqueeze(0) if z_A.dim() == 1 else z_A.to(self.device)
            z_B = z_B.to(self.device).unsqueeze(0) if z_B.dim() == 1 else z_B.to(self.device)

            # Map to prior space: u = F_φ(z)
            u_A, _ = self.flow(z_A)
            u_B, _ = self.flow(z_B)

            # Generate interpolation parameters
            alphas = np.linspace(alpha_range[0], alpha_range[1], num_offspring)

            # Interpolate in prior space: u(α) = (1-α)u_A + αu_B
            z_offspring_list = []

            for alpha in alphas:
                u_interp = (1 - alpha) * u_A + alpha * u_B

                # Map back to latent space: z(α) = F_φ^(-1)(u(α))
                z_interp = self.flow.inverse(u_interp)
                z_offspring_list.append(z_interp)

            # Stack offspring
            z_offspring = torch.cat(z_offspring_list, dim=0)

        return z_offspring, alphas

    def multi_parent_crossover(
        self,
        z_parents: List[torch.Tensor],
        num_offspring: int = 10,
        interpolation_method: str = 'spherical'
    ) -> torch.Tensor:
        """
        Perform crossover with multiple parents (>2).

        Args:
            z_parents: List of parent embeddings, each [dim]
            num_offspring: Number of offspring to generate
            interpolation_method: 'linear' or 'spherical'

        Returns:
            z_offspring: Offspring embeddings [num_offspring, dim]
        """
        self.flow.eval()

        with torch.no_grad():
            # Map all parents to prior space
            u_parents = []
            for z in z_parents:
                z = z.to(self.device).unsqueeze(0) if z.dim() == 1 else z.to(self.device)
                u, _ = self.flow(z)
                u_parents.append(u.squeeze(0))

            u_parents = torch.stack(u_parents)  # [num_parents, dim]

            # Generate offspring
            z_offspring_list = []

            for _ in range(num_offspring):
                if interpolation_method == 'linear':
                    # Random convex combination
                    weights = torch.rand(len(z_parents), device=self.device)
                    weights = weights / weights.sum()

                    u_interp = torch.sum(weights.unsqueeze(1) * u_parents, dim=0, keepdim=True)

                elif interpolation_method == 'spherical':
                    # Spherical interpolation (better for high-dimensional spaces)
                    # Randomly select two parents and interpolate
                    idx = np.random.choice(len(z_parents), size=2, replace=False)
                    u_A, u_B = u_parents[idx[0]], u_parents[idx[1]]

                    # Random interpolation parameter
                    alpha = np.random.rand()

                    # Normalize and interpolate
                    u_A_norm = u_A / (torch.norm(u_A) + 1e-8)
                    u_B_norm = u_B / (torch.norm(u_B) + 1e-8)

                    u_interp = (1 - alpha) * u_A_norm + alpha * u_B_norm

                    # Scale back to similar magnitude
                    avg_norm = (torch.norm(u_A) + torch.norm(u_B)) / 2
                    u_interp = u_interp / (torch.norm(u_interp) + 1e-8) * avg_norm
                    u_interp = u_interp.unsqueeze(0)

                else:
                    raise ValueError(f"Unknown interpolation method: {interpolation_method}")

                # Map back to latent space
                z_interp = self.flow.inverse(u_interp)
                z_offspring_list.append(z_interp)

            z_offspring = torch.cat(z_offspring_list, dim=0)

        return z_offspring

    def crossover_from_database(
        self,
        program_db: ProgramDatabase,
        num_pairs: int = 5,
        offspring_per_pair: int = 5,
        top_q_percent: float = 0.2,
        alpha_range: Tuple[float, float] = (0.2, 0.8)
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], np.ndarray]:
        """
        Generate offspring by crossing over top programs from database.

        Args:
            program_db: ProgramDatabase instance
            num_pairs: Number of parent pairs to select
            offspring_per_pair: Number of offspring per pair
            top_q_percent: Select parents from top q%
            alpha_range: Range for interpolation (excludes exact copies at 0 and 1)

        Returns:
            all_offspring: All generated offspring embeddings [num_pairs*offspring_per_pair, dim]
            parent_pairs: List of (parent_A_id, parent_B_id) tuples
            alphas: Interpolation parameters [num_pairs*offspring_per_pair]
        """
        all_offspring = []
        parent_pairs = []
        all_alphas = []

        for _ in range(num_pairs):
            # Select two parents
            parent_ids = self.select_parents(
                program_db,
                top_q_percent=top_q_percent,
                num_parents=2
            )

            # Get embeddings
            z_A = torch.tensor(program_db.get_by_id(parent_ids[0])['z'], dtype=torch.float32)
            z_B = torch.tensor(program_db.get_by_id(parent_ids[1])['z'], dtype=torch.float32)

            # Perform crossover
            offspring, alphas = self.crossover_two_parents(
                z_A, z_B,
                num_offspring=offspring_per_pair,
                alpha_range=alpha_range
            )

            all_offspring.append(offspring)
            parent_pairs.append((parent_ids[0], parent_ids[1]))
            all_alphas.append(alphas)

        # Concatenate all offspring
        all_offspring = torch.cat(all_offspring, dim=0)
        all_alphas = np.concatenate(all_alphas)

        return all_offspring, parent_pairs, all_alphas

    def adaptive_crossover(
        self,
        z_A: torch.Tensor,
        z_B: torch.Tensor,
        score_A: float,
        score_B: float,
        num_offspring: int = 5
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Adaptive crossover: bias interpolation toward better parent.

        Args:
            z_A: First parent embedding
            z_B: Second parent embedding
            score_A: Score of parent A (higher is better)
            score_B: Score of parent B
            num_offspring: Number of offspring

        Returns:
            z_offspring: Offspring embeddings [num_offspring, dim]
            alphas: Interpolation parameters used [num_offspring]
        """
        # Compute weights based on scores
        total_score = score_A + score_B
        if total_score <= 0:
            # If both scores are negative, use uniform interpolation
            return self.crossover_two_parents(z_A, z_B, num_offspring)

        # Bias toward better parent
        weight_A = score_A / total_score
        weight_B = score_B / total_score

        # Generate biased alphas (more samples near better parent)
        if weight_A > weight_B:
            # Bias toward A (lower alpha values)
            alphas = np.random.beta(2, 5, num_offspring)
        else:
            # Bias toward B (higher alpha values)
            alphas = np.random.beta(5, 2, num_offspring)

        self.flow.eval()

        with torch.no_grad():
            z_A = z_A.to(self.device).unsqueeze(0) if z_A.dim() == 1 else z_A.to(self.device)
            z_B = z_B.to(self.device).unsqueeze(0) if z_B.dim() == 1 else z_B.to(self.device)

            u_A, _ = self.flow(z_A)
            u_B, _ = self.flow(z_B)

            z_offspring_list = []

            for alpha in alphas:
                u_interp = (1 - alpha) * u_A + alpha * u_B
                z_interp = self.flow.inverse(u_interp)
                z_offspring_list.append(z_interp)

            z_offspring = torch.cat(z_offspring_list, dim=0)

        return z_offspring, alphas


def test_crossover():
    """Test crossover functionality"""
    print("="*70)
    print("Testing Prior-Space Crossover")
    print("="*70)

    from normalizing_flow import NormalizingFlow

    # Create dummy flow
    dim = 768
    flow = NormalizingFlow(dim=dim, num_layers=4, hidden_dim=256)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flow = flow.to(device)

    # Initialize ActNorm layers with dummy data
    print("Initializing flow with dummy data...")
    dummy_data = torch.randn(32, dim, device=device)
    with torch.no_grad():
        _ = flow(dummy_data)

    flow.eval()

    # Create crossover operator
    crossover_op = PriorSpaceCrossover(flow, device=device)

    # Test with random embeddings
    z_A = torch.randn(dim)
    z_B = torch.randn(dim)

    print(f"\nParent A norm: {torch.norm(z_A).item():.4f}")
    print(f"Parent B norm: {torch.norm(z_B).item():.4f}")

    # Perform crossover
    offspring, alphas = crossover_op.crossover_two_parents(
        z_A, z_B,
        num_offspring=5,
        alpha_range=(0.2, 0.8)
    )

    print(f"\nGenerated {len(offspring)} offspring")
    print(f"Alphas: {alphas}")
    print(f"Offspring norms: {[torch.norm(z).item() for z in offspring]}")

    # Test multi-parent crossover
    print("\n" + "="*70)
    print("Testing Multi-Parent Crossover")
    print("="*70)

    z_parents = [torch.randn(dim) for _ in range(4)]

    offspring_multi = crossover_op.multi_parent_crossover(
        z_parents,
        num_offspring=10,
        interpolation_method='spherical'
    )

    print(f"\nGenerated {len(offspring_multi)} offspring from {len(z_parents)} parents")
    print(f"Offspring shape: {offspring_multi.shape}")


if __name__ == '__main__':
    test_crossover()
