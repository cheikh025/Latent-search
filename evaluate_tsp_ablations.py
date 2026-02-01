"""
Evaluate TSP Ablation Experiments

This script evaluates different ablation versions for TSP heuristic generation.
For each ablation, it reports:
1. Number of generated heuristics
2. Success rate (% of valid functions out of 100)
3. Best heuristic from generation (by score)
4. Performance on large benchmark test set
5. Average evaluation time

Usage:
    python evaluate_tsp_ablations.py --ablation_dir tsp_ablation
    python evaluate_tsp_ablations.py --ablation_dir tsp_ablation --n_benchmark 500 --problem_size 100
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import time
from tqdm import tqdm

# Import TSP evaluator
from base.evaluate import SecureEvaluator
from task.tsp_construct.evaluation import TSPEvaluation


class TSPAblationEvaluator:
    """Evaluate TSP ablation experiments."""

    def __init__(
        self,
        ablation_dir: str,
        n_benchmark: int = 500,
        problem_size: int = 100,
        timeout_seconds: int = 30,
        output_dir: str = "ablation_results"
    ):
        """
        Initialize the ablation evaluator.

        Args:
            ablation_dir: Directory containing ablation JSON files
            n_benchmark: Number of test instances for benchmark evaluation
            problem_size: TSP problem size for benchmark
            timeout_seconds: Timeout for each evaluation
            output_dir: Directory to save results
        """
        self.ablation_dir = Path(ablation_dir)
        self.n_benchmark = n_benchmark
        self.problem_size = problem_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize TSP evaluator with large benchmark set
        print(f"Initializing TSP evaluator (n={n_benchmark}, size={problem_size})...")
        base_evaluator = TSPEvaluation(
            n_instance=n_benchmark,
            problem_size=problem_size,
            timeout_seconds=timeout_seconds
        )
        self.evaluator = SecureEvaluator(base_evaluator, debug_mode=False)

        print(f"  ✓ Evaluator ready with {n_benchmark} test instances\n")

    def load_ablation_data(self) -> Dict[str, List[Dict]]:
        """
        Load all ablation JSON files.

        Returns:
            Dictionary mapping ablation names to list of heuristic entries
        """
        if not self.ablation_dir.exists():
            raise FileNotFoundError(f"Ablation directory not found: {self.ablation_dir}")

        ablations = {}
        json_files = list(self.ablation_dir.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.ablation_dir}")

        print(f"Loading ablation data from {len(json_files)} files...\n")

        for json_file in sorted(json_files):
            ablation_name = json_file.stem  # Filename without extension

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, dict):
                # Convert dict to list
                heuristics = list(data.values())
            else:
                heuristics = data

            ablations[ablation_name] = heuristics
            print(f"  {ablation_name}: {len(heuristics)} heuristics")

        return ablations

    def analyze_ablation(self, ablation_name: str, heuristics: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a single ablation version.

        Args:
            ablation_name: Name of the ablation
            heuristics: List of heuristic entries

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {ablation_name}")
        print(f"{'='*70}")

        total_generated = len(heuristics)
        success_rate = (total_generated / 100.0) * 100.0  # Assuming 100 attempts

        # Count valid heuristics (those with code)
        valid_heuristics = [h for h in heuristics if h.get('code') and h.get('score') is not None]
        num_valid = len(valid_heuristics)

        print(f"Total generated: {total_generated}")
        print(f"Valid heuristics: {num_valid} ({num_valid/total_generated*100:.1f}%)")
        print(f"Success rate: {success_rate:.1f}%")

        if num_valid == 0:
            print(f"[WARNING] No valid heuristics found in {ablation_name}")
            return {
                'ablation': ablation_name,
                'total_generated': total_generated,
                'num_valid': 0,
                'success_rate': success_rate,
                'best_gen_score': None,
                'benchmark_score': None,
                'benchmark_time': None,
                'status': 'no_valid_heuristics'
            }

        # Find best heuristic by generation score (higher is better for negative costs)
        # Scores are negative (tour lengths), so best = maximum (least negative)
        best_heuristic = max(valid_heuristics, key=lambda h: h['score'])
        best_gen_score = best_heuristic['score']

        print(f"\nBest heuristic from generation:")
        print(f"  ID: {best_heuristic.get('program_id', 'N/A')}")
        print(f"  Generation score: {best_gen_score:.6f}")
        print(f"  Iteration: {best_heuristic.get('iteration', 'N/A')}")

        # Evaluate on large benchmark set
        print(f"\nEvaluating best heuristic on benchmark ({self.n_benchmark} instances)...")
        start_time = time.time()

        try:
            benchmark_score = self.evaluator.evaluate_program(best_heuristic['code'])
            benchmark_time = time.time() - start_time

            if benchmark_score is None:
                print(f"[FAILED] Benchmark evaluation returned None")
                status = 'benchmark_failed'
            else:
                print(f"[SUCCESS] Benchmark score: {benchmark_score:.6f} (Time: {benchmark_time:.2f}s)")
                status = 'success'

        except Exception as e:
            benchmark_score = None
            benchmark_time = time.time() - start_time
            print(f"[ERROR] Benchmark evaluation failed: {str(e)}")
            status = 'benchmark_error'

        # Collect statistics on all generation scores
        gen_scores = [h['score'] for h in valid_heuristics]

        return {
            'ablation': ablation_name,
            'total_generated': total_generated,
            'num_valid': num_valid,
            'success_rate': success_rate,
            'best_gen_score': best_gen_score,
            'mean_gen_score': np.mean(gen_scores),
            'std_gen_score': np.std(gen_scores),
            'worst_gen_score': min(gen_scores),
            'benchmark_score': benchmark_score,
            'benchmark_time': benchmark_time,
            'best_program_id': best_heuristic.get('program_id', 'N/A'),
            'best_iteration': best_heuristic.get('iteration', 'N/A'),
            'status': status
        }

    def run_evaluation(self, save_results: bool = True) -> pd.DataFrame:
        """
        Run evaluation for all ablations.

        Args:
            save_results: Whether to save results to CSV

        Returns:
            DataFrame with evaluation results
        """
        ablations = self.load_ablation_data()
        results = []

        print(f"\n{'#'*70}")
        print(f"# TSP ABLATION EVALUATION")
        print(f"# Ablation versions: {len(ablations)}")
        print(f"# Benchmark instances: {self.n_benchmark}")
        print(f"# Problem size: {self.problem_size}")
        print(f"{'#'*70}\n")

        # Evaluate each ablation
        for ablation_name, heuristics in ablations.items():
            result = self.analyze_ablation(ablation_name, heuristics)
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Add rank based on benchmark score (higher is better for negative costs)
        df_success = df[df['status'] == 'success'].copy()
        if len(df_success) > 0:
            df_success['benchmark_rank'] = df_success['benchmark_score'].rank(ascending=False, method='min')
            df = df.merge(df_success[['ablation', 'benchmark_rank']], on='ablation', how='left')
        else:
            df['benchmark_rank'] = None

        # Print summary
        self._print_summary(df)

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"tsp_ablation_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")

            # Also save a detailed report
            report_path = self.output_dir / f"tsp_ablation_report_{timestamp}.txt"
            self._save_detailed_report(df, report_path)
            print(f"Detailed report saved to: {report_path}")

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"ABLATION EVALUATION SUMMARY")
        print(f"{'='*80}")

        # Overall statistics
        total = len(df)
        success = len(df[df['status'] == 'success'])
        failed = total - success

        print(f"\nTotal ablations: {total}")
        print(f"  Successful evaluations: {success} ({success/total*100:.1f}%)")
        print(f"  Failed evaluations: {failed} ({failed/total*100:.1f}%)")

        # Generation statistics
        print(f"\nGeneration Statistics:")
        print(f"  Total heuristics generated: {df['total_generated'].sum()}")
        print(f"  Valid heuristics: {df['num_valid'].sum()}")
        print(f"  Mean per ablation: {df['total_generated'].mean():.1f} ± {df['total_generated'].std():.1f}")

        # Benchmark performance (for successful evaluations)
        df_success = df[df['status'] == 'success']
        if len(df_success) > 0:
            print(f"\nBenchmark Performance (n={len(df_success)}):")
            print(f"  Best benchmark score:    {df_success['benchmark_score'].max():.6f}")
            print(f"  Worst benchmark score:   {df_success['benchmark_score'].min():.6f}")
            print(f"  Mean benchmark score:    {df_success['benchmark_score'].mean():.6f}")
            print(f"  Std benchmark score:     {df_success['benchmark_score'].std():.6f}")
            print(f"  Mean evaluation time:    {df_success['benchmark_time'].mean():.2f}s")

            # Detailed ranking table
            print(f"\n{'='*80}")
            print(f"ABLATION RANKING (by benchmark score)")
            print(f"{'='*80}")
            print(f"{'Rank':<6} {'Ablation':<30} {'#Gen':<8} {'Success%':<10} {'GenScore':<12} {'BenchScore':<12} {'Time(s)':<10}")
            print("-" * 80)

            df_sorted = df_success.sort_values('benchmark_score', ascending=False)
            for idx, row in df_sorted.iterrows():
                print(f"{int(row['benchmark_rank']):<6} "
                      f"{row['ablation']:<30} "
                      f"{row['total_generated']:<8} "
                      f"{row['success_rate']:<10.1f} "
                      f"{row['best_gen_score']:<12.6f} "
                      f"{row['benchmark_score']:<12.6f} "
                      f"{row['benchmark_time']:<10.2f}")

        # Failed ablations
        df_failed = df[df['status'] != 'success']
        if len(df_failed) > 0:
            print(f"\nFailed Ablations ({len(df_failed)}):")
            for idx, row in df_failed.iterrows():
                print(f"  - {row['ablation']}: {row['status']}")

        print(f"\n{'='*80}\n")

    def _save_detailed_report(self, df: pd.DataFrame, report_path: Path):
        """Save detailed text report."""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TSP ABLATION EVALUATION - DETAILED REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Benchmark instances: {self.n_benchmark}\n")
            f.write(f"Problem size: {self.problem_size}\n")
            f.write(f"Total ablations: {len(df)}\n\n")

            f.write("="*80 + "\n")
            f.write("RESULTS BY ABLATION\n")
            f.write("="*80 + "\n\n")

            for idx, row in df.iterrows():
                f.write(f"Ablation: {row['ablation']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Generated heuristics:    {row['total_generated']}\n")
                f.write(f"  Valid heuristics:        {row['num_valid']}\n")
                f.write(f"  Success rate:            {row['success_rate']:.1f}%\n")

                # Format values with None checks
                best_gen_str = f"{row['best_gen_score']:.6f}" if row['best_gen_score'] is not None else 'N/A'
                mean_gen_str = f"{row['mean_gen_score']:.6f}" if pd.notna(row['mean_gen_score']) else 'N/A'
                benchmark_score_str = f"{row['benchmark_score']:.6f}" if row['benchmark_score'] is not None else 'N/A'
                benchmark_time_str = f"{row['benchmark_time']:.2f}s" if row['benchmark_time'] is not None else 'N/A'
                benchmark_rank_str = f"{int(row['benchmark_rank'])}" if pd.notna(row['benchmark_rank']) else 'N/A'

                f.write(f"  Best generation score:   {best_gen_str}\n")
                f.write(f"  Mean generation score:   {mean_gen_str}\n")
                f.write(f"  Benchmark score:         {benchmark_score_str}\n")
                f.write(f"  Benchmark time:          {benchmark_time_str}\n")
                f.write(f"  Benchmark rank:          {benchmark_rank_str}\n")
                f.write(f"  Status:                  {row['status']}\n")
                f.write("\n")

            # Summary statistics
            df_success = df[df['status'] == 'success']
            if len(df_success) > 0:
                f.write("="*80 + "\n")
                f.write("SUMMARY STATISTICS\n")
                f.write("="*80 + "\n\n")
                f.write(f"Successful ablations: {len(df_success)}/{len(df)}\n\n")
                f.write(f"Total heuristics generated: {df['total_generated'].sum()}\n")
                f.write(f"Total valid heuristics: {df['num_valid'].sum()}\n\n")
                f.write(f"Benchmark scores:\n")
                f.write(f"  Best:   {df_success['benchmark_score'].max():.6f}\n")
                f.write(f"  Worst:  {df_success['benchmark_score'].min():.6f}\n")
                f.write(f"  Mean:   {df_success['benchmark_score'].mean():.6f}\n")
                f.write(f"  Median: {df_success['benchmark_score'].median():.6f}\n")
                f.write(f"  Std:    {df_success['benchmark_score'].std():.6f}\n\n")
                f.write(f"Average evaluation time: {df_success['benchmark_time'].mean():.2f}s\n")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Evaluate TSP ablation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--ablation_dir',
        type=str,
        default='tsp_ablation',
        help='Directory containing ablation JSON files'
    )

    parser.add_argument(
        '--n_benchmark',
        type=int,
        default=100,
        help='Number of test instances for benchmark evaluation'
    )

    parser.add_argument(
        '--problem_size',
        type=int,
        default=50,
        help='TSP problem size for benchmark'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout in seconds for each evaluation'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_results',
        help='Directory to save results'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = TSPAblationEvaluator(
        ablation_dir=args.ablation_dir,
        n_benchmark=args.n_benchmark,
        problem_size=args.problem_size,
        timeout_seconds=args.timeout,
        output_dir=args.output_dir
    )

    df_results = evaluator.run_evaluation(save_results=True)

    return df_results


if __name__ == '__main__':
    main()
