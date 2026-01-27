"""
Benchmark Methods Evaluation Script

This script evaluates multiple heuristic methods (including ours) on a specified task
and reports performance metrics.

Usage:
    python benchmark_methods.py --task tsp_construct --methods_json path/to/methods.json
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import time

# Import base evaluator
from base.evaluate import SecureEvaluator

# Import task evaluators (actual class names from the code)
from task.tsp_construct.evaluation import TSPEvaluation
from task.cvrp_construct.evaluation import CVRPEvaluation
from task.knapsack_construct.evaluation import KnapsackEvaluation
from task.jssp_construct.evaluation import JSSPEvaluation
from task.vrptw_construct.evaluation import VRPTWEvaluation
from task.qap_construct.evaluation import QAPEvaluation
from task.set_cover_construct.evaluation import SCPEvaluation
from task.cflp_construct.evaluation import CFLPEvaluation
from task.online_bin_packing.evaluation import OBPEvaluation

# Task configurations
TASK_CONFIG = {
    'tsp_construct': {
        'evaluator': TSPEvaluation,
        'params': {'n_instance': 100, 'problem_size': 50, 'timeout_seconds': 30}
    },
    'cvrp_construct': {
        'evaluator': CVRPEvaluation,
        'params': {'n_instance': 100, 'problem_size': 50, 'timeout_seconds': 30}
    },
    'knapsack_construct': {
        'evaluator': KnapsackEvaluation,
        'params': {'n_instance': 100, 'n_items': 50, 'knapsack_capacity': 500, 'timeout_seconds': 20}
    },
    'jssp_construct': {
        'evaluator': JSSPEvaluation,
        'params': {'n_instance': 100, 'n_jobs': 10, 'n_machines': 10, 'timeout_seconds': 30}
    },
    'vrptw_construct': {
        'evaluator': VRPTWEvaluation,
        'params': {'n_instance': 100, 'problem_size': 50, 'timeout_seconds': 60}
    },
    'qap_construct': {
        'evaluator': QAPEvaluation,
        'params': {'n_instance': 100, 'n_facilities': 20, 'timeout_seconds': 30}
    },
    'set_cover_construct': {
        'evaluator': SCPEvaluation,
        'params': {'n_instance': 100, 'n_elements': 50, 'n_subsets': 30, 'max_subset_size': 10, 'timeout_seconds': 20}
    },
    'online_bin_packing': {
        'evaluator': OBPEvaluation,
        'params': {'n_instances': 100, 'n_items': 5000, 'capacity': 100, 'timeout_seconds': 30}
    },
    'cflp_construct': {
        'evaluator': CFLPEvaluation,
        'params': {'n_instance': 100, 'n_facilities': 20, 'n_customers': 50, 'timeout_seconds': 30}
    }
}


class BenchmarkRunner:
    """Run benchmarks for multiple heuristic methods on a specified task."""

    def __init__(self, task_name: str, methods_json_path: str, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark runner.

        Args:
            task_name: Name of the task (e.g., 'tsp_construct')
            methods_json_path: Path to JSON file containing methods
            output_dir: Directory to save results
        """
        self.task_name = task_name
        self.methods_json_path = Path(methods_json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Validate task
        if task_name not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_CONFIG.keys())}")

        # Initialize evaluator
        config = TASK_CONFIG[task_name]
        base_evaluator = config['evaluator'](**config['params'])

        # Wrap with SecureEvaluator for proper evaluation
        self.evaluator = SecureEvaluator(base_evaluator, debug_mode=False)

        # Get number of instances (different tasks use different param names)
        self.n_instance = config['params'].get('n_instance', config['params'].get('n_instances', 'N/A'))

        print(f"Initialized {task_name} evaluator with {config['params']}")

    def load_methods(self) -> Dict[str, str]:
        """
        Load methods from JSON file.

        Returns:
            Dictionary mapping method names to code strings
        """
        if not self.methods_json_path.exists():
            raise FileNotFoundError(f"Methods file not found: {self.methods_json_path}")

        with open(self.methods_json_path, 'r', encoding='utf-8') as f:
            methods = json.load(f)

        print(f"Loaded {len(methods)} methods from {self.methods_json_path}")
        return methods

    def evaluate_method(self, method_name: str, code: str) -> Dict[str, Any]:
        """
        Evaluate a single method.

        Args:
            method_name: Name of the method
            code: Python code string

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Evaluate the method using SecureEvaluator
            score = self.evaluator.evaluate_program(code)
            eval_time = time.time() - start_time

            if score is None:
                print(f"[FAILED] {method_name}: Evaluation returned None (timeout or error)")
                return {
                    'method': method_name,
                    'score': None,
                    'status': 'failed',
                    'eval_time': eval_time,
                    'error': 'Evaluation returned None'
                }

            print(f"[SUCCESS] {method_name}: Score = {score:.6f} (Time: {eval_time:.2f}s)")
            return {
                'method': method_name,
                'score': score,
                'status': 'success',
                'eval_time': eval_time,
                'error': None
            }

        except Exception as e:
            eval_time = time.time() - start_time
            print(f"[ERROR] {method_name}: {str(e)}")
            return {
                'method': method_name,
                'score': None,
                'status': 'error',
                'eval_time': eval_time,
                'error': str(e)
            }

    def run_benchmark(self, save_results: bool = True) -> pd.DataFrame:
        """
        Run benchmark for all methods.

        Args:
            save_results: Whether to save results to CSV

        Returns:
            DataFrame with benchmark results
        """
        methods = self.load_methods()
        results = []

        print(f"\n{'#'*60}")
        print(f"# BENCHMARK: {self.task_name}")
        print(f"# Methods to evaluate: {len(methods)}")
        print(f"# Test instances: {self.n_instance}")
        print(f"{'#'*60}\n")

        # Evaluate each method
        for method_name, code in methods.items():
            result = self.evaluate_method(method_name, code)
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Add rank (lower score is better for most tasks)
        df_success = df[df['status'] == 'success'].copy()
        if len(df_success) > 0:
            df_success['rank'] = df_success['score'].rank(method='min')
            df = df.merge(df_success[['method', 'rank']], on='method', how='left')
        else:
            df['rank'] = None

        # Print summary
        self._print_summary(df)

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"{self.task_name}_benchmark_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print benchmark summary."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY: {self.task_name}")
        print(f"{'='*80}")

        # Overall statistics
        total = len(df)
        success = len(df[df['status'] == 'success'])
        failed = len(df[df['status'] == 'failed'])
        error = len(df[df['status'] == 'error'])

        print(f"\nTotal methods: {total}")
        print(f"  Success: {success} ({success/total*100:.1f}%)")
        print(f"  Failed:  {failed} ({failed/total*100:.1f}%)")
        print(f"  Error:   {error} ({error/total*100:.1f}%)")

        # Score statistics (for successful methods)
        df_success = df[df['status'] == 'success']
        if len(df_success) > 0:
            scores = df_success['score'].values
            print(f"\nScore Statistics (n={len(df_success)}):")
            print(f"  Best:    {np.min(scores):.6f}")
            print(f"  Worst:   {np.max(scores):.6f}")
            print(f"  Mean:    {np.mean(scores):.6f}")
            print(f"  Median:  {np.median(scores):.6f}")
            print(f"  Std:     {np.std(scores):.6f}")

            # Top 5 methods
            print(f"\nTop 5 Methods:")
            print(f"{'Rank':<6} {'Method':<40} {'Score':<15} {'Time (s)':<10}")
            print("-" * 80)
            top5 = df_success.nsmallest(5, 'score')
            for idx, row in top5.iterrows():
                print(f"{int(row['rank']):<6} {row['method']:<40} {row['score']:<15.6f} {row['eval_time']:<10.2f}")

            # Bottom 5 methods
            if len(df_success) > 5:
                print(f"\nBottom 5 Methods:")
                print(f"{'Rank':<6} {'Method':<40} {'Score':<15} {'Time (s)':<10}")
                print("-" * 80)
                bottom5 = df_success.nlargest(5, 'score')
                for idx, row in bottom5.iterrows():
                    print(f"{int(row['rank']):<6} {row['method']:<40} {row['score']:<15.6f} {row['eval_time']:<10.2f}")

        # Failed/Error methods
        df_failed = df[df['status'].isin(['failed', 'error'])]
        if len(df_failed) > 0:
            print(f"\nFailed/Error Methods ({len(df_failed)}):")
            for idx, row in df_failed.iterrows():
                print(f"  - {row['method']}: {row['status']} ({row['error'][:50] if row['error'] else 'N/A'}...)")

        print(f"\n{'='*80}\n")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple heuristic methods on a specified task"
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=list(TASK_CONFIG.keys()),
        help='Task name (e.g., tsp_construct)'
    )

    parser.add_argument(
        '--methods_json',
        type=str,
        required=True,
        help='Path to JSON file containing methods'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='benchmark_results',
        help='Directory to save results (default: benchmark_results)'
    )

    parser.add_argument(
        '--n_instance',
        type=int,
        default=None,
        help='Override number of test instances (default: use task config)'
    )

    args = parser.parse_args()

    # Override n_instance if specified
    if args.n_instance is not None:
        TASK_CONFIG[args.task]['params']['n_instance'] = args.n_instance

    # Run benchmark
    runner = BenchmarkRunner(
        task_name=args.task,
        methods_json_path=args.methods_json,
        output_dir=args.output_dir
    )

    df_results = runner.run_benchmark(save_results=True)

    return df_results


if __name__ == '__main__':
    # Example usage without CLI:
    # Uncomment and modify these lines to run directly

    # TASK = 'tsp_construct'
    # METHODS_JSON = 'task/tsp_construct/heuristics.json'
    #
    # runner = BenchmarkRunner(
    #     task_name=TASK,
    #     methods_json_path=METHODS_JSON,
    #     output_dir='benchmark_results'
    # )
    #
    # df = runner.run_benchmark(save_results=True)

    # Or use CLI:
    main()
