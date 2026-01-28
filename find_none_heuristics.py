"""
Find heuristics with None scores for each task.

For each task:
1. Load all heuristics from task/{task_name}/heuristics.json
2. Evaluate them in parallel
3. Print names of heuristics that return None score
"""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from base.evaluate import SecureEvaluator


# All available tasks and their evaluators
TASK_EVALUATORS = {
    'tsp_construct': ('task.tsp_construct.evaluation', 'TSPEvaluation'),
    'cvrp_construct': ('task.cvrp_construct.evaluation', 'CVRPEvaluation'),
    'vrptw_construct': ('task.vrptw_construct.evaluation', 'VRPTWEvaluation'),
    'jssp_construct': ('task.jssp_construct.evaluation', 'JSSPEvaluation'),
    'knapsack_construct': ('task.knapsack_construct.evaluation', 'KnapsackEvaluation'),
    'online_bin_packing': ('task.online_bin_packing.evaluation', 'OBPEvaluation'),
    'qap_construct': ('task.qap_construct.evaluation', 'QAPEvaluation'),
    'set_cover_construct': ('task.set_cover_construct.evaluation', 'SetCoverEvaluation'),
    'cflp_construct': ('task.cflp_construct.evaluation', 'CFLPEvaluation'),
    'admissible_set': ('task.admissible_set.evaluation', 'ASPEvaluation'),
}


def load_evaluator(task_name: str):
    """Load evaluator for a specific task."""
    if task_name not in TASK_EVALUATORS:
        raise ValueError(f"Unknown task: {task_name}")

    module_path, class_name = TASK_EVALUATORS[task_name]
    module = __import__(module_path, fromlist=[class_name])
    EvaluatorClass = getattr(module, class_name)
    return EvaluatorClass()


def load_heuristics(task_name: str) -> dict:
    """Load heuristics from task/{task_name}/heuristics.json."""
    heuristics_path = Path(f"task/{task_name}/heuristics.json")

    if not heuristics_path.exists():
        print(f"  Warning: {heuristics_path} not found")
        return {}

    with open(heuristics_path, 'r') as f:
        return json.load(f)


def evaluate_task_heuristics(task_name: str, num_workers: int = 4):
    """
    Evaluate all heuristics for a task and return those with None scores.

    Args:
        task_name: Name of the task
        num_workers: Number of parallel workers

    Returns:
        List of heuristic names with None scores
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    # Load heuristics
    heuristics = load_heuristics(task_name)
    if not heuristics:
        print(f"  No heuristics found for {task_name}")
        return []

    print(f"  Loaded {len(heuristics)} heuristics")

    # Load evaluator
    try:
        evaluator = load_evaluator(task_name)
        secure_eval = SecureEvaluator(evaluator, debug_mode=False)
        print(f"  Evaluator loaded (timeout: {evaluator.timeout_seconds}s)")
    except Exception as e:
        print(f"  Error loading evaluator: {e}")
        return []

    # Evaluation function for parallel execution
    def eval_heuristic(name_code):
        name, code = name_code
        try:
            score = secure_eval.evaluate_program(code)
            return name, score, None
        except Exception as e:
            return name, None, str(e)

    # Evaluate in parallel
    results = []
    none_scores = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_heuristic, (name, code)): name
                   for name, code in heuristics.items()}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"  Evaluating {task_name}"):
            name, score, error = future.result()
            results.append((name, score, error))

            if score is None:
                none_scores.append((name, error))

    # Summary
    valid_count = sum(1 for _, score, _ in results if score is not None)
    none_count = len(none_scores)

    print(f"\n  Results: {valid_count} valid, {none_count} with None score")

    if none_scores:
        print(f"\n  Heuristics with None score:")
        print(f"  {'-'*50}")
        for name, error in none_scores:
            if error:
                print(f"    - {name}: {error[:80]}...")
            else:
                print(f"    - {name}")

    return [name for name, _ in none_scores]


def main():
    """Evaluate all tasks and find heuristics with None scores."""
    print("="*60)
    print("Finding Heuristics with None Scores")
    print("="*60)

    num_workers = 4  # Adjust based on your system

    all_none_heuristics = {}

    for task_name in TASK_EVALUATORS.keys():
        try:
            none_names = evaluate_task_heuristics(task_name, num_workers=num_workers)
            if none_names:
                all_none_heuristics[task_name] = none_names
        except Exception as e:
            print(f"  Error processing {task_name}: {e}")
            continue

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    if all_none_heuristics:
        total_none = sum(len(names) for names in all_none_heuristics.values())
        print(f"\nTotal heuristics with None score: {total_none}")
        print()

        for task_name, names in all_none_heuristics.items():
            print(f"{task_name} ({len(names)} heuristics with None):")
            for name in names:
                print(f"  - {name}")
            print()
    else:
        print("\nNo heuristics with None score found in any task!")

    return all_none_heuristics


if __name__ == "__main__":
    main()
