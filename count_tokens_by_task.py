"""
Count tokens for best_found and augmented heuristics for each task.

This script:
1. Loads best_found JSON files for each task (TSP, CVRP, Knapsack, Online Bin Packing)
2. For each task:
   - Loads the decoder model and tokenizer
   - Counts tokens for each method in best_found (after removing comments)
   - Counts tokens for augmented heuristics (after removing comments)
   - Prints average token counts
"""

import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from model_config import DEFAULT_DECODER


def remove_comments_and_docstrings(code: str) -> str:
    """
    Remove comments and docstrings from Python code using AST.

    Args:
        code: Python source code string

    Returns:
        Code with comments and docstrings removed
    """
    try:
        tree = ast.parse(code)

        # Remove docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (ast.get_docstring(node) is not None and
                    node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant)):
                    node.body.pop(0)

        # Convert back to source code
        code_no_docstrings = ast.unparse(tree)

        # Remove single-line comments using regex
        lines = code_no_docstrings.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove inline comments but preserve strings
            # This is a simple approach - doesn't handle all edge cases
            if '#' in line:
                # Check if # is inside a string
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                    elif char == '#' and not in_string:
                        line = line[:i].rstrip()
                        break
            if line.strip():  # Only add non-empty lines
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    except SyntaxError:
        # If parsing fails, try simple regex-based removal
        # Remove docstrings (triple quotes)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        # Remove comments
        lines = code.split('\n')
        cleaned_lines = [line.split('#')[0].rstrip() for line in lines]
        cleaned_lines = [line for line in cleaned_lines if line.strip()]

        return '\n'.join(cleaned_lines)


def count_tokens(code: str, tokenizer) -> int:
    """
    Count the number of tokens in code using the tokenizer.

    Args:
        code: Python source code
        tokenizer: HuggingFace tokenizer

    Returns:
        Number of tokens
    """
    tokens = tokenizer.encode(code, add_special_tokens=False)
    return len(tokens)


def process_task(task_name: str,
                 best_found_file: str,
                 augmented_file: str,
                 tokenizer) -> Dict:
    """
    Process a single task: count tokens for best_found and augmented heuristics.

    Args:
        task_name: Name of the task (e.g., "TSP")
        best_found_file: Path to best_found JSON file
        augmented_file: Path to augmented JSON file
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*70}")
    print(f"Processing: {task_name}")
    print(f"{'='*70}")

    stats = {
        'task': task_name,
        'best_found': {},
        'augmented': {}
    }

    # Process best_found heuristics
    if Path(best_found_file).exists():
        with open(best_found_file, 'r', encoding='utf-8') as f:
            best_found_data = json.load(f)

        print(f"\nBest Found Heuristics ({len(best_found_data)} methods):")
        print("-" * 70)

        method_tokens = []
        for method_name, code in best_found_data.items():
            # Remove comments and docstrings
            clean_code = remove_comments_and_docstrings(code)

            # Count tokens
            num_tokens = count_tokens(clean_code, tokenizer)
            method_tokens.append(num_tokens)

            print(f"  {method_name:<25} {num_tokens:>6} tokens")

        avg_tokens = sum(method_tokens) / len(method_tokens) if method_tokens else 0
        print(f"\n  {'Average:':<25} {avg_tokens:>6.1f} tokens")

        stats['best_found'] = {
            'count': len(method_tokens),
            'tokens_per_method': method_tokens,
            'average': avg_tokens,
            'min': min(method_tokens) if method_tokens else 0,
            'max': max(method_tokens) if method_tokens else 0
        }
    else:
        print(f"  ⚠️  Best found file not found: {best_found_file}")

    # Process augmented heuristics
    if Path(augmented_file).exists():
        with open(augmented_file, 'r', encoding='utf-8') as f:
            augmented_data = json.load(f)

        print(f"\nAugmented Heuristics ({len(augmented_data)} heuristics):")
        print("-" * 70)

        augmented_tokens = []
        for heur_name, code in augmented_data.items():
            # Remove comments and docstrings
            clean_code = remove_comments_and_docstrings(code)

            # Count tokens
            num_tokens = count_tokens(clean_code, tokenizer)
            augmented_tokens.append(num_tokens)

        avg_tokens = sum(augmented_tokens) / len(augmented_tokens) if augmented_tokens else 0
        print(f"  {'Average:':<25} {avg_tokens:>6.1f} tokens")
        print(f"  {'Min:':<25} {min(augmented_tokens) if augmented_tokens else 0:>6} tokens")
        print(f"  {'Max:':<25} {max(augmented_tokens) if augmented_tokens else 0:>6} tokens")

        stats['augmented'] = {
            'count': len(augmented_tokens),
            'tokens_per_heuristic': augmented_tokens,
            'average': avg_tokens,
            'min': min(augmented_tokens) if augmented_tokens else 0,
            'max': max(augmented_tokens) if augmented_tokens else 0
        }
    else:
        print(f"  ⚠️  Augmented file not found: {augmented_file}")

    return stats


def main():
    """Main execution function."""

    # Task configuration
    tasks = [
        {
            'name': 'TSP',
            'best_found': 'best_found/tsp.json',
            'augmented': 'task/tsp_construct/augmented.json'
        },
        {
            'name': 'CVRP',
            'best_found': 'best_found/cvrp.json',
            'augmented': 'task/cvrp_construct/augmented.json'
        },
        {
            'name': 'Knapsack',
            'best_found': 'best_found/knapsack.json',
            'augmented': 'task/knapsack_construct/augmented.json'
        },
        {
            'name': 'Online Bin Packing',
            'best_found': 'best_found/op.json',
            'augmented': 'task/online_bin_packing/augmented.json'
        }
    ]

    print("="*70)
    print("Token Counting for Best Found and Augmented Heuristics")
    print("="*70)
    print(f"\nDecoder Model: {DEFAULT_DECODER}")
    print("\nLoading tokenizer...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_DECODER,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded successfully")

    # Process each task
    all_stats = []
    for task_config in tasks:
        stats = process_task(
            task_name=task_config['name'],
            best_found_file=task_config['best_found'],
            augmented_file=task_config['augmented'],
            tokenizer=tokenizer
        )
        all_stats.append(stats)

    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"\n{'Task':<25} {'Best Found Avg':<20} {'Augmented Avg':<20}")
    print("-" * 70)

    for stats in all_stats:
        best_avg = stats['best_found'].get('average', 0)
        aug_avg = stats['augmented'].get('average', 0)
        print(f"{stats['task']:<25} {best_avg:>8.1f} tokens      {aug_avg:>8.1f} tokens")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
