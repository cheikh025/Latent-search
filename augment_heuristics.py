"""
Data Augmentation Script for Heuristic Code
Implements the 3-tier augmentation strategy:
1. Syntactic & Structural Rewriting
2. Parameter & Hyperparameter Tuning
3. Semantic-Preserving Behavioral Diversity
"""

import json
import re
import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import litellm
from dotenv import load_dotenv

# Import existing utilities from the project
from utils import is_valid_python, extract_python_code_robust
from programDB import are_codes_structurally_same
from base.code import TextFunctionProgramConverter

# Load environment variables from .env file
load_dotenv()

# Suppress litellm logging
litellm.suppress_debug_info = True

# API key mapping
API_KEY_MAP = {
    "claude": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "o1": "OPENAI_API_KEY",
    "groq":"GROQ_API_KEY",
}


def check_api_key(model: str) -> bool:
    """Check if required API key exists in environment."""
    for provider, env_var in API_KEY_MAP.items():
        if provider in model.lower():
            if not os.getenv(env_var):
                print(f"Error: {env_var} not found. Set it in .env file or environment.")
                return False
            return True
    return True


def discover_tasks(base_dir: str = "task") -> List[Tuple[str, Path, Path]]:
    """
    Step 1: Discover all task directories with heuristics.json files.

    Returns:
        List of tuples: (task_name, heuristics_path, template_path)
    """
    tasks = []
    base_path = Path(base_dir)

    for heuristics_file in base_path.glob("*/heuristics.json"):
        task_dir = heuristics_file.parent
        task_name = task_dir.name
        template_file = task_dir / "template.py"

        if template_file.exists():
            tasks.append((task_name, heuristics_file, template_file))
        else:
            print(f"Warning: No template.py found for {task_name}, skipping...")

    return tasks


def extract_function_name(template_path: Path) -> str:
    """
    Extract the main function name from template.py using existing utilities.

    Args:
        template_path: Path to template.py file

    Returns:
        Function name as string, or None if extraction fails
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract template_program string
    match = re.search(r'template_program\s*=\s*[\'\"]{3}(.*?)[\'\"]{3}', content, re.DOTALL)
    if match:
        template_code = match.group(1)
        # Use existing utility to parse the program
        program = TextFunctionProgramConverter.text_to_program(template_code)
        if program and program.functions:
            return program.functions[0].name

    # If template parsing fails, return None - we'll extract from generated code instead
    return None


def check_for_runtime_issues(code: str) -> bool:
    """
    Check for potential runtime warnings in generated code.

    Args:
        code: Python code string

    Returns:
        True if no issues found, False if potential issues detected
    """
    # Check for unprotected divisions
    # Look for patterns like: / np.mean(...) or / np.sum(...) without epsilon
    dangerous_patterns = [
        r'/\s*np\.(mean|sum|std|max|min|median)\([^)]+\)(?!\s*[+\-]\s*[\d.e-]+)',  # division without epsilon
        r'/\s*\([^)]*np\.(mean|sum|std|max|min|median)[^)]*\)(?!\s*[+\-]\s*[\d.e-]+)',  # division in parens
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            # Check if there's a safety epsilon nearby
            return False

    return True


def validate_generated_code(code: str, required_function: str = None) -> bool:
    """
    Step 4: Validate Python code syntax and structure.
    Uses existing utilities from the project.

    Args:
        code: Python code string
        required_function: Expected function name (optional) - if None, just checks that at least one function exists

    Returns:
        True if valid, False otherwise
    """
    # First check if it's valid Python using existing utility
    if not is_valid_python(code):
        return False

    # Check that code contains at least one function using existing utility
    program = TextFunctionProgramConverter.text_to_program(code)
    if program is None or not program.functions:
        return False

    # If a required function is specified, verify it exists
    # Otherwise, accept any function (we'll extract the name dynamically later, like programDB does)
    if required_function:
        function_names = [f.name for f in program.functions]
        if required_function not in function_names:
            return False

    # Check for potential runtime issues
    if not check_for_runtime_issues(code):
        return False

    return True


def generate_augmentation_prompt(original_code: str, task_name: str, num_variations: int) -> str:
    """
    Step 3: Create prompt for LLM to generate code variations.

    Args:
        original_code: Original heuristic code
        task_name: Name of the task for context
        num_variations: Number of variations to generate

    Returns:
        Formatted prompt string
    """
    # Mix of all strategies for diversity
    strategy_instructions = """Apply a MIX of these augmentation strategies to create diverse variations:

1. SYNTACTIC & STRUCTURAL REWRITING:
   - Rename variables (e.g., distance_matrix → dist_mat, unvisited_nodes → candidates)
   - Transform control flow (for ↔ while, list comprehensions, vectorized numpy)
   - Use mathematical synonyms (x*x vs x**2 vs np.square(x))
   - Invert conditional logic where appropriate

2. PARAMETER & HYPERPARAMETER TUNING:
   - Modify weights in scoring functions (e.g., 0.8*dist + 0.2*cost → 0.6*dist + 0.4*cost)
   - Adjust threshold values (e.g., top_k = 5 → top_k = 7)
   - Tune constants (e.g., epsilon values, scaling factors)

3. SEMANTIC-PRESERVING BEHAVIORAL DIVERSITY:
   - Add small deterministic noise terms for tie-breaking
   - Change aggregation methods (np.mean ↔ np.median, np.sum ↔ np.max)
   - Replace greedy argmin with softmin or random choice among top-k
   - Use proxy approximations for calculations

Each variation should use a DIFFERENT combination of these strategies."""

    # Generate variation placeholders
    variation_format = "\n\n".join([
        f"**Variation {i}:**\n```python\n# complete code here\n```"
        for i in range(1, num_variations + 1)
    ])

    prompt = f"""You are an expert Python algorithm developer specializing in combinatorial optimization.

TASK: {task_name}

ORIGINAL HEURISTIC CODE:
```python
{original_code}
```

INSTRUCTIONS:
{strategy_instructions}

REQUIREMENTS:
- Generate EXACTLY {num_variations} distinct variations of the code
- MUST preserve the function signature exactly as shown
- MUST return valid, executable Python code
- MUST use only numpy (no external libraries beyond numpy)
- Each variation should be meaningfully different from the others
- Apply DIFFERENT augmentation strategies to each variation for maximum diversity
- Maintain code quality and readability
- CRITICAL: Always add epsilon (e.g., + 1e-12) to denominators to prevent division by zero
- CRITICAL: Use np.clip() when values must be bounded to prevent invalid operations

OUTPUT FORMAT:
Return {num_variations} variations in this exact format:

{variation_format}

Each variation must be complete, runnable Python code with imports.
"""

    return prompt


def generate_variations(
    original_code: str,
    task_name: str,
    num_variations: int,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7
) -> List[str]:
    """
    Step 3: Use LLM to generate code variations.

    Args:
        original_code: Original heuristic code
        task_name: Name of the task
        num_variations: Number of variations to generate
        model: LLM model to use
        temperature: Sampling temperature

    Returns:
        List of generated code variations
    """
    prompt = generate_augmentation_prompt(original_code, task_name, num_variations)

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=4000 + (num_variations * 500)  # Scale tokens with variations
        )

        content = response.choices[0].message.content
        #print(content)  # For debugging
        # Extract code blocks using regex - look for python code blocks
        # Pattern matches: ```python ... ``` or just ``` ... ```
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n\s*```', content, re.DOTALL)

        if code_blocks:
            return code_blocks
        else:
            # Fallback: try to extract code after "Variation N:" markers
            variation_sections = re.split(r'\*\*Variation \d+:?\*\*', content)[1:]
            all_variations = []
            for section in variation_sections:
                # Look for code patterns (def ... or import ...)
                code_match = re.search(r'((?:import.*?\n)*def\s+\w+.*?)(?=\*\*Variation|\Z)', section, re.DOTALL)
                if code_match:
                    all_variations.append(code_match.group(1).strip())
            return all_variations

    except Exception as e:
        if "api" in str(e).lower() and "key" in str(e).lower():
            print(f"API Key Error: {e}")
            raise
        print(f"Error generating variations: {e}")
        return []


def augment_heuristics(
    task_name: str,
    heuristics_path: Path,
    template_path: Path,
    model: str,
    temperature: float,
    target_size: int,
    output_suffix: str = "augmented"
) -> Dict[str, int]:
    """
    Main augmentation pipeline for a single task.

    Args:
        task_name: Name of the task
        heuristics_path: Path to heuristics.json
        template_path: Path to template.py
        model: LLM model to use
        temperature: Sampling temperature
        target_size: Target number of heuristics
        output_suffix: Suffix for output file

    Returns:
        Statistics dictionary
    """
    print(f"\nAugmenting: {task_name}")

    # Load original heuristics
    with open(heuristics_path, 'r', encoding='utf-8') as f:
        original_heuristics = json.load(f)

    original_count = len(original_heuristics)
    required_function = extract_function_name(template_path)
    variations_needed = max(0, target_size - original_count)

    print(f"Original: {original_count} | Target: {target_size} | Need: {variations_needed}")

    if variations_needed <= 0:
        return {"original": original_count, "generated": 0, "valid": 0, "final": original_count}

    # Augment each original heuristic - one query per heuristic
    augmented_heuristics = original_heuristics.copy()
    generated_count = 0
    valid_count = 0

    original_items = list(original_heuristics.items())

    # Calculate how many variations each heuristic should contribute
    base_variations_per_heuristic = variations_needed // original_count
    extra_variations_needed = variations_needed % original_count

    # Assign variation counts: some get base+1, others get base
    variation_assignments = []
    for i, (name, code) in enumerate(original_items):
        if i < extra_variations_needed:
            variation_assignments.append((name, code, base_variations_per_heuristic + 1))
        else:
            variation_assignments.append((name, code, base_variations_per_heuristic))

    print(f"Variation distribution: {base_variations_per_heuristic}-{base_variations_per_heuristic + 1} per heuristic")

    with tqdm(total=len(variation_assignments), desc="Processing heuristics") as pbar:
        for orig_name, orig_code, num_to_generate in variation_assignments:
            if num_to_generate == 0:
                pbar.update(1)
                continue

            # Generate ALL variations for this heuristic in ONE query
            variations = generate_variations(
                orig_code,
                task_name,
                num_variations=num_to_generate,
                model=model,
                temperature=temperature
            )

            # Validate and add variations
            added_for_this_heuristic = 0
            for var_code in variations:
                if added_for_this_heuristic >= num_to_generate:
                    break

                # Clean the code using existing utility
                try:
                    clean_code = extract_python_code_robust(var_code, include_preface=True)
                except:
                    # If extraction fails, try the raw code
                    clean_code = var_code

                # Validate using existing utility
                if validate_generated_code(clean_code, required_function):
                    # Check for structural duplicates using existing utility
                    is_duplicate = False
                    for existing_code in augmented_heuristics.values():
                        if are_codes_structurally_same(clean_code, existing_code):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        # Create unique name based on original heuristic
                        var_name = f"{orig_name}_aug_{generated_count}"
                        augmented_heuristics[var_name] = clean_code
                        valid_count += 1
                        generated_count += 1
                        added_for_this_heuristic += 1

            # Report if we didn't get enough valid variations
            if added_for_this_heuristic < num_to_generate:
                print(f"\n  Warning: {orig_name} only produced {added_for_this_heuristic}/{num_to_generate} valid variations")

            pbar.update(1)

    # Save augmented heuristics
    output_path = heuristics_path.parent / f"{output_suffix}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_heuristics, f, indent=2)

    success_rate = (generated_count / variations_needed * 100) if variations_needed > 0 else 0
    print(f"\nDone: {len(augmented_heuristics)} total | {generated_count}/{variations_needed} added ({success_rate:.1f}%) | Saved: {output_path.name}")

    return {
        "original": original_count,
        "generated": generated_count,
        "valid": valid_count,
        "final": len(augmented_heuristics),
        "target": target_size,
        "success_rate": success_rate
    }


def main():
    parser = argparse.ArgumentParser(
        description="Augment heuristic code datasets using LLMs"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to augment (e.g., 'tsp_construct'). If not specified, user will be prompted."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use (default: claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=500,
        help="Target number of heuristics (default: 500)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="augmented",
        help="Output file suffix (default: 'augmented')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Augment all tasks"
    )

    args = parser.parse_args()

    # Check API key
    if not check_api_key(args.model):
        return

    # Discover tasks
    tasks = discover_tasks()
    if not tasks:
        print("No tasks found!")
        return

    # Determine which tasks to augment
    if args.all:
        tasks_to_augment = tasks
    elif args.task:
        tasks_to_augment = [t for t in tasks if t[0] == args.task]
        if not tasks_to_augment:
            print(f"Task '{args.task}' not found!")
            return
    else:
        # Interactive selection
        for i, (name, _, _) in enumerate(tasks, 1):
            print(f"{i}. {name}")
        print(f"{len(tasks) + 1}. All tasks")

        try:
            choice = int(input(f"Choice (1-{len(tasks) + 1}): "))
            tasks_to_augment = tasks if choice == len(tasks) + 1 else [tasks[choice - 1]] if 1 <= choice <= len(tasks) else []
            if not tasks_to_augment:
                print("Invalid choice!")
                return
        except (ValueError, IndexError):
            print("Invalid input!")
            return

    # Augment selected tasks
    overall_stats = []
    for task_name, heuristics_path, template_path in tasks_to_augment:
        try:
            stats = augment_heuristics(
                task_name, heuristics_path, template_path,
                args.model, args.temperature, args.target_size, args.output_suffix
            )
            stats["task"] = task_name
            overall_stats.append(stats)
        except Exception as e:
            print(f"Error: {task_name} - {e}")

    # Summary
    if len(overall_stats) > 1:
        print("\nSummary:")
        for stats in overall_stats:
            print(f"  {stats['task']}: {stats['original']} → {stats['final']}")


if __name__ == "__main__":
    main()
