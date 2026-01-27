"""
Example script showing how to run different optimization methods (EoH, FunSearch, ReEvo, HillClimb, MCTS_AHD, LHNS)
with seed initialization, similar to later_use.py

NOTE: For fair comparison between methods, use run_fair_comparison.py instead,
      which uses standardized configuration from fair_comparison_config.py
"""

import json
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.base import TextFunctionProgramConverter as tfpc
from llm4ad.tools.llm.llm_api_openrouter import OpenRouterAPI
from dotenv import load_dotenv
import os

# Import all the methods
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.funsearch import FunSearch
from llm4ad.method.reevo import ReEvo, ReEvoProfiler
from llm4ad.method.hillclimb import HillClimb
from llm4ad.method.mcts_ahd import MCTS_AHD, MAProfiler
from llm4ad.method.lhns import LHNS, LHNSProfiler
from llm4ad.tools.profiler import ProfilerBase

# ============================================================================
# CONFIGURATION
# ============================================================================
# For fair comparison, all methods should use the same computational budget.
# You can either:
# 1. Use fair_comparison_config.py (recommended for fair comparison)
# 2. Set parameters manually here (for custom experiments)

# Computational budget (main fairness parameter)
MAX_SAMPLE_NUMS = 100  # Same for all methods for fair comparison

# Population size (for population-based methods)
POP_SIZE = 10

# Parallel execution
NUM_SAMPLERS = 1
NUM_EVALUATORS = 1

# Load environment variables
load_dotenv()
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Setup LLM (same for all methods)
llm = OpenRouterAPI(
    key=API_KEY,
    model='qwen/qwen3-4b:free',
    timeout=120
)

# Setup task
task = TSPEvaluation()

# Load seeds
with open("llm4ad/task/optimization/tsp_construct/heuristics.json") as f:
    seeds = json.load(f)


def add_seeds_to_population(method, seeds):
    """
    Helper function to add seed heuristics to the method's population.
    Works for methods that have _population with _next_gen_pop attribute.
    """
    for name, code in seeds.items():
        score, eval_time = method._evaluator.evaluate_program_record_time(code)
        if score is None:
            print(f"Skipping {name}: Evaluation failed (score is None).")
            continue
        func = tfpc.text_to_function(code)
        if func is None:
            print(f"Warning: Failed to parse function '{name}', skipping.")
            continue
        func.score = score
        func.evaluate_time = eval_time
        method._population._next_gen_pop.append(func)
        if method._profiler:
            method._profiler.register_function(func, program=code, resume_mode=True)

    # Move seeds from _next_gen_pop to _population and increment generation
    method._population.survival()


def add_seeds_to_funsearch(method, seeds):
    """
    Helper function to add seed heuristics to FunSearch's database.
    FunSearch stores the database as _database (not _programs_database).
    """
    for name, code in seeds.items():
        score, eval_time = method._evaluator.evaluate_program_record_time(code)
        if score is None:
            print(f"Skipping {name}: Evaluation failed (score is None).")
            continue

        # FunSearch uses Function objects, not Program
        func = tfpc.text_to_function(code)
        if func is None:
            print(f"Warning: Failed to parse function '{name}', skipping.")
            continue

        # FunSearch's register_function takes score as a separate parameter
        # island_id=None means add to all islands (for initialization)
        # IMPORTANT: The attribute is _database (NOT _programs_database)
        method._database.register_function(
            function=func,
            island_id=None,  # Add to all islands
            score=score
        )

        if method._profiler:
            # Set score for profiler logging
            func.score = score
            func.evaluate_time = eval_time
            method._profiler.register_function(func, program=code, resume_mode=True)


def add_seeds_to_hillclimb(method, seeds):
    """
    Helper function to initialize HillClimb with the BEST seed heuristic.
    HillClimb is a local search method that starts from a single solution.
    """
    print("  Evaluating seeds to find best starting point for HillClimb...")

    best_seed_name = None
    best_seed_code = None
    best_seed_score = float('-inf')
    best_seed_eval_time = None

    for name, code in seeds.items():
        score, eval_time = method._evaluator.evaluate_program_record_time(code)
        if score is None:
            print(f"    Skipping {name}: Evaluation failed.")
            continue

        print(f"    {name}: score = {score:.6f}")

        # Track the best seed
        if score > best_seed_score:
            best_seed_name = name
            best_seed_code = code
            best_seed_score = score
            best_seed_eval_time = eval_time

    if best_seed_code is None:
        raise RuntimeError("No valid seed found for HillClimb!")

    print(f"  Best seed: {best_seed_name} with score = {best_seed_score:.6f}")

    # Initialize HillClimb with the best seed
    func = tfpc.text_to_function(best_seed_code)
    if func is None:
        raise RuntimeError(f"Failed to parse best seed: {best_seed_name}")

    func.score = best_seed_score
    func.evaluate_time = best_seed_eval_time

    # Set as starting point for hill climbing
    method._best_function_found = func

    # Register with profiler
    if method._profiler:
        method._profiler.register_function(func, program=best_seed_code, resume_mode=True)


def add_seeds_to_lhns(method, seeds):
    """
    Helper function to add seed heuristics to LHNS's elite set.
    LHNS uses an elite_set structure with specialized function types.
    """
    from llm4ad.method.lhns.func_ruin import LHNSTextFunctionProgramConverter

    converter = LHNSTextFunctionProgramConverter()
    for name, code in seeds.items():
        score, eval_time = method._evaluator.evaluate_program_record_time(code)
        if score is None:
            print(f"Skipping {name}: Evaluation failed (score is None).")
            continue

        func = converter.text_to_function(code)
        if func is None:
            print(f"Warning: Failed to parse function '{name}', skipping.")
            continue
        func.score = score
        func.evaluate_time = eval_time
        method._elite_set.insert(func)
        if method._profiler:
            method._profiler.register_function(func, program=code, resume_mode=True)


# ============================================================================
# 1. EoH (Evolution of Heuristics)
# ============================================================================
def run_eoh():
    print("\n" + "="*80)
    print("Running EoH (Evolution of Heuristics)...")
    print(f"Budget: {MAX_SAMPLE_NUMS} evaluations")
    print("="*80)

    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir='logs/eoh', log_style='simple'),
        evaluation=task,
        max_sample_nums=MAX_SAMPLE_NUMS,
        pop_size=POP_SIZE,
        resume_mode=True,      # skip random init; use seeds instead
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        debug_mode=False
    )

    # Add seeds to EoH's population
    add_seeds_to_population(method, seeds)

    method.run()
    print("EoH completed!")


# ============================================================================
# 2. FunSearch
# ============================================================================
def run_funsearch():
    print("\n" + "="*80)
    print("Running FunSearch...")
    print(f"Budget: {MAX_SAMPLE_NUMS} evaluations")
    print("="*80)

    method = FunSearch(
        llm=llm,
        profiler=ProfilerBase(log_dir='logs/funsearch', log_style='simple'),
        evaluation=task,
        max_sample_nums=MAX_SAMPLE_NUMS,
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
    )

    # Add seeds to FunSearch's programs database
    add_seeds_to_funsearch(method, seeds)

    method.run()
    print("FunSearch completed!")


# ============================================================================
# 3. ReEvo (Reflective Evolution)
# ============================================================================
def run_reevo():
    print("\n" + "="*80)
    print("Running ReEvo...")
    print(f"Budget: {MAX_SAMPLE_NUMS} evaluations")
    print("="*80)

    method = ReEvo(
        llm=llm,
        profiler=ReEvoProfiler(log_dir='logs/reevo', log_style='simple'),
        evaluation=task,
        max_sample_nums=MAX_SAMPLE_NUMS,
        pop_size=POP_SIZE,
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        resume_mode=True,      # skip random init; use seeds instead
        debug_mode=False
    )

    # Add seeds to ReEvo's population
    add_seeds_to_population(method, seeds)

    method.run()
    print("ReEvo completed!")


# ============================================================================
# 4. HillClimb
# ============================================================================
def run_hillclimb():
    print("\n" + "="*80)
    print("Running HillClimb...")
    print(f"Budget: {MAX_SAMPLE_NUMS} evaluations")
    print("="*80)

    method = HillClimb(
        llm=llm,
        profiler=ProfilerBase(log_dir='logs/hillclimb', log_style='simple'),
        evaluation=task,
        max_sample_nums=MAX_SAMPLE_NUMS,
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        resume_mode=True,  # Skip template evaluation
    )

    # HillClimb starts from a single solution (local search)
    # Initialize with the BEST seed as starting point
    add_seeds_to_hillclimb(method, seeds)

    method.run()
    print("HillClimb completed!")


# ============================================================================
# 5. MCTS_AHD (Monte Carlo Tree Search for Automatic Heuristic Design)
# ============================================================================
def run_mcts_ahd():
    print("\n" + "="*80)
    print("Running MCTS_AHD...")
    print(f"Budget: {MAX_SAMPLE_NUMS} evaluations")
    print("="*80)

    method = MCTS_AHD(
        llm=llm,
        profiler=MAProfiler(log_dir='logs/mcts_ahd', log_style='simple'),
        evaluation=task,
        max_sample_nums=MAX_SAMPLE_NUMS,
        init_size=4,
        pop_size=POP_SIZE,
        selection_num=2,
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        alpha=0.5,
        lambda_0=0.1,
        resume_mode=True,      # skip random init; use seeds instead
        debug_mode=False
    )

    # Add seeds to MCTS_AHD's population
    add_seeds_to_population(method, seeds)

    method.run()
    print("MCTS_AHD completed!")


# ============================================================================
# 6. LHNS (Large Language Model based Hyper-heuristic for Neighborhood Search)
# ============================================================================
def run_lhns():
    print("\n" + "="*80)
    print("Running LHNS...")
    print(f"Budget: {MAX_SAMPLE_NUMS} evaluations")
    print("="*80)

    method = LHNS(
        llm=llm,
        profiler=LHNSProfiler(log_dir='logs/lhns', log_style='simple'),
        evaluation=task,
        max_sample_nums=MAX_SAMPLE_NUMS,
        cooling_rate=0.1,
        elite_set_size=5,
        method='vns',  # options: 'vns', 'ils', 'ts'
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        resume_mode=True,
        debug_mode=False
    )

    # Add seeds to LHNS's elite set
    add_seeds_to_lhns(method, seeds)

    method.run()
    print("LHNS completed!")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    # You can run each method individually by uncommenting the one you want:

    # run_eoh()
    # run_funsearch()
    # run_reevo()
    # run_hillclimb()
    # run_mcts_ahd()
    # run_lhns()

    # Or run all methods sequentially (warning: this will take a long time!)
    # run_eoh()
    # run_funsearch()
    # run_reevo()
    # run_hillclimb()
    # run_mcts_ahd()
    # run_lhns()

    print("\n" + "="*80)
    print("Script ready! Uncomment the method you want to run in the main section.")
    print("Available methods:")
    print("  - run_eoh()         : EoH (Evolution of Heuristics)")
    print("  - run_funsearch()   : FunSearch algorithm")
    print("  - run_reevo()       : ReEvo (Reflective Evolution)")
    print("  - run_hillclimb()   : Hill Climbing")
    print("  - run_mcts_ahd()    : Monte Carlo Tree Search for AHD")
    print("  - run_lhns()        : LLM-based Hyper-heuristic NS")
    print("="*80)
