# program_database.py

import pandas as pd
import torch
import numpy as np
import os
import json
import utils as utils
import torch.nn.functional as F
from typing import Optional, Tuple, List, Sequence
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm

import ast
from base.code import TextFunctionProgramConverter
from base.evaluate import SecureEvaluator

class CodeNormalizer(ast.NodeTransformer):
    """
    Traverses an AST and normalizes variable and function names.
    """
    def __init__(self):
        self.name_map = {}
        self.counter = 0

    def get_normalized_name(self, name):
        """Creates a generic name like 'v_0', 'v_1', etc."""
        if name not in self.name_map:
            self.name_map[name] = f"v_{self.counter}"
            self.counter += 1
        return self.name_map[name]

    def visit_Name(self, node):
        """Handles variable names."""
        node.id = self.get_normalized_name(node.id)
        return node

    def visit_FunctionDef(self, node):
        """Handles function names and their arguments."""
        node.name = self.get_normalized_name(node.name)
        # Also normalize argument names
        if node.args.args:
            for arg in node.args.args:
                arg.arg = self.get_normalized_name(arg.arg)
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        """Handles arguments in function definitions (for Python 3.8+)."""
        node.arg = self.get_normalized_name(node.arg)
        return node

def get_structure_hash(code_str: str) -> Optional[str]:
    """
    Computes a hash string representing the structure of the code,
    invariant to variable names, whitespace, and comments.
    Returns None if the code has syntax errors.
    """
    try:
        # 1. Parse code strings into ASTs
        tree = ast.parse(code_str)

        # 2. Normalize tree
        normalizer = CodeNormalizer()
        normalized_tree = normalizer.visit(tree)

        # 3. Dump the normalized tree
        return ast.dump(normalized_tree)
    except SyntaxError:
        return None

def are_codes_structurally_same(code1_str: str, code2_str: str) -> bool:
    """
    Checks if two Python code strings are structurally equivalent,
    ignoring comments, whitespace, and variable/function names.
    """
    h1 = get_structure_hash(code1_str)
    h2 = get_structure_hash(code2_str)
    return h1 is not None and h2 is not None and h1 == h2


class ProgramDatabase:
    """Minimal archive with island model columns."""

    def __init__(self):
        self.df = pd.DataFrame(
            columns=[
                "program_id", "code", "z", "score", "origin",
                "generation", "structure_hash"
            ]
        )
        self.df.set_index("program_id", inplace=True)
        self.program_counter = 0
    def add_program(
        self,
        code,
        z,
        score: Optional[float] = None,
        origin: str = "generated",
        generation: int = 0,
    ) -> int:
        """Insert a new program  Returns the program_id."""
        pid = self.program_counter
        self.program_counter += 1

        
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy().astype(np.float32)
        z = np.squeeze(z)
                

        # Calculate mean score from array or use single value
        if isinstance(score, (list, np.ndarray)):
            score_array = np.array(score)
            mean_score = float(np.nanmean(score_array)) if np.isfinite(score_array).any() else np.nan
        else:
            mean_score = float(score) if score is not None and np.isfinite(score) else np.nan

        # Compute structure hash
        struct_hash = get_structure_hash(code)

        row = {
            "program_id": pid,
            "code": code,
            "z": z,
            "score": mean_score,
            "origin": origin,
            "generation": generation,
            "structure_hash": struct_hash,
        }
        self.df.loc[pid] = row
        return pid
    def remove_program(self, program_id):
        # Remove the row with the given program_id
        if program_id in self.df.index:
            self.df.drop(program_id, inplace=True)

    def get_top_n(self, n=5):
        return self.df.sort_values('score', ascending=False).head(n)

    def update_score(self, program_id, new_score):
        self.df.at[program_id, 'score'] = new_score

    def get_by_id(self, program_id):
        return self.df.loc[program_id]

    def exists(self, code_to_check):
        target_hash = get_structure_hash(code_to_check)
        if target_hash is None:
            return False
        # Use fast lookup via structure_hash column
        return (self.df['structure_hash'] == target_hash).any()

    def to_disk(self, path):
        df_copy = self.df.copy()
        df_copy['z'] = df_copy['z'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df_copy.to_parquet(path, compression='brotli')

    def load_from_disk(self, path):
        if not os.path.exists(path):
            print(f"No database file at {path}. Starting fresh.")
            return
        self.df = pd.read_parquet(path)
        #self.df.set_index('program_id', inplace=True)
        self.df['z'] = self.df['z'].apply(lambda x: np.squeeze(np.array(x, dtype=np.float32)))

        # Backfill structure_hash if missing
        if 'structure_hash' not in self.df.columns:
             print("Computing structure hashes for existing database entries...")
             self.df['structure_hash'] = self.df['code'].apply(get_structure_hash)

        self.program_counter = len(self.df)

    def __len__(self):
        return len(self.df)

    def all_codes(self):
        return self.df['code'].tolist()

    def load_func_from_json(self, json_path, encoder_model, encoder_tokenizer, eval, device="cpu"):
        """
        Initialize the database from a JSON file of function name: code.
        Each code is encoded using encoder_model and encoder_tokenizer.
        """
        # Create secure evaluator wrapper
        secure_eval = SecureEvaluator(eval, debug_mode=False)

        with open(json_path, "r") as f:
            func_dict = json.load(f)
        for func_name, code_str in tqdm(func_dict.items(), desc="Processing"):
            with torch.no_grad():
                z = encoder_model.encode([code_str])[0]
                score = None
                try:
                    # Use secure evaluation (handles exec, timeout, and process isolation)
                    score = secure_eval.evaluate_program(code_str)
                except Exception as e:
                    print(f"  - Could not evaluate function {func_name}. Error: {e}")
                    score = None
                if score is None :
                    print(f"  - Could not evaluate function {func_name}")
                    continue
            self.add_program(code=code_str, z=z, score=score, origin="LLM")
        self.program_counter = len(self.df)
        print(f"Initialized with {len(func_dict)} programs from {json_path}")
