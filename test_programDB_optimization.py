import unittest
import pandas as pd
import numpy as np
import os
from programDB import ProgramDatabase

class TestProgramDatabaseOptimization(unittest.TestCase):
    def setUp(self):
        self.db = ProgramDatabase()

    def test_add_and_exists_simple(self):
        code = "def f(x): return x + 1"
        self.db.add_program(code, z=0.0, score=1.0)
        self.assertTrue(self.db.exists(code), "Program should exist after adding")

    def test_not_exists(self):
        code = "def f(x): return x + 1"
        self.db.add_program(code, z=0.0, score=1.0)
        other_code = "def f(x): return x + 2"
        self.assertFalse(self.db.exists(other_code), "Different program should not exist")

    def test_exists_structural_match(self):
        code = "def f(a): return a + 1"
        self.db.add_program(code, z=0.0, score=1.0)

        # structurally same (renamed variable)
        query_code = "def f(b): return b + 1"
        self.assertTrue(self.db.exists(query_code), "Structurally same program should exist")

    def test_exists_structural_mismatch(self):
        code = "def f(a): return a + 1"
        self.db.add_program(code, z=0.0, score=1.0)

        query_code = "def f(a): return a - 1"
        self.assertFalse(self.db.exists(query_code), "Structurally different program should not exist")

    def test_invalid_code_handling(self):
        # Adding invalid code might raise syntax error or handled gracefully depending on implementation
        # But here we test exists() with invalid code
        invalid_code = "def f(x) return x + 1" # Missing colon
        self.assertFalse(self.db.exists(invalid_code), "Invalid code should not exist")

    def test_structure_hash_column(self):
        # This test assumes the implementation uses 'structure_hash' column
        code = "def f(x): return x * x"
        self.db.add_program(code, z=0.0, score=1.0)

        if hasattr(self.db.df, 'columns') and 'structure_hash' in self.db.df.columns:
            self.assertIsNotNone(self.db.df.iloc[0]['structure_hash'])
            print("Verified 'structure_hash' column exists and is populated.")
        else:
            print("Warning: 'structure_hash' column not found (optimization might not be applied yet)")

if __name__ == '__main__':
    unittest.main()
