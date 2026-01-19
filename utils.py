import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import ast

    
def get_z_from_code_cls(source_code, encoder, tokenizer):
    """Encodes source code into a latent vector z using the [CLS] token's embedding."""
    with torch.no_grad():
        if isinstance(source_code, str):
            source_code = [source_code]
        outputs = encoder.encode(source_code)
    return outputs
    
def get_mean_cls_embedding(source_codes, encoder, tokenizer):
    """Encodes a list of source codes into latent vectors using the [CLS] token's embedding and returns the mean."""
    with torch.no_grad():
        if not isinstance(source_codes, list): 
            raise ValueError("Input should be a list of strings.")
        
        all_cls_embeddings = []
        
        for code in source_codes:
            inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512).to(encoder.device)
            outputs = encoder(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0]
            all_cls_embeddings.append(cls_embedding)

        # Stack all CLS embeddings and compute the mean
        all_cls_embeddings = torch.stack(all_cls_embeddings)
        mean_cls_embedding = torch.mean(all_cls_embeddings, dim=0)
        
        return mean_cls_embedding


def optimize_latent_with_gradient_ascent(z_start, regressor, steps=10, lr=0.001):
    """Optimizes a latent vector 'z' by performing gradient ascent on R(z)."""
    z = z_start.clone().detach().requires_grad_(True)
    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()
        pred = regressor(z) 
        pred.backward()
        with torch.no_grad():
            z += lr * z.grad
    return z.detach()

def find_closest_program(target_z, known_z_tensor, known_programs, use_cosine=True):
    """Finds the program with the latent vector closest to the target_z."""
    with torch.no_grad():
        if use_cosine: 
            similarities = F.cosine_similarity(target_z, known_z_tensor)
            closest_index = torch.argmax(similarities).item()
        else:
            distances = torch.cdist(target_z, known_z_tensor)
            closest_index = torch.argmin(distances).item()
    return known_programs[closest_index]

def is_valid_python(code):
   try:
       ast.parse(code)
   except SyntaxError:
       return False
   return True


def compare_code(code1: str, code2: str) -> bool:
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        # 2. Dump the trees into a string representation. This string
        #    represents the code's structure, including names.
        dump1 = ast.dump(tree1)
        dump2 = ast.dump(tree2)
        return dump1 == dump2

    except SyntaxError as e:
        #print(f"Could not compare code due to a syntax error: {e}")
        return False
    
def extract_python_code(text):
    """
    Extracts Python code blocks from a string.
    """
    code_blocks = []
    start_marker = "```python"
    end_marker = "```"
    start_index = text.find(start_marker)
    while start_index != -1:
        start_index += len(start_marker)
        end_index = text.find(end_marker, start_index)
        if end_index != -1:
            code = text[start_index:end_index].strip()
            code_blocks.append(code)
            start_index = text.find(start_marker, end_index + len(end_marker))
        else:
            break
    return code_blocks


def extract_python_code_robust(text, include_preface=True):
    """
    Robustly extracts Python code from text, removing trailing comments and examples.

    This function:
    1. Extracts code from markdown blocks (```python...```) or uses raw text
    2. Uses AST parsing to extract only functions and their preface (imports, globals)
    3. Removes trailing comments like "# Example usage:" sections

    Args:
        text: The text containing Python code (with or without markdown blocks)
        include_preface: If True, includes imports/globals before functions. Default True.

    Returns:
        str: Clean Python code containing only the preface and function definitions
    """
    from base.code import TextFunctionProgramConverter

    # First, try to extract code from markdown blocks
    if "```python" in text:
        start_idx = text.find("```python") + len("```python")
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            code = text[start_idx:end_idx].strip()
        else:
            code = text[start_idx:].strip()
    else:
        code = text.strip()

    # Use AST-based parsing to extract only the program structure
    # This automatically removes trailing comments and example usage
    program = TextFunctionProgramConverter.text_to_program(code)

    if program is None:
        # If AST parsing fails, fall back to the original code
        return code

    # Convert the program back to string (this excludes trailing comments)
    if include_preface:
        return str(program)
    else:
        # Return only the functions without preface
        return '\n'.join([str(f) for f in program.functions])

def parse_bin_packing_file(file_path, org_datasets, name):
    datasets = {name: {}}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Number of datasets
        num_datasets = int(lines[i].strip())
        i += 1

        for _ in range(num_datasets):
            dataset_name = lines[i].strip()
            i += 1

            # Capacity and number of items
            capacity, num_items,_ = map(int, lines[i].split())
            i += 1

            # List of items
            items = []
            for _ in range(num_items):
                item_size = int(lines[i].strip())
                items.append(item_size)
                i += 1

            datasets[name][dataset_name] = {
                'capacity': capacity,
                'num_items': num_items,
                'items': items
            }
    org_datasets[name] = datasets[name]

    return org_datasets