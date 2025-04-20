import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from tempfile import NamedTemporaryFile
import subprocess
import sys
import pickle
from collections import defaultdict
from codebleu import calc_codebleu
import warnings
import argparse
from datasets import load_dataset

# Suppress warnings
warnings.filterwarnings("ignore")

def safe_exec(func_code: str, timeout: int = 30) -> Any:
    """
    Safely execute Python code in an isolated subprocess.
    
    Args:
        func_code: Python code to execute (string)
        timeout: Maximum execution time in seconds
        
    Returns:
        The value of 'result' from the executed code
        
    Raises:
        TimeoutError: If execution exceeds timeout
        RuntimeError: If code execution fails
    """
    # Create a temporary file for the execution
    with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write the code with proper result capture
        f.write(f"""
import pickle
import sys
from traceback import format_exc

try:
    local_vars = {{}}
    exec('''{func_code}''', {{}}, local_vars)
    result = local_vars.get('result', None)
    with open('{f.name}.result', 'wb') as res_file:
        pickle.dump(('success', result), res_file)
except Exception as e:
    with open('{f.name}.result', 'wb') as res_file:
        pickle.dump(('error', str(e), format_exc()), res_file)
""")
        temp_path = f.name
    
    try:
        # Run in a subprocess with resource limits
        proc = subprocess.Popen(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise TimeoutError(f"Execution timed out after {timeout} seconds")
        
        # Check for results
        result_file = f"{temp_path}.result"
        if os.path.exists(result_file):
            with open(result_file, 'rb') as res_file:
                status, *data = pickle.load(res_file)
                
            if status == 'success':
                return data[0]
            else:
                error_msg, traceback = data
                raise RuntimeError(f"Execution failed: {error_msg}\\n\\n{traceback}")
        else:
            raise RuntimeError("No result file produced. Possible crash.\\n"
                              f"STDOUT: {stdout.decode()}\\nSTDERR: {stderr.decode()}")
            
    finally:
        # Clean up temporary files
        for fpath in [temp_path, f"{temp_path}.result"]:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except:
                pass

def split_asserts(input_string: str) -> List[str]:
    """
    Split a string containing assert statements into a list of individual assert statements.
    
    Args:
        input_string: String containing assert statements
        
    Returns:
        List of individual assert statements
    """
    # Split the string by lines and process each line that starts with 'assert'
    lines = input_string.split('\n')
    asserts = []
    current_assert = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('assert'):
            # If we have a current assert being built, add it to the list
            if current_assert is not None:
                asserts.append(current_assert)
            # Start a new assert
            current_assert = line
        elif current_assert is not None:
            # Continuation of a multi-line assert
            current_assert += ' ' + line.strip()
    
    # Add the last assert if there is one
    if current_assert is not None:
        asserts.append(current_assert)
    
    # Clean up each assert statement
    cleaned_asserts = []
    for assert_stmt in asserts:
        # Remove any trailing commas
        if assert_stmt.endswith(','):
            assert_stmt = assert_stmt[:-1]
        # Remove extra whitespace
        assert_stmt = ' '.join(assert_stmt.split())
        cleaned_asserts.append(assert_stmt)
    
    return cleaned_asserts

def count_passed_assertions(code: str, assertions: List[str]) -> Tuple[int, int]:
    """
    Count how many assertions pass when executed with the given code.
    
    Args:
        code: Python code to test
        assertions: List of assert statements to run against the code
        
    Returns:
        Tuple of (passed_count, total_count)
    """
    passed, total = 0, 0
    
    for stmt in assertions:
        try:
            full_code = f"""{code}\n\n{stmt}"""
            safe_exec(full_code, 10)
            passed += 1
        except (TimeoutError, RuntimeError, MemoryError, AssertionError, Exception):
            pass
        total += 1
    
    return passed, total

def get_code_bleu(refined: str, canonical: str) -> float:
    """
    Calculate CodeBLEU score between refined and canonical code.
    
    Args:
        refined: Refined code
        canonical: Canonical code (reference)
        
    Returns:
        CodeBLEU score
    """
    return calc_codebleu([canonical], [refined], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)['codebleu']

def generate_incorrect_solutions(prompt: str, num_solutions: int = 5) -> List[str]:
    """
    Generate incorrect solutions for a given problem prompt.
    
    This is a placeholder function that in a real implementation would call a language model
    to generate buggy or incomplete solutions for a problem description.
    
    Args:
        prompt: Problem description
        num_solutions: Number of solutions to generate
        
    Returns:
        List of incorrect solution strings
    """
    # This should be implemented with an actual LLM call
    # For demonstration purposes, we return placeholder solutions
    return [
        f"# Buggy solution {i} for {prompt[:20]}...\n\ndef solution():\n    pass" 
        for i in range(num_solutions)
    ]

def prepare_mbpp_dataset() -> pd.DataFrame:
    """
    Prepare the MBPP dataset for code refinement.
    
    Returns:
        DataFrame containing MBPP problems with buggy solutions and metrics
    """
    print("Preparing MBPP dataset...")
    # Load MBPP dataset
    splits = {'train': 'full/train-00000-of-00001.parquet', 
              'test': 'full/test-00000-of-00001.parquet', 
              'validation': 'full/validation-00000-of-00001.parquet', 
              'prompt': 'full/prompt-00000-of-00001.parquet'}
    
    try:
        mbpp_df = pd.read_parquet("hf://datasets/google-research-datasets/mbpp/" + splits["test"])
    except Exception as e:
        print(f"Error loading MBPP dataset: {e}")
        print("Downloading MBPP dataset from HuggingFace...")
        # Alternative loading method
        mbpp_dataset = load_dataset("google-research-datasets/mbpp", split="test")
        mbpp_df = mbpp_dataset.to_pandas()
    
    # Generate incorrect solutions
    results = []
    for i, row in tqdm(mbpp_df.iterrows(), total=len(mbpp_df), desc="Generating buggy solutions for MBPP"):
        problem_text = row['text']
        canonical_solution = row['code']
        test_list = row['test_list']
        
        # Generate incorrect solutions
        buggy_solutions = generate_incorrect_solutions(problem_text)
        
        for buggy_solution in buggy_solutions:
            # Evaluate solution
            passed, total = count_passed_assertions(buggy_solution, test_list)
            codebleu_score = get_code_bleu(buggy_solution, canonical_solution)
            
            # Calculate combined reward
            test_ratio = passed / total if total > 0 else 0
            combined_reward = 0.5 * codebleu_score + 0.5 * test_ratio
            
            # Store results
            results.append({
                'dataset': 'MBPP',
                'task_id': row['task_id'],
                'problem_description': problem_text,
                'canonical_solution': canonical_solution,
                'buggy_solution': buggy_solution,
                'tests': test_list,
                'passed_tests': passed,
                'total_tests': total,
                'test_ratio': test_ratio,
                'codebleu': codebleu_score,
                'combined_reward': combined_reward
            })
    
    return pd.DataFrame(results)

def prepare_humaneval_dataset() -> pd.DataFrame:
    """
    Prepare the HumanEval dataset for code refinement.
    
    Returns:
        DataFrame containing HumanEval problems with buggy solutions and metrics
    """
    print("Preparing HumanEval dataset...")
    # Load HumanEval dataset
    try:
        humaneval_dataset = load_dataset("openai/human-eval", split="test")
        humaneval_df = humaneval_dataset.to_pandas()
    except Exception as e:
        print(f"Error loading HumanEval dataset: {e}")
        return pd.DataFrame()
    
    results = []
    for i, row in tqdm(humaneval_df.iterrows(), total=len(humaneval_df), desc="Generating buggy solutions for HumanEval"):
        prompt = row['prompt']
        canonical_solution = row['canonical_solution']
        test = row['test']
        entry_point = row['entry_point']
        
        # Extract assert statements from test
        test_code = test.split("\n\nMETADATA")[0]
        check_function_pattern = r'def check\(candidate\):(.*?)(?=\ndef|\Z)'
        check_function_match = re.search(check_function_pattern, test_code, re.DOTALL)
        
        if not check_function_match:
            continue
            
        check_body = check_function_match.group(1).strip()
        assert_statements = [stmt.strip() for stmt in check_body.split('\n') if stmt.strip().startswith('assert ')]
        
        # Modify assert statements to use the function directly instead of 'candidate'
        modified_asserts = []
        for stmt in assert_statements:
            modified_stmt = stmt.replace('candidate', entry_point)
            modified_asserts.append(modified_stmt)
        
        # Generate incorrect solutions
        full_prompt = prompt
        buggy_solutions = generate_incorrect_solutions(full_prompt)
        
        for buggy_solution in buggy_solutions:
            # Combine prompt and buggy solution for evaluation
            full_solution = prompt + buggy_solution
            
            # Evaluate solution
            passed, total = count_passed_assertions(full_solution, modified_asserts)
            codebleu_score = get_code_bleu(buggy_solution, canonical_solution)
            
            # Calculate combined reward
            test_ratio = passed / total if total > 0 else 0
            combined_reward = 0.5 * codebleu_score + 0.5 * test_ratio
            
            # Store results
            results.append({
                'dataset': 'HumanEval',
                'task_id': row['task_id'],
                'problem_description': prompt,
                'canonical_solution': canonical_solution,
                'buggy_solution': buggy_solution,
                'tests': modified_asserts,
                'passed_tests': passed,
                'total_tests': total,
                'test_ratio': test_ratio,
                'codebleu': codebleu_score,
                'combined_reward': combined_reward
            })
    
    return pd.DataFrame(results)

def prepare_apps_dataset() -> pd.DataFrame:
    """
    Prepare the APPS dataset for code refinement.
    
    Returns:
        DataFrame containing APPS problems with buggy solutions and metrics
    """
    print("Preparing APPS dataset...")
    # Load APPS dataset
    try:
        apps_dataset = load_dataset("codeparrot/apps", split="test")
        apps_df = apps_dataset.to_pandas()
    except Exception as e:
        print(f"Error loading APPS dataset: {e}")
        return pd.DataFrame()
    
    results = []
    for i, row in tqdm(apps_df.iterrows(), total=len(apps_df), desc="Generating buggy solutions for APPS"):
        problem_id = row['problem_id']
        problem = row['problem']
        solutions = row['solutions']
        test_inputs = row['input']
        test_outputs = row['output']
        
        if not solutions or not isinstance(solutions, list):
            continue
            
        canonical_solution = solutions[0]
        
        # Generate test assertions
        test_asserts = []
        for idx, (test_in, test_out) in enumerate(zip(test_inputs, test_outputs)):
            if not isinstance(test_in, str) or not isinstance(test_out, str):
                continue
                
            # Create a simple test function that validates inputs and outputs
            test_assert = f"""
def test_case_{idx}(solution_func):
    import io
    import sys
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO('''{test_in}''')
    sys.stdout = io.StringIO()
    solution_func()
    output = sys.stdout.getvalue().strip()
    sys.stdin, sys.stdout = old_stdin, old_stdout
    expected = '''{test_out}'''.strip()
    assert output == expected, f"Expected: {{expected}}, Got: {{output}}"

# Run the test
test_case_{idx}(main)
"""
            test_asserts.append(test_assert)
        
        # Generate incorrect solutions
        buggy_solutions = generate_incorrect_solutions(problem)
        
        for buggy_solution in buggy_solutions:
            # Evaluate solution
            passed, total = 0, len(test_asserts)
            
            for test_assert in test_asserts:
                solution_with_main = buggy_solution + "\n\ndef main():\n    solution()"
                try:
                    safe_exec(solution_with_main + "\n\n" + test_assert, 30)
                    passed += 1
                except Exception:
                    pass
            
            codebleu_score = get_code_bleu(buggy_solution, canonical_solution)
            
            # Calculate combined reward
            test_ratio = passed / total if total > 0 else 0
            combined_reward = 0.5 * codebleu_score + 0.5 * test_ratio
            
            # Store results
            results.append({
                'dataset': 'APPS',
                'task_id': problem_id,
                'problem_description': problem,
                'canonical_solution': canonical_solution,
                'buggy_solution': buggy_solution,
                'tests': test_asserts,
                'passed_tests': passed,
                'total_tests': total,
                'test_ratio': test_ratio,
                'codebleu': codebleu_score,
                'combined_reward': combined_reward
            })
    
    return pd.DataFrame(results)

def get_combined_dataset() -> pd.DataFrame:
    """
    Create a combined dataset from MBPP, HumanEval, and APPS.
    
    Returns:
        Combined DataFrame with all datasets
    """
    mbpp_df = prepare_mbpp_dataset()
    humaneval_df = prepare_humaneval_dataset()
    apps_df = prepare_apps_dataset()
    
    combined_df = pd.concat([mbpp_df, humaneval_df, apps_df], ignore_index=True)
    return combined_df

def main():
    """Main function to run the data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare datasets for code refinement training")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save prepared datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["mbpp", "humaneval", "apps"], 
                        help="Datasets to prepare (options: mbpp, humaneval, apps, all)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if "all" in args.datasets or set(args.datasets) == {"mbpp", "humaneval", "apps"}:
        combined_df = get_combined_dataset()
        combined_df.to_parquet(os.path.join(args.output_dir, "combined_dataset.parquet"))
        print(f"Combined dataset saved to {os.path.join(args.output_dir, 'combined_dataset.parquet')}")
    else:
        if "mbpp" in args.datasets:
            mbpp_df = prepare_mbpp_dataset()
            mbpp_df.to_parquet(os.path.join(args.output_dir, "mbpp_dataset.parquet"))
            print(f"MBPP dataset saved to {os.path.join(args.output_dir, 'mbpp_dataset.parquet')}")
            
        if "humaneval" in args.datasets:
            humaneval_df = prepare_humaneval_dataset()
            humaneval_df.to_parquet(os.path.join(args.output_dir, "humaneval_dataset.parquet"))
            print(f"HumanEval dataset saved to {os.path.join(args.output_dir, 'humaneval_dataset.parquet')}")
            
        if "apps" in args.datasets:
            apps_df = prepare_apps_dataset()
            apps_df.to_parquet(os.path.join(args.output_dir, "apps_dataset.parquet"))
            print(f"APPS dataset saved to {os.path.join(args.output_dir, 'apps_dataset.parquet')}")

if __name__ == "__main__":
    main() 