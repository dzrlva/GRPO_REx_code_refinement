#!/usr/bin/env python3
import os
import json
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CodeProblem:
    problem_id: str
    prompt: str
    tests: List[str]
    solution: str
    buggy_solutions: List[str] = None
    source: str = "unknown"

def load_humaneval_dataset(humaneval_path: str) -> List[CodeProblem]:
    """Load HumanEval dataset."""
    logger.info(f"Loading HumanEval dataset from {humaneval_path}")
    problems = []
    
    with open(humaneval_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data['task_id']
            prompt = data['prompt']
            tests = []
            
            # Extract tests from test and entry_point
            test_code = data['test']
            entry_point = data['entry_point']
            
            # Split test code into individual test cases
            test_lines = test_code.strip().split("\n")
            current_test = []
            for line in test_lines:
                if line.startswith("def"):
                    if current_test:
                        tests.append("\n".join(current_test))
                        current_test = []
                current_test.append(line)
            
            if current_test:
                tests.append("\n".join(current_test))
            
            # Extract solution (canonical solution)
            solution = data['canonical_solution']
            
            # Create CodeProblem
            problem = CodeProblem(
                problem_id=task_id,
                prompt=prompt,
                tests=tests,
                solution=solution,
                source="humaneval"
            )
            problems.append(problem)
    
    logger.info(f"Loaded {len(problems)} problems from HumanEval")
    return problems

def load_mbpp_dataset(mbpp_path: str) -> List[CodeProblem]:
    """Load MBPP dataset."""
    logger.info(f"Loading MBPP dataset from {mbpp_path}")
    problems = []
    
    with open(mbpp_path, 'r') as f:
        data = json.load(f)
        
    for item in data:
        # Skip if no solution
        if 'code' not in item or not item['code'].strip():
            continue
            
        # Create CodeProblem
        problem = CodeProblem(
            problem_id=f"mbpp_{item['task_id']}",
            prompt=item['text'],
            tests=item['test_list'],
            solution=item['code'],
            source="mbpp"
        )
        problems.append(problem)
    
    logger.info(f"Loaded {len(problems)} problems from MBPP")
    return problems

def load_apps_dataset(apps_path: str, max_problems: int = 500) -> List[CodeProblem]:
    """Load a subset of APPS dataset."""
    logger.info(f"Loading APPS dataset from {apps_path}")
    problems = []
    
    # Get list of problem directories
    problem_dirs = list(Path(apps_path).glob("*/*"))
    
    # Limit number of problems
    if max_problems > 0:
        random.shuffle(problem_dirs)
        problem_dirs = problem_dirs[:max_problems]
    
    for problem_dir in tqdm(problem_dirs, desc="Loading APPS problems"):
        # Read problem
        try:
            with open(problem_dir / "question.txt", "r", encoding="utf-8") as f:
                prompt = f.read()
            
            # Read solution if available
            solutions_file = problem_dir / "solutions.json"
            if not solutions_file.exists():
                continue
                
            with open(solutions_file, "r", encoding="utf-8") as f:
                solutions_data = json.load(f)
                
            if not solutions_data:
                continue
                
            # Get first solution
            solution = solutions_data[0]
            
            # Read test cases
            with open(problem_dir / "input_output.json", "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            tests = []
            for i, (test_input, test_output) in enumerate(zip(test_data["inputs"], test_data["outputs"])):
                test_case = f'def test_{i}():\n    assert solve({test_input}) == {test_output}'
                tests.append(test_case)
            
            problem = CodeProblem(
                problem_id=f"apps_{problem_dir.parent.name}_{problem_dir.name}",
                prompt=prompt,
                tests=tests,
                solution=solution,
                source="apps"
            )
            problems.append(problem)
            
        except Exception as e:
            logger.warning(f"Error loading APPS problem {problem_dir}: {e}")
    
    logger.info(f"Loaded {len(problems)} problems from APPS")
    return problems

def generate_buggy_solutions(solution: str, num_bugs: int = 3) -> List[str]:
    """Generate buggy versions of a solution with syntax and semantic errors."""
    buggy_solutions = []
    
    # Types of bugs to introduce
    bug_types = [
        "change_variable_name",
        "remove_line",
        "change_operator",
        "modify_condition",
        "off_by_one",
        "flip_condition",
        "change_return_value",
        "add_unnecessary_code",
    ]
    
    # Helper function to introduce bugs
    def introduce_bug(code: str, bug_type: str) -> str:
        lines = code.split("\n")
        if not lines:
            return code
            
        # Skip leading imports, docstrings, and function definition
        start_idx = 0
        for i, line in enumerate(lines):
            if (line.strip() and not line.strip().startswith("import") and 
                not line.strip().startswith("from") and 
                not line.strip().startswith("#") and
                not line.strip().startswith('"""') and
                not line.strip().startswith("'''") and
                "def " not in line):
                start_idx = i
                break
        
        # Don't modify empty functions
        if start_idx >= len(lines) - 1:
            return code
        
        # Apply bug based on type
        if bug_type == "change_variable_name":
            # Find variable names in the code
            code_body = "\n".join(lines[start_idx:])
            variables = []
            for line in lines[start_idx:]:
                # Look for variable assignments
                if "=" in line and not "==" in line and not "<=" in line and not ">=" in line:
                    var_name = line.split("=")[0].strip()
                    if var_name and var_name.isidentifier():
                        variables.append(var_name)
            
            if variables:
                var_to_change = random.choice(variables)
                new_var = var_to_change + "_x"
                
                # Replace one occurrence
                changed = False
                for i in range(start_idx, len(lines)):
                    if var_to_change in lines[i] and random.random() < 0.7 and not changed:
                        lines[i] = lines[i].replace(var_to_change, new_var, 1)
                        changed = True
                        
                        # But keep other occurrences the same to introduce a bug
                        break
            
        elif bug_type == "remove_line":
            # Remove a non-empty, non-function definition line
            valid_lines = [i for i in range(start_idx, len(lines)) 
                          if lines[i].strip() and "def " not in lines[i] and "return" not in lines[i]]
            if valid_lines:
                idx_to_remove = random.choice(valid_lines)
                lines.pop(idx_to_remove)
                
        elif bug_type == "change_operator":
            # Change +, -, *, / to another operator
            operators = {"+": "-", "-": "+", "*": "/", "/": "*", "==": "!=", "!=": "==", "<": ">=", ">": "<="}
            for i in range(start_idx, len(lines)):
                for op, new_op in operators.items():
                    if op in lines[i] and random.random() < 0.3:
                        lines[i] = lines[i].replace(op, new_op, 1)
                        return "\n".join(lines)
                        
        elif bug_type == "modify_condition":
            # Find if conditions and modify them
            for i in range(start_idx, len(lines)):
                if "if " in lines[i] or "elif " in lines[i] or "while " in lines[i]:
                    if "<" in lines[i]:
                        lines[i] = lines[i].replace("<", "<=")
                    elif "<=" in lines[i]:
                        lines[i] = lines[i].replace("<=", "<")
                    elif ">" in lines[i]:
                        lines[i] = lines[i].replace(">", ">=")
                    elif ">=" in lines[i]:
                        lines[i] = lines[i].replace(">=", ">")
                    elif "==" in lines[i]:
                        lines[i] = lines[i].replace("==", "!=")
                    elif "!=" in lines[i]:
                        lines[i] = lines[i].replace("!=", "==")
                    return "\n".join(lines)
                    
        elif bug_type == "off_by_one":
            # Introduce off-by-one errors in loops or array indices
            for i in range(start_idx, len(lines)):
                # Look for indices like a[i] or ranges like range(n)
                if "[" in lines[i] and "]" in lines[i]:
                    idx_start = lines[i].find("[")
                    idx_end = lines[i].find("]", idx_start)
                    idx_content = lines[i][idx_start+1:idx_end].strip()
                    
                    if idx_content.isdigit():
                        # Change numeric index
                        new_idx = int(idx_content) + random.choice([-1, 1])
                        if new_idx >= 0:  # Avoid negative indices
                            lines[i] = lines[i][:idx_start+1] + str(new_idx) + lines[i][idx_end:]
                            return "\n".join(lines)
                            
                elif "range(" in lines[i]:
                    # Change range limit
                    range_start = lines[i].find("range(")
                    range_end = lines[i].find(")", range_start)
                    range_args = lines[i][range_start+6:range_end].split(",")
                    
                    if len(range_args) == 1 and range_args[0].strip().isdigit():
                        # range(n) -> range(n+1) or range(n-1)
                        limit = int(range_args[0].strip())
                        new_limit = limit + random.choice([-1, 1])
                        if new_limit > 0:  # Avoid negative or zero ranges
                            lines[i] = lines[i][:range_start+6] + str(new_limit) + lines[i][range_end:]
                            return "\n".join(lines)
                    
        elif bug_type == "flip_condition":
            # Flip a condition (if x becomes if not x)
            for i in range(start_idx, len(lines)):
                if "if " in lines[i] and "not " not in lines[i]:
                    cond_start = lines[i].find("if ") + 3
                    cond_end = lines[i].find(":", cond_start)
                    if cond_end > cond_start:
                        condition = lines[i][cond_start:cond_end].strip()
                        lines[i] = lines[i][:cond_start] + "not (" + condition + ")" + lines[i][cond_end:]
                        return "\n".join(lines)
                elif "if not" in lines[i]:
                    cond_start = lines[i].find("if not") + 7
                    cond_end = lines[i].find(":", cond_start)
                    if cond_end > cond_start:
                        condition = lines[i][cond_start:cond_end].strip()
                        if condition.startswith("(") and condition.endswith(")"):
                            condition = condition[1:-1]
                        lines[i] = lines[i][:lines[i].find("if not")] + "if " + condition + lines[i][cond_end:]
                        return "\n".join(lines)
                        
        elif bug_type == "change_return_value":
            # Modify the return value
            return_lines = [i for i in range(len(lines)) if "return " in lines[i]]
            if return_lines:
                i = random.choice(return_lines)
                return_start = lines[i].find("return ") + 7
                return_val = lines[i][return_start:].strip()
                
                if return_val.isdigit():
                    # Change numeric return value
                    new_val = int(return_val) + random.choice([-1, 1])
                    lines[i] = lines[i][:return_start] + str(new_val)
                elif return_val in ["True", "False"]:
                    # Flip boolean return value
                    new_val = "False" if return_val == "True" else "True"
                    lines[i] = lines[i][:return_start] + new_val
                elif return_val.startswith('"') or return_val.startswith("'"):
                    # Modify string return value
                    lines[i] = lines[i][:return_start] + return_val + " + '_bug'"
                elif return_val:
                    # Add operation to other return values
                    if not return_val.endswith((")", "]", "}")):  # Avoid complex expressions
                        lines[i] = lines[i][:return_start] + return_val + " + 1" if "+" not in return_val else lines[i][:return_start] + return_val + " - 1"
                        
        elif bug_type == "add_unnecessary_code":
            # Add unnecessary statement before return
            for i in range(len(lines)-1, -1, -1):
                if "return " in lines[i]:
                    # Add a useless assignment before return
                    lines.insert(i, "    temp_var = 0  # Unnecessary assignment")
                    break
            
        return "\n".join(lines)
    
    # Generate buggy solutions
    for _ in range(num_bugs):
        bug_type = random.choice(bug_types)
        buggy_solution = introduce_bug(solution, bug_type)
        if buggy_solution != solution:
            buggy_solutions.append(buggy_solution)
    
    # Ensure we have enough bugs, even if some failed to generate
    while len(buggy_solutions) < num_bugs:
        bug_type = random.choice(bug_types)
        buggy_solution = introduce_bug(solution, bug_type)
        if buggy_solution != solution and buggy_solution not in buggy_solutions:
            buggy_solutions.append(buggy_solution)
    
    return buggy_solutions[:num_bugs]  # Ensure exactly num_bugs

def prepare_dataset(
    humaneval_path: str,
    mbpp_path: str,
    apps_path: str,
    output_path: str,
    num_bugs: int = 3,
    seed: int = 42
) -> DatasetDict:
    """Prepare the dataset from HumanEval, MBPP, and APPS."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load datasets
    humaneval_problems = load_humaneval_dataset(humaneval_path)
    mbpp_problems = load_mbpp_dataset(mbpp_path)
    apps_problems = load_apps_dataset(apps_path)
    
    # Combine problems
    all_problems = humaneval_problems + mbpp_problems + apps_problems
    logger.info(f"Total problems: {len(all_problems)}")
    
    # Generate buggy solutions
    for problem in tqdm(all_problems, desc="Generating buggy solutions"):
        problem.buggy_solutions = generate_buggy_solutions(problem.solution, num_bugs)
    
    # Convert to dataset format
    dataset_records = []
    for problem in all_problems:
        # Add original solution
        dataset_records.append({
            "problem_id": problem.problem_id,
            "prompt": problem.prompt,
            "tests": problem.tests,
            "solution": problem.solution,
            "buggy_solution": None,  # No bug in original solution
            "is_buggy": False,
            "source": problem.source,
        })
        
        # Add buggy solutions
        for buggy_solution in problem.buggy_solutions:
            dataset_records.append({
                "problem_id": problem.problem_id,
                "prompt": problem.prompt,
                "tests": problem.tests,
                "solution": problem.solution,
                "buggy_solution": buggy_solution,
                "is_buggy": True,
                "source": problem.source,
            })
    
    # Shuffle and split the dataset
    random.shuffle(dataset_records)
    
    # Split dataset: 80% train, 10% validation, 10% test
    n = len(dataset_records)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_data = dataset_records[:train_size]
    val_data = dataset_records[train_size:train_size + val_size]
    test_data = dataset_records[train_size + val_size:]
    
    # Create datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save dataset
    dataset_dict.save_to_disk(output_path)
    logger.info(f"Dataset saved to {output_path}")
    
    return dataset_dict

def load_combined_dataset(dataset_path: str) -> DatasetDict:
    """Load the combined dataset from disk."""
    logger.info(f"Loading dataset from {dataset_path}")
    dataset_dict = DatasetDict.load_from_disk(dataset_path)
    logger.info(f"Loaded dataset with {len(dataset_dict['train'])} train, {len(dataset_dict['validation'])} validation, {len(dataset_dict['test'])} test examples")
    return dataset_dict

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare code refinement dataset")
    parser.add_argument("--humaneval_path", type=str, default="data/humaneval/HumanEval.jsonl", help="Path to HumanEval dataset")
    parser.add_argument("--mbpp_path", type=str, default="data/mbpp/mbpp.jsonl", help="Path to MBPP dataset")
    parser.add_argument("--apps_path", type=str, default="data/apps", help="Path to APPS dataset directory")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save processed dataset")
    parser.add_argument("--num_bugs", type=int, default=3, help="Number of buggy versions to generate per solution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create dataset
    prepare_dataset(
        humaneval_path=args.humaneval_path,
        mbpp_path=args.mbpp_path,
        apps_path=args.apps_path,
        output_path=args.output_path,
        num_bugs=args.num_bugs,
        seed=args.seed
    ) 