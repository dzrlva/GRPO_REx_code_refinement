import os
import re
import json
import random
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_humaneval_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the HumanEval dataset.
    
    Args:
        data_path: Path to the HumanEval dataset JSON file
        
    Returns:
        DataFrame containing the HumanEval problems
    """
    try:
        logger.info(f"Loading HumanEval dataset from {data_path}")
        
        # Load the dataset from JSON
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to a list of dictionaries for easier manipulation
        problems = []
        for task_id, problem in data.items():
            entry = {
                'task_id': task_id,
                'problem_description': problem['prompt'],
                'canonical_solution': problem['canonical_solution'],
                'entry_point': problem['entry_point'],
                'tests': [test.strip() for test in problem['test'].split('\n') if test.strip()],
                'dataset': 'humaneval'
            }
            problems.append(entry)
        
        logger.info(f"Loaded {len(problems)} problems from HumanEval dataset")
        return pd.DataFrame(problems)
    
    except Exception as e:
        logger.error(f"Error loading HumanEval dataset: {e}")
        return pd.DataFrame()

def load_mbpp_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the MBPP dataset.
    
    Args:
        data_path: Path to the MBPP dataset JSON file
        
    Returns:
        DataFrame containing the MBPP problems
    """
    try:
        logger.info(f"Loading MBPP dataset from {data_path}")
        
        # Load the dataset from JSON
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to a list of dictionaries for easier manipulation
        problems = []
        for problem in data:
            # Skip problems without test cases
            if not problem.get('test_list'):
                continue
                
            entry = {
                'task_id': f"mbpp_{problem['task_id']}",
                'problem_description': problem['text'],
                'canonical_solution': problem['code'],
                'entry_point': None,  # MBPP doesn't specify entry points
                'tests': problem['test_list'],
                'dataset': 'mbpp'
            }
            problems.append(entry)
        
        logger.info(f"Loaded {len(problems)} problems from MBPP dataset")
        return pd.DataFrame(problems)
    
    except Exception as e:
        logger.error(f"Error loading MBPP dataset: {e}")
        return pd.DataFrame()

def load_apps_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the APPS dataset.
    
    Args:
        data_path: Path to the APPS dataset directory
        
    Returns:
        DataFrame containing the APPS problems
    """
    try:
        logger.info(f"Loading APPS dataset from {data_path}")
        
        problems = []
        problem_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        
        for problem_id in tqdm(problem_dirs, desc="Loading APPS problems"):
            problem_dir = os.path.join(data_path, problem_id)
            
            # Read problem description
            try:
                with open(os.path.join(problem_dir, 'question.txt'), 'r', encoding='utf-8') as f:
                    problem_description = f.read().strip()
            except:
                continue
                
            # Read solutions
            solutions_file = os.path.join(problem_dir, 'solutions.json')
            if not os.path.exists(solutions_file):
                continue
                
            with open(solutions_file, 'r', encoding='utf-8') as f:
                solutions = json.load(f)
                
            if not solutions:
                continue
                
            # Use the first solution as canonical
            canonical_solution = solutions[0]
            
            # Read test cases
            test_file = os.path.join(problem_dir, 'input_output.json')
            tests = []
            
            if os.path.exists(test_file):
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                    
                    for i, (input_data, output_data) in enumerate(zip(test_data["inputs"], test_data["outputs"])):
                        # Create a simple test function that checks input against expected output
                        test_fn = f"def test_{i}(solution_func):\n    assert solution_func({input_data}) == {output_data}"
                        tests.append(test_fn)
                except:
                    pass
            
            entry = {
                'task_id': f"apps_{problem_id}",
                'problem_description': problem_description,
                'canonical_solution': canonical_solution,
                'entry_point': None,  # Extract function name if possible
                'tests': tests,
                'dataset': 'apps'
            }
            
            # Try to extract the function name from the solution
            function_match = re.search(r'def\s+(\w+)\s*\(', canonical_solution)
            if function_match:
                entry['entry_point'] = function_match.group(1)
            
            problems.append(entry)
        
        logger.info(f"Loaded {len(problems)} problems from APPS dataset")
        return pd.DataFrame(problems)
    
    except Exception as e:
        logger.error(f"Error loading APPS dataset: {e}")
        return pd.DataFrame()

def generate_buggy_solutions(
    df: pd.DataFrame, 
    num_bugs_per_solution: int = 3, 
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate buggy versions of the canonical solutions.
    
    Args:
        df: DataFrame containing the problems with canonical solutions
        num_bugs_per_solution: Number of buggy versions to generate per solution
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with original problems and additional buggy solutions
    """
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Generating {num_bugs_per_solution} buggy versions per solution")
    
    # Bug types with corresponding transformation functions
    bug_types = [
        "variable_renaming",
        "operator_change",
        "off_by_one",
        "conditional_change",
        "return_change",
        "loop_change"
    ]
    
    # Function to introduce variable renaming bugs
    def introduce_variable_renaming(code: str) -> str:
        lines = code.split('\n')
        variables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()', code)
        variables = [v for v in variables if v not in ['def', 'if', 'else', 'elif', 'for', 'while', 
                                                      'return', 'True', 'False', 'None', 'and', 'or', 
                                                      'not', 'in', 'is', 'lambda', 'import', 'from', 
                                                      'as', 'class', 'try', 'except', 'finally',
                                                      'with', 'continue', 'break', 'pass', 'raise']]
        
        if not variables:
            return code
            
        # Choose a random variable to rename
        var_to_rename = random.choice(variables)
        new_var_name = var_to_rename + '_x'
        
        # Rename only some occurrences to create inconsistency
        occurrences = [i for i, line in enumerate(lines) if re.search(r'\b' + var_to_rename + r'\b', line)]
        if not occurrences:
            return code
            
        # Select which occurrences to rename (at least one, but not all)
        num_to_rename = random.randint(1, max(1, len(occurrences) - 1))
        rename_indices = random.sample(occurrences, num_to_rename)
        
        for i in rename_indices:
            lines[i] = re.sub(r'\b' + var_to_rename + r'\b', new_var_name, lines[i])
            
        return '\n'.join(lines)
    
    # Function to introduce operator change bugs
    def introduce_operator_change(code: str) -> str:
        operators_map = {
            '+': '-', '-': '+', '*': '/', '/': '*',
            '==': '!=', '!=': '==', '<': '>', '>': '<',
            '<=': '>=', '>=': '<=', 'and': 'or', 'or': 'and'
        }
        
        lines = code.split('\n')
        operator_pattern = r'(\+|\-|\*|\/|==|!=|<=|>=|<|>|\band\b|\bor\b)'
        operator_lines = [i for i, line in enumerate(lines) if re.search(operator_pattern, line)]
        
        if not operator_lines:
            return code
            
        line_idx = random.choice(operator_lines)
        line = lines[line_idx]
        
        # Find all operators in the line
        operators = re.findall(operator_pattern, line)
        if not operators:
            return code
            
        # Choose a random operator to change
        op_to_change = random.choice(operators)
        if op_to_change in operators_map:
            # Replace just the first occurrence to make it subtle
            lines[line_idx] = line.replace(op_to_change, operators_map[op_to_change], 1)
            
        return '\n'.join(lines)
    
    # Function to introduce off-by-one bugs
    def introduce_off_by_one(code: str) -> str:
        lines = code.split('\n')
        number_lines = [i for i, line in enumerate(lines) if re.search(r'\b\d+\b', line)]
        
        if not number_lines:
            return code
            
        line_idx = random.choice(number_lines)
        line = lines[line_idx]
        
        # Find all numbers in the line
        numbers = re.findall(r'\b(\d+)\b', line)
        if not numbers:
            return code
            
        # Choose a random number to change
        num_to_change = random.choice(numbers)
        
        # Randomly add or subtract 1
        change = random.choice([1, -1])
        new_num = str(max(0, int(num_to_change) + change))
        
        # Replace just the first occurrence to make it subtle
        lines[line_idx] = line.replace(num_to_change, new_num, 1)
        
        return '\n'.join(lines)
    
    # Function to introduce conditional change bugs
    def introduce_conditional_change(code: str) -> str:
        lines = code.split('\n')
        if_lines = [i for i, line in enumerate(lines) if re.search(r'\bif\b|\belif\b', line)]
        
        if not if_lines:
            return code
            
        line_idx = random.choice(if_lines)
        line = lines[line_idx]
        
        # Simple approach: negate the condition
        if '==' in line:
            lines[line_idx] = line.replace('==', '!=', 1)
        elif '!=' in line:
            lines[line_idx] = line.replace('!=', '==', 1)
        elif '>' in line and not '>=' in line:
            lines[line_idx] = line.replace('>', '<=', 1)
        elif '<' in line and not '<=' in line:
            lines[line_idx] = line.replace('<', '>=', 1)
        elif '>=' in line:
            lines[line_idx] = line.replace('>=', '<', 1)
        elif '<=' in line:
            lines[line_idx] = line.replace('<=', '>', 1)
        elif ' not ' in line:
            lines[line_idx] = line.replace(' not ', ' ', 1)
        else:
            # Add a "not" at the beginning of the condition
            condition_match = re.search(r'(if|elif)\s+(.*?):', line)
            if condition_match:
                condition = condition_match.group(2)
                lines[line_idx] = line.replace(condition, f"not ({condition})", 1)
            
        return '\n'.join(lines)
    
    # Function to introduce return change bugs
    def introduce_return_change(code: str) -> str:
        lines = code.split('\n')
        return_lines = [i for i, line in enumerate(lines) if re.search(r'\breturn\b', line)]
        
        if not return_lines:
            return code
            
        line_idx = random.choice(return_lines)
        line = lines[line_idx]
        
        # Simple modifications to the return statement
        if 'return True' in line:
            lines[line_idx] = line.replace('return True', 'return False', 1)
        elif 'return False' in line:
            lines[line_idx] = line.replace('return False', 'return True', 1)
        elif 'return None' in line:
            lines[line_idx] = line.replace('return None', 'return 0', 1)
        elif 'return 0' in line:
            lines[line_idx] = line.replace('return 0', 'return 1', 1)
        elif 'return 1' in line:
            lines[line_idx] = line.replace('return 1', 'return 0', 1)
        elif 'return []' in line:
            lines[line_idx] = line.replace('return []', 'return [0]', 1)
        else:
            # For other return statements, try adding a simple operation
            return_match = re.search(r'return\s+(.*)', line)
            if return_match:
                return_val = return_match.group(1)
                # Randomly choose a modification
                mod_type = random.choice(['add', 'subtract', 'negate'])
                if mod_type == 'add' and not return_val.strip() in ['True', 'False', 'None']:
                    lines[line_idx] = line.replace(f"return {return_val}", f"return {return_val} + 1", 1)
                elif mod_type == 'subtract' and not return_val.strip() in ['True', 'False', 'None']:
                    lines[line_idx] = line.replace(f"return {return_val}", f"return {return_val} - 1", 1)
                elif mod_type == 'negate':
                    lines[line_idx] = line.replace(f"return {return_val}", f"return not {return_val}", 1)
            
        return '\n'.join(lines)
    
    # Function to introduce loop change bugs
    def introduce_loop_change(code: str) -> str:
        lines = code.split('\n')
        loop_lines = [i for i, line in enumerate(lines) if re.search(r'\bfor\b|\bwhile\b', line)]
        
        if not loop_lines:
            return code
            
        line_idx = random.choice(loop_lines)
        line = lines[line_idx]
        
        if 'range' in line:
            # Modify range parameters
            range_match = re.search(r'range\s*\(\s*(\d+)?\s*,?\s*(\d+)?\s*,?\s*(\d+)?\s*\)', line)
            if range_match:
                groups = range_match.groups()
                if groups[0] is not None and groups[1] is not None and groups[2] is not None:
                    # range(start, end, step)
                    start, end, step = groups
                    # Randomly choose what to modify
                    mod_type = random.choice(['start', 'end', 'step'])
                    if mod_type == 'start':
                        new_start = str(int(start) + random.choice([-1, 1]))
                        lines[line_idx] = line.replace(f"range({start}, {end}, {step})", 
                                                     f"range({new_start}, {end}, {step})", 1)
                    elif mod_type == 'end':
                        new_end = str(int(end) + random.choice([-1, 1]))
                        lines[line_idx] = line.replace(f"range({start}, {end}, {step})", 
                                                     f"range({start}, {new_end}, {step})", 1)
                    else:  # step
                        new_step = str(int(step) + random.choice([-1, 1]))
                        if new_step != '0':  # Avoid step=0
                            lines[line_idx] = line.replace(f"range({start}, {end}, {step})", 
                                                         f"range({start}, {end}, {new_step})", 1)
                elif groups[0] is not None and groups[1] is not None:
                    # range(start, end)
                    start, end = groups[0], groups[1]
                    mod_type = random.choice(['start', 'end'])
                    if mod_type == 'start':
                        new_start = str(int(start) + random.choice([-1, 1]))
                        lines[line_idx] = line.replace(f"range({start}, {end})", 
                                                     f"range({new_start}, {end})", 1)
                    else:  # end
                        new_end = str(int(end) + random.choice([-1, 1]))
                        lines[line_idx] = line.replace(f"range({start}, {end})", 
                                                     f"range({start}, {new_end})", 1)
                elif groups[0] is not None:
                    # range(end)
                    end = groups[0]
                    new_end = str(int(end) + random.choice([-1, 1]))
                    lines[line_idx] = line.replace(f"range({end})", f"range({new_end})", 1)
        elif 'while' in line:
            # Modify while condition similar to conditional_change
            return introduce_conditional_change(code)
            
        return '\n'.join(lines)
    
    # Map bug types to functions
    bug_introducers = {
        "variable_renaming": introduce_variable_renaming,
        "operator_change": introduce_operator_change,
        "off_by_one": introduce_off_by_one,
        "conditional_change": introduce_conditional_change,
        "return_change": introduce_return_change,
        "loop_change": introduce_loop_change
    }
    
    # Create the new dataframe with buggy solutions
    all_problems = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating buggy solutions"):
        # First add the original problem with canonical solution
        all_problems.append(row.to_dict())
        
        # Generate buggy versions
        for i in range(num_bugs_per_solution):
            # Create a copy of the original problem
            buggy_problem = row.to_dict()
            
            # Choose a random bug type
            bug_type = random.choice(bug_types)
            
            # Introduce the bug
            introducer_func = bug_introducers[bug_type]
            buggy_solution = introducer_func(row['canonical_solution'])
            
            # Ensure the bug was actually introduced (code changed)
            if buggy_solution == row['canonical_solution']:
                # Try a different bug type if the first one didn't work
                alt_bug_types = [bt for bt in bug_types if bt != bug_type]
                if alt_bug_types:
                    bug_type = random.choice(alt_bug_types)
                    introducer_func = bug_introducers[bug_type]
                    buggy_solution = introducer_func(row['canonical_solution'])
            
            # Only add if the bug was successfully introduced
            if buggy_solution != row['canonical_solution']:
                buggy_problem['task_id'] = f"{row['task_id']}_buggy_{i+1}"
                buggy_problem['buggy_solution'] = buggy_solution
                buggy_problem['bug_type'] = bug_type
                all_problems.append(buggy_problem)
    
    result_df = pd.DataFrame(all_problems)
    
    # For canonical solutions, make the buggy and canonical solutions the same
    mask = ~result_df['task_id'].str.contains('_buggy_')
    result_df.loc[mask, 'buggy_solution'] = result_df.loc[mask, 'canonical_solution']
    result_df.loc[mask, 'bug_type'] = 'none'
    
    logger.info(f"Generated dataset with {len(result_df)} problems (original + buggy)")
    return result_df

def validate_solution(solution: str) -> bool:
    """Check if a solution is valid Python code."""
    try:
        compile(solution, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def get_combined_dataset(
    humaneval_path: Optional[str] = None,
    mbpp_path: Optional[str] = None,
    apps_path: Optional[str] = None,
    output_path: str = 'data/refined_code_dataset',
    num_bugs_per_solution: int = 3,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Combine datasets, generate buggy solutions, and split into train/val/test.
    
    Args:
        humaneval_path: Path to HumanEval dataset (optional)
        mbpp_path: Path to MBPP dataset (optional)
        apps_path: Path to APPS dataset (optional)
        output_path: Path to save the combined dataset
        num_bugs_per_solution: Number of buggy versions to generate per solution
        seed: Random seed for reproducibility
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load datasets
    dataframes = []
    
    if humaneval_path:
        humaneval_df = load_humaneval_dataset(humaneval_path)
        if not humaneval_df.empty:
            dataframes.append(humaneval_df)
    
    if mbpp_path:
        mbpp_df = load_mbpp_dataset(mbpp_path)
        if not mbpp_df.empty:
            dataframes.append(mbpp_df)
    
    if apps_path:
        apps_df = load_apps_dataset(apps_path)
        if not apps_df.empty:
            dataframes.append(apps_df)
    
    if not dataframes:
        raise ValueError("No valid datasets were loaded")
    
    # Combine datasets
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataset has {len(combined_df)} problems")
    
    # Filter out invalid canonical solutions
    valid_mask = combined_df['canonical_solution'].apply(validate_solution)
    combined_df = combined_df[valid_mask].reset_index(drop=True)
    logger.info(f"After filtering invalid solutions: {len(combined_df)} problems")
    
    # Generate buggy solutions
    augmented_df = generate_buggy_solutions(
        combined_df, 
        num_bugs_per_solution=num_bugs_per_solution,
        seed=seed
    )
    
    # Split into train/val/test based on original problem IDs, not augmented ones
    # Extract base task IDs (without _buggy_X suffix)
    augmented_df['base_task_id'] = augmented_df['task_id'].apply(
        lambda x: x.split('_buggy_')[0] if '_buggy_' in x else x
    )
    
    # Get unique base task IDs
    base_task_ids = augmented_df['base_task_id'].unique()
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(base_task_ids)
    
    train_size = int(len(base_task_ids) * train_ratio)
    val_size = int(len(base_task_ids) * val_ratio)
    
    train_ids = set(base_task_ids[:train_size])
    val_ids = set(base_task_ids[train_size:train_size + val_size])
    test_ids = set(base_task_ids[train_size + val_size:])
    
    # Create train/val/test dataframes
    train_df = augmented_df[augmented_df['base_task_id'].isin(train_ids)]
    val_df = augmented_df[augmented_df['base_task_id'].isin(val_ids)]
    test_df = augmented_df[augmented_df['base_task_id'].isin(test_ids)]
    
    # Clean up and save
    train_df = train_df.drop(columns=['base_task_id'])
    val_df = val_df.drop(columns=['base_task_id'])
    test_df = test_df.drop(columns=['base_task_id'])
    
    logger.info(f"Split dataset into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test examples")
    
    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save the dataset
    dataset_dict.save_to_disk(output_path)
    logger.info(f"Saved combined dataset to {output_path}")
    
    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description='Prepare code datasets for refinement training')
    parser.add_argument('--humaneval_path', help='Path to HumanEval dataset')
    parser.add_argument('--mbpp_path', help='Path to MBPP dataset')
    parser.add_argument('--apps_path', help='Path to APPS dataset')
    parser.add_argument('--output_path', default='data/refined_code_dataset', help='Path to save combined dataset')
    parser.add_argument('--num_bugs', type=int, default=3, help='Number of buggy versions per solution')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test data ratio')
    
    args = parser.parse_args()
    
    # Validate at least one dataset path is provided
    if not (args.humaneval_path or args.mbpp_path or args.apps_path):
        parser.error("At least one dataset path must be provided")
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        parser.error("Train, validation, and test ratios must sum to 1.0")
    
    # Generate the dataset
    get_combined_dataset(
        humaneval_path=args.humaneval_path,
        mbpp_path=args.mbpp_path,
        apps_path=args.apps_path,
        output_path=args.output_path,
        num_bugs_per_solution=args.num_bugs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

if __name__ == "__main__":
    main() 