import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import re
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_humaneval_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the HumanEval dataset from the given path.
    
    Args:
        data_path: Path to the HumanEval dataset
        
    Returns:
        DataFrame containing the HumanEval dataset
    """
    logger.info(f"Loading HumanEval dataset from {data_path}")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    processed_data = []
    for task_id, task in data.items():
        entry = {
            "task_id": task_id,
            "dataset": "humaneval",
            "problem_description": task["prompt"],
            "canonical_solution": task["canonical_solution"],
            "tests": [task["test"]] if isinstance(task["test"], str) else task["test"],
            "entry_point": task.get("entry_point", None),
        }
        processed_data.append(entry)
    
    return pd.DataFrame(processed_data)

def load_mbpp_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the MBPP dataset from the given path.
    
    Args:
        data_path: Path to the MBPP dataset
        
    Returns:
        DataFrame containing the MBPP dataset
    """
    logger.info(f"Loading MBPP dataset from {data_path}")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    processed_data = []
    for task in data:
        # Skip entries without necessary fields
        if not all(k in task for k in ["task_id", "text", "code", "test_list"]):
            continue
            
        entry = {
            "task_id": f"mbpp_{task['task_id']}",
            "dataset": "mbpp",
            "problem_description": task["text"],
            "canonical_solution": task["code"],
            "tests": task["test_list"],
        }
        processed_data.append(entry)
    
    return pd.DataFrame(processed_data)

def load_apps_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the APPS dataset from the given path.
    
    Args:
        data_path: Path to the APPS dataset
        
    Returns:
        DataFrame containing the APPS dataset
    """
    logger.info(f"Loading APPS dataset from {data_path}")
    
    processed_data = []
    
    # APPS dataset is organized into directories by problem ID
    for problem_dir in os.listdir(data_path):
        problem_path = os.path.join(data_path, problem_dir)
        
        if not os.path.isdir(problem_path):
            continue
            
        # Read problem description
        try:
            with open(os.path.join(problem_path, "question.txt"), "r", encoding="utf-8") as f:
                problem_description = f.read().strip()
                
            # Read test cases
            test_cases = []
            test_dir = os.path.join(problem_path, "test")
            
            if os.path.exists(test_dir):
                for test_file in os.listdir(test_dir):
                    if test_file.startswith("input"):
                        input_path = os.path.join(test_dir, test_file)
                        output_number = test_file.split("input")[1].split(".")[0]
                        output_path = os.path.join(test_dir, f"output{output_number}.txt")
                        
                        if os.path.exists(output_path):
                            with open(input_path, "r", encoding="utf-8") as f_in:
                                input_text = f_in.read().strip()
                            with open(output_path, "r", encoding="utf-8") as f_out:
                                output_text = f_out.read().strip()
                                
                            test_cases.append({
                                "input": input_text,
                                "output": output_text
                            })
            
            # Read solutions if available
            solutions = []
            solutions_path = os.path.join(problem_path, "solutions")
            
            if os.path.exists(solutions_path):
                for solution_file in os.listdir(solutions_path):
                    if solution_file.endswith(".py"):
                        solution_path = os.path.join(solutions_path, solution_file)
                        with open(solution_path, "r", encoding="utf-8") as f:
                            solutions.append(f.read().strip())
            
            # Skip if no solutions available
            if not solutions:
                continue
                
            # Convert test cases to test strings
            formatted_tests = []
            for i, test in enumerate(test_cases):
                # Create a test function that checks the output
                test_str = f"""def test_{i}():\n    assert str(solution({test['input']})).strip() == '''{test['output']}'''.strip()"""
                formatted_tests.append(test_str)
            
            if not formatted_tests:
                continue
                
            entry = {
                "task_id": f"apps_{problem_dir}",
                "dataset": "apps",
                "problem_description": problem_description,
                "canonical_solution": solutions[0],  # Use first solution as canonical
                "tests": formatted_tests,
            }
            processed_data.append(entry)
            
        except Exception as e:
            logger.warning(f"Error processing APPS problem {problem_dir}: {e}")
            continue
    
    return pd.DataFrame(processed_data)

def generate_buggy_solutions(df: pd.DataFrame, num_bugs_per_solution: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Generate buggy versions of the canonical solutions.
    
    Args:
        df: DataFrame containing the dataset
        num_bugs_per_solution: Number of buggy versions to generate per solution
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with buggy solutions added
    """
    logger.info(f"Generating {num_bugs_per_solution} buggy versions per solution")
    
    np.random.seed(seed)
    
    # List of common bug types to introduce
    bug_types = [
        "off_by_one",
        "wrong_operator",
        "missing_check",
        "wrong_variable",
        "incorrect_loop_condition",
        "incorrect_return",
        "syntax_error",
        "missing_initialization",
    ]
    
    def introduce_bug(code: str, bug_type: str) -> str:
        """Introduce a specific type of bug into the code."""
        lines = code.split('\n')
        if len(lines) <= 1:
            return code  # Too short to modify
            
        # Select a random non-empty line (avoiding imports and docstrings)
        valid_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped and 
                not stripped.startswith('#') and 
                not stripped.startswith('import') and 
                not stripped.startswith('from') and 
                '"""' not in stripped and 
                "'''" not in stripped):
                valid_lines.append(i)
                
        if not valid_lines:
            return code  # No valid lines to modify
            
        line_idx = np.random.choice(valid_lines)
        line = lines[line_idx]
        
        # Apply bug based on type
        if bug_type == "off_by_one":
            # Find a number and increase or decrease it by 1
            number_pattern = r'\b\d+\b'
            matches = list(re.finditer(number_pattern, line))
            if matches:
                match = np.random.choice(matches)
                start, end = match.span()
                num = int(line[start:end])
                operation = np.random.choice([-1, 1])
                new_num = max(0, num + operation)  # Avoid negative numbers
                lines[line_idx] = line[:start] + str(new_num) + line[end:]
        
        elif bug_type == "wrong_operator":
            # Replace an operator with another
            operators = ['+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', 'and', 'or']
            for op in operators:
                if op in line:
                    replacements = [o for o in operators if o != op]
                    new_op = np.random.choice(replacements)
                    lines[line_idx] = line.replace(op, new_op, 1)
                    break
        
        elif bug_type == "missing_check":
            # Remove an if condition
            if 'if ' in line:
                indentation = len(line) - len(line.lstrip())
                next_line_idx = line_idx + 1
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent > indentation:  # It's an indented block
                        lines[line_idx] = ' ' * next_indent + next_line.lstrip()
                        lines.pop(next_line_idx)
        
        elif bug_type == "wrong_variable":
            # Replace a variable with another
            variable_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            matches = list(re.finditer(variable_pattern, line))
            if len(matches) >= 2:
                var_idxs = np.random.choice(len(matches), size=2, replace=False)
                var1 = matches[var_idxs[0]].group()
                var2 = matches[var_idxs[1]].group()
                if var1 != var2 and not (var1 in ['if', 'for', 'while', 'def', 'class', 'return'] or 
                                         var2 in ['if', 'for', 'while', 'def', 'class', 'return']):
                    lines[line_idx] = line.replace(var1, '_temp_var_', 1)
                    lines[line_idx] = lines[line_idx].replace(var2, var1, 1)
                    lines[line_idx] = lines[line_idx].replace('_temp_var_', var2, 1)
        
        elif bug_type == "incorrect_loop_condition":
            # Modify a loop condition
            if any(keyword in line for keyword in ['for ', 'while ']):
                if 'for ' in line and ' in ' in line:
                    before, after = line.split(' in ', 1)
                    if ':' in after:
                        collection, rest = after.split(':', 1)
                        if 'range(' in collection:
                            # Modify range parameters
                            range_params = collection.split('range(')[1].split(')')[0]
                            params = [p.strip() for p in range_params.split(',')]
                            if len(params) == 1:  # range(end)
                                try:
                                    end = int(params[0])
                                    new_end = max(1, end + np.random.choice([-2, 2]))
                                    lines[line_idx] = before + ' in range(' + str(new_end) + '):' + rest
                                except ValueError:
                                    pass
                            elif len(params) >= 2:  # range(start, end) or range(start, end, step)
                                try:
                                    start = int(params[0])
                                    end = int(params[1])
                                    new_start = max(0, start + np.random.choice([-1, 1]))
                                    new_end = max(new_start + 1, end + np.random.choice([-2, 2]))
                                    if len(params) == 2:
                                        lines[line_idx] = before + ' in range(' + str(new_start) + ', ' + str(new_end) + '):' + rest
                                    else:
                                        lines[line_idx] = before + ' in range(' + str(new_start) + ', ' + str(new_end) + ', ' + params[2] + '):' + rest
                                except ValueError:
                                    pass
                elif 'while ' in line and ':' in line:
                    condition = line.split('while ')[1].split(':')[0]
                    if '==' in condition:
                        lines[line_idx] = line.replace('==', '!=', 1)
                    elif '!=' in condition:
                        lines[line_idx] = line.replace('!=', '==', 1)
                    elif '<' in condition:
                        lines[line_idx] = line.replace('<', '<=', 1)
                    elif '<=' in condition:
                        lines[line_idx] = line.replace('<=', '<', 1)
                    elif '>' in condition:
                        lines[line_idx] = line.replace('>', '>=', 1)
                    elif '>=' in condition:
                        lines[line_idx] = line.replace('>=', '>', 1)
        
        elif bug_type == "incorrect_return":
            # Modify a return statement
            if 'return ' in line:
                return_value = line.split('return ')[1]
                if return_value.strip() in ['True', 'False']:
                    lines[line_idx] = line.replace('True', 'False') if 'True' in return_value else line.replace('False', 'True')
                elif re.search(r'\d+', return_value):
                    # Modify a number in the return statement
                    number_pattern = r'\b\d+\b'
                    matches = list(re.finditer(number_pattern, return_value))
                    if matches:
                        match = np.random.choice(matches)
                        start, end = match.span()
                        full_start = line.find(return_value) + start
                        full_end = line.find(return_value) + end
                        num = int(line[full_start:full_end])
                        operation = np.random.choice([-1, 1])
                        new_num = max(0, num + operation)
                        lines[line_idx] = line[:full_start] + str(new_num) + line[full_end:]
        
        elif bug_type == "syntax_error":
            # Introduce a syntax error
            error_types = [
                (':', ''),  # Remove colon
                (')', ''),  # Remove closing parenthesis
                (']', ''),  # Remove closing bracket
                ('else:', 'else')  # Remove colon after else
            ]
            for error_from, error_to in error_types:
                if error_from in line:
                    lines[line_idx] = line.replace(error_from, error_to, 1)
                    break
        
        elif bug_type == "missing_initialization":
            # Find a variable initialization and remove it
            if '=' in line and not any(op in line for op in ['==', '>=', '<=', '!=']):
                indentation = len(line) - len(line.lstrip())
                # Remove the line if it's a variable initialization
                if not any(keyword in line for keyword in ['if', 'for', 'while', 'def', 'class']):
                    lines.pop(line_idx)
        
        return '\n'.join(lines)
    
    results = []
    
    for idx, row in df.iterrows():
        code = row['canonical_solution']
        canonical_id = row['task_id']
        
        # Add the original row
        results.append(row.to_dict())
        
        # Generate buggy versions
        for i in range(num_bugs_per_solution):
            bug_row = row.copy()
            bug_type = np.random.choice(bug_types)
            buggy_code = introduce_bug(code, bug_type)
            
            # Only add if the bug was successfully introduced (code is different)
            if buggy_code != code:
                bug_row = bug_row.to_dict()
                bug_row['task_id'] = f"{canonical_id}_bug_{i+1}"
                bug_row['buggy_solution'] = buggy_code
                results.append(bug_row)
    
    return pd.DataFrame(results)

def get_combined_dataset(
    humaneval_path: Optional[str] = None,
    mbpp_path: Optional[str] = None,
    apps_path: Optional[str] = None,
    output_path: Optional[str] = None,
    num_bugs_per_solution: int = 3,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Combine HumanEval, MBPP, and APPS datasets and prepare them for training.
    
    Args:
        humaneval_path: Path to the HumanEval dataset
        mbpp_path: Path to the MBPP dataset
        apps_path: Path to the APPS dataset
        output_path: Path to save the combined dataset
        num_bugs_per_solution: Number of buggy versions to generate per solution
        seed: Random seed for reproducibility
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        
    Returns:
        Dictionary containing train, validation, and test DataFrames
    """
    combined_data = pd.DataFrame()
    
    # Load datasets if paths are provided
    if humaneval_path and os.path.exists(humaneval_path):
        humaneval_df = load_humaneval_dataset(humaneval_path)
        combined_data = pd.concat([combined_data, humaneval_df], ignore_index=True)
    
    if mbpp_path and os.path.exists(mbpp_path):
        mbpp_df = load_mbpp_dataset(mbpp_path)
        combined_data = pd.concat([combined_data, mbpp_df], ignore_index=True)
    
    if apps_path and os.path.exists(apps_path):
        apps_df = load_apps_dataset(apps_path)
        combined_data = pd.concat([combined_data, apps_df], ignore_index=True)
    
    if combined_data.empty:
        logger.warning("No data loaded from the provided paths")
        return {"train": pd.DataFrame(), "validation": pd.DataFrame(), "test": pd.DataFrame()}
    
    # Generate buggy solutions
    data_with_bugs = generate_buggy_solutions(combined_data, num_bugs_per_solution, seed)
    
    # Split data into train, validation, and test sets
    np.random.seed(seed)
    
    # Group by task_id to ensure related examples stay together
    task_ids = combined_data['task_id'].unique()
    np.random.shuffle(task_ids)
    
    n_tasks = len(task_ids)
    train_size = int(n_tasks * train_ratio)
    val_size = int(n_tasks * val_ratio)
    
    train_ids = task_ids[:train_size]
    val_ids = task_ids[train_size:train_size+val_size]
    test_ids = task_ids[train_size+val_size:]
    
    # Extract base IDs without bug suffixes
    base_train_ids = set()
    for task_id in train_ids:
        base_id = task_id.split('_bug_')[0] if '_bug_' in task_id else task_id
        base_train_ids.add(base_id)
    
    base_val_ids = set()
    for task_id in val_ids:
        base_id = task_id.split('_bug_')[0] if '_bug_' in task_id else task_id
        base_val_ids.add(base_id)
    
    base_test_ids = set()
    for task_id in test_ids:
        base_id = task_id.split('_bug_')[0] if '_bug_' in task_id else task_id
        base_test_ids.add(base_id)
    
    # Split the data with bugs
    train_df = data_with_bugs[data_with_bugs['task_id'].apply(
        lambda x: x.split('_bug_')[0] if '_bug_' in x else x).isin(base_train_ids)]
    
    val_df = data_with_bugs[data_with_bugs['task_id'].apply(
        lambda x: x.split('_bug_')[0] if '_bug_' in x else x).isin(base_val_ids)]
    
    test_df = data_with_bugs[data_with_bugs['task_id'].apply(
        lambda x: x.split('_bug_')[0] if '_bug_' in x else x).isin(base_test_ids)]
    
    # Save combined dataset if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        train_df.to_parquet(os.path.join(output_path, "train.parquet"))
        val_df.to_parquet(os.path.join(output_path, "validation.parquet"))
        test_df.to_parquet(os.path.join(output_path, "test.parquet"))
        
        logger.info(f"Saved combined dataset to {output_path}")
        logger.info(f"Train set: {len(train_df)} examples")
        logger.info(f"Validation set: {len(val_df)} examples")
        logger.info(f"Test set: {len(test_df)} examples")
    
    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }

def convert_dataset_to_hf_format(df: pd.DataFrame) -> Dataset:
    """
    Convert a pandas DataFrame to a HuggingFace Dataset.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        HuggingFace Dataset
    """
    # Create a formatted dataset with the required fields
    formatted_data = []
    
    for _, row in df.iterrows():
        if 'buggy_solution' in row and pd.notna(row['buggy_solution']):
            # Training example with buggy solution
            formatted_data.append({
                "task_id": row['task_id'],
                "dataset": row['dataset'],
                "problem_description": row['problem_description'],
                "buggy_solution": row['buggy_solution'],
                "canonical_solution": row['canonical_solution'],
                "tests": row['tests'],
            })
        else:
            # Create buggy solution by introducing random bugs
            buggy_solution = introduce_bug(row['canonical_solution'], np.random.choice([
                "off_by_one", "wrong_operator", "missing_check", "wrong_variable",
                "incorrect_loop_condition", "incorrect_return", "syntax_error", "missing_initialization"
            ]))
            
            formatted_data.append({
                "task_id": row['task_id'],
                "dataset": row['dataset'],
                "problem_description": row['problem_description'],
                "buggy_solution": buggy_solution,
                "canonical_solution": row['canonical_solution'],
                "tests": row['tests'],
            })
    
    return Dataset.from_pandas(pd.DataFrame(formatted_data))

def prepare_dataset_for_training(
    data_path: str,
    output_path: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Prepare a dataset for training from a combined dataset.
    
    Args:
        data_path: Path to the combined dataset (directory with train.parquet, validation.parquet, test.parquet)
        output_path: Path to save the prepared dataset
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict containing train, validation, and test Datasets
    """
    # Load datasets
    train_df = pd.read_parquet(os.path.join(data_path, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data_path, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(data_path, "test.parquet"))
    
    # Convert to HuggingFace Datasets
    train_dataset = convert_dataset_to_hf_format(train_df)
    val_dataset = convert_dataset_to_hf_format(val_df)
    test_dataset = convert_dataset_to_hf_format(test_df)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # Save if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        dataset_dict.save_to_disk(output_path)
        logger.info(f"Saved prepared dataset to {output_path}")
    
    return dataset_dict

def format_dataset_for_training(
    dataset: Union[Dataset, pd.DataFrame],
    prompt_template: str = "Problem description: {problem_description}\n\nBuggy code:\n{buggy_solution}\n\nRefined code:"
) -> Dataset:
    """
    Format a dataset for training with prompt templates.
    
    Args:
        dataset: Dataset to format
        prompt_template: Template for the prompt
        
    Returns:
        Formatted dataset
    """
    # Convert pandas DataFrame to HuggingFace Dataset if needed
    if isinstance(dataset, pd.DataFrame):
        dataset = Dataset.from_pandas(dataset)
    
    def apply_formatting(example):
        prompt = prompt_template.format(
            problem_description=example["problem_description"],
            buggy_solution=example["buggy_solution"]
        )
        
        return {
            "prompt": prompt,
            "completion": example["canonical_solution"],
            "original_task_id": example["task_id"],
            "dataset": example["dataset"],
            "tests": example["tests"],
            "buggy_solution": example["buggy_solution"],
            "canonical_solution": example["canonical_solution"],
        }
    
    return dataset.map(apply_formatting)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess datasets for code refinement")
    parser.add_argument("--humaneval_path", type=str, help="Path to HumanEval dataset")
    parser.add_argument("--mbpp_path", type=str, help="Path to MBPP dataset")
    parser.add_argument("--apps_path", type=str, help="Path to APPS dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the combined dataset")
    parser.add_argument("--num_bugs", type=int, default=3, help="Number of buggy versions to generate per solution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data to use for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for testing")
    
    args = parser.parse_args()
    
    # Validate ratio sum equals 1
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    
    # Get combined dataset
    combined_data = get_combined_dataset(
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
    
    # Prepare dataset for training
    dataset_dict = prepare_dataset_for_training(
        data_path=args.output_path,
        output_path=os.path.join(args.output_path, "hf_dataset"),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    logger.info("Dataset preprocessing complete!") 