import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional, Any, Union
import json

# Import custom modules
from models.reward_model import CodeRefinementRewardModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate code refinement models")
    
    # Model arguments
    parser.add_argument("--models", type=str, nargs="+", required=True,
                       help="Paths to model checkpoints to evaluate")
    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                       help="Human-readable names for the models (defaults to directory names)")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the test dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    
    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to generate per model")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Disable visualization of results")
    
    return parser.parse_args()

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the test dataset.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        DataFrame containing the test dataset
    """
    logger.info(f"Loading dataset from {dataset_path}")
    return pd.read_parquet(dataset_path)

def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and its tokenizer.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate_refinements(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem_desc: str,
    buggy_code: str,
    num_samples: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024
) -> List[str]:
    """
    Generate code refinements using the model.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        problem_desc: Description of the problem
        buggy_code: Buggy code to refine
        num_samples: Number of refinements to generate
        temperature: Temperature for generation
        top_p: Top-p for generation
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        List of generated refinements
    """
    prompt = f"Problem description: {problem_desc}\n\nBuggy code:\n{buggy_code}\n\nRefined code:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length - max_new_tokens
    ).to(model.device)
    
    # Generate refinements
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract only the generated part
    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    
    # Decode the generated refinements
    refinements = []
    for ids in gen_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        refinements.append(text)
        
    return refinements

def evaluate_refinements(
    refinements: List[str],
    reference_code: str,
    test_cases: List[str]
) -> Dict[str, List[Dict[str, float]]]:
    """
    Evaluate generated refinements against the reference solution.
    
    Args:
        refinements: List of generated refinements
        reference_code: Reference/canonical solution
        test_cases: List of test cases to run against the refinements
        
    Returns:
        Dictionary with evaluation metrics for each refinement
    """
    reward_model = CodeRefinementRewardModel()
    
    results = []
    for refinement in refinements:
        # Calculate reward
        reward_info = reward_model.compute_reward(
            generated_code=refinement,
            reference_code=reference_code,
            test_cases=test_cases
        )
        results.append(reward_info)
        
    return results

def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: pd.DataFrame,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Evaluate a model on the test dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Test dataset
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    all_results = []
    
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Evaluating examples"):
        problem_desc = row["problem_description"]
        buggy_code = row["buggy_solution"]
        reference_code = row["canonical_solution"]
        test_cases = row["tests"]
        
        # Generate refinements
        refinements = generate_refinements(
            model=model,
            tokenizer=tokenizer,
            problem_desc=problem_desc,
            buggy_code=buggy_code,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens
        )
        
        # Evaluate refinements
        evaluation_results = evaluate_refinements(
            refinements=refinements,
            reference_code=reference_code,
            test_cases=test_cases
        )
        
        # Store results
        result_entry = {
            "task_id": row["task_id"],
            "dataset": row["dataset"],
            "problem_description": problem_desc,
            "buggy_solution": buggy_code,
            "canonical_solution": reference_code,
            "refinements": refinements,
            "metrics": evaluation_results
        }
        all_results.append(result_entry)
    
    # Calculate aggregate metrics
    pass_rates = {"fully_passed": 0, "partially_passed": 0, "failed": 0}
    total_examples = len(dataset)
    
    for result in all_results:
        # Get the best refinement based on combined reward
        best_refinement_idx = np.argmax([m["combined_reward"] for m in result["metrics"]])
        best_metrics = result["metrics"][best_refinement_idx]
        
        # Determine pass status
        passed_tests = best_metrics["passed_tests"]
        total_tests = best_metrics["total_tests"]
        
        if passed_tests == total_tests:
            pass_rates["fully_passed"] += 1
        elif passed_tests > 0:
            pass_rates["partially_passed"] += 1
        else:
            pass_rates["failed"] += 1
    
    # Convert to percentages
    for key in pass_rates:
        pass_rates[key] = (pass_rates[key] / total_examples) * 100
    
    return {
        "detailed_results": all_results,
        "pass_rates": pass_rates,
        "num_examples": total_examples
    }

def plot_model_comparison(model_results: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Plot comparison of model performance.
    
    Args:
        model_results: Dictionary with results for each model
        output_dir: Directory to save the plots
    """
    # Extract pass rates for each model
    model_names = list(model_results.keys())
    fully_passed = [results["pass_rates"]["fully_passed"] for results in model_results.values()]
    partially_passed = [results["pass_rates"]["partially_passed"] for results in model_results.values()]
    failed = [results["pass_rates"]["failed"] for results in model_results.values()]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax.bar(x - width, fully_passed, width, label='Fully Passed', color='green')
    ax.bar(x, partially_passed, width, label='Partially Passed', color='orange')
    ax.bar(x + width, failed, width, label='Failed', color='red')
    
    ax.set_ylabel('Percentage')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()

def plot_model_comparison_radar_custom_scale(model_results: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Plot radar chart with custom scale for model performance comparison.
    
    Args:
        model_results: Dictionary with results for each model
        output_dir: Directory to save the plots
    """
    # Extract dataset-specific pass rates
    datasets = []
    for model_name, results in model_results.items():
        dataset_metrics = {}
        
        # Group results by dataset
        for result in results["detailed_results"]:
            dataset = result["dataset"]
            if dataset not in dataset_metrics:
                dataset_metrics[dataset] = {"fully_passed": 0, "partially_passed": 0, "failed": 0, "total": 0}
            
            # Get the best refinement based on combined reward
            best_refinement_idx = np.argmax([m["combined_reward"] for m in result["metrics"]])
            best_metrics = result["metrics"][best_refinement_idx]
            
            # Determine pass status
            passed_tests = best_metrics["passed_tests"]
            total_tests = best_metrics["total_tests"]
            
            dataset_metrics[dataset]["total"] += 1
            
            if passed_tests == total_tests:
                dataset_metrics[dataset]["fully_passed"] += 1
            elif passed_tests > 0:
                dataset_metrics[dataset]["partially_passed"] += 1
            else:
                dataset_metrics[dataset]["failed"] += 1
        
        # Convert to percentages
        processed_metrics = {}
        for dataset, metrics in dataset_metrics.items():
            processed_metrics[dataset] = {
                "fully_passed": (metrics["fully_passed"] / metrics["total"]) * 100,
                "partially_passed": (metrics["partially_passed"] / metrics["total"]) * 100,
                "failed": (metrics["failed"] / metrics["total"]) * 100
            }
        
        datasets.append({
            "name": model_name,
            "metrics": processed_metrics
        })
    
    # Extract all unique dataset names
    all_datasets = set()
    for data in datasets:
        all_datasets.update(data["metrics"].keys())
    all_datasets = sorted(list(all_datasets))
    
    # Create radar chart
    categories = []
    for dataset in all_datasets:
        categories.extend([
            f"{dataset} ✔ Fully Passed",
            f"{dataset} ⚠ Partially Passed",
            f"{dataset} ✖ Failed"
        ])
    
    # Create dataframe with pass@1 values
    data_values = {}
    for data in datasets:
        values = []
        for dataset in all_datasets:
            if dataset in data["metrics"]:
                values.extend([
                    data["metrics"][dataset]["fully_passed"],
                    data["metrics"][dataset]["partially_passed"],
                    data["metrics"][dataset]["failed"]
                ])
            else:
                values.extend([0, 0, 0])
        data_values[data["name"]] = values
    
    # Close the loop for the radar chart
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 1.5)
    ax.set_rlabel_position(0)
    
    # Plot each model
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))
    for i, (model_name, values) in enumerate(data_values.items()):
        values_with_closure = values + [values[0]]
        angles_with_closure = angles
        
        ax.plot(angles_with_closure, values_with_closure, label=model_name, color=colors[i], linewidth=2)
        ax.fill(angles_with_closure, values_with_closure, color=colors[i], alpha=0.25)
        
        # Add value labels
        for j, value in enumerate(values):
            ax.text(angles[j], value, f"{int(value)}", ha='center', va='center', fontsize=10)
    
    # Set labels and legend
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # Set radial limits
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    
    # Add grid and legend
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.title("Model Performance by Dataset", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_radar.png"))
    plt.close()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Generate model names if not provided
    if args.model_names is None:
        args.model_names = [os.path.basename(path.rstrip("/")) for path in args.models]
    
    # Ensure we have a name for each model
    if len(args.model_names) < len(args.models):
        for i in range(len(args.model_names), len(args.models)):
            args.model_names.append(f"Model {i+1}")
    
    # Evaluate each model
    model_results = {}
    for model_path, model_name in zip(args.models, args.model_names):
        logger.info(f"Evaluating model: {model_name}")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Evaluate model
        results = evaluate_model(model, tokenizer, dataset, args)
        
        # Print summary statistics
        logger.info(f"Model: {model_name}")
        logger.info(f"Fully passed: {results['pass_rates']['fully_passed']:.2f}%")
        logger.info(f"Partially passed: {results['pass_rates']['partially_passed']:.2f}%")
        logger.info(f"Failed: {results['pass_rates']['failed']:.2f}%")
        
        # Store results
        model_results[model_name] = results
        
        # Save detailed results
        results_path = os.path.join(args.output_dir, f"{model_name}_results.json")
        with open(results_path, "w") as f:
            # Convert numpy values to Python types
            json_results = {
                "pass_rates": results["pass_rates"],
                "num_examples": results["num_examples"]
            }
            json.dump(json_results, f, indent=2)
    
    # Save aggregated results
    all_results_path = os.path.join(args.output_dir, "all_model_results.json")
    with open(all_results_path, "w") as f:
        aggregated_results = {
            model_name: {
                "pass_rates": results["pass_rates"],
                "num_examples": results["num_examples"]
            }
            for model_name, results in model_results.items()
        }
        json.dump(aggregated_results, f, indent=2)
    
    # Create visualizations
    if not args.no_visualize:
        logger.info("Creating visualizations")
        plot_model_comparison(model_results, args.output_dir)
        plot_model_comparison_radar_custom_scale(model_results, args.output_dir)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 