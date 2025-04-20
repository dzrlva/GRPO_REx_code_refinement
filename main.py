#!/usr/bin/env python3
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from refinement_tree import RefinementTree, RefinementTreeConfig, Program, CodeProblem
from reward_model import ExecutionRewardModel, NeuralRewardModel, HybridRewardModel
from code_generator import QwenCodeGenerator, CodeLlamaGenerator, DeepseekCoderGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("refinement_tree_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_problem(problem_path: str) -> CodeProblem:
    """Load a code problem from a JSON file."""
    with open(problem_path, 'r') as f:
        problem_data = json.load(f)
    
    return CodeProblem(
        id=problem_data.get('id', 'unknown'),
        prompt=problem_data.get('prompt', ''),
        tests=problem_data.get('tests', []),
        solutions=problem_data.get('solutions', [])
    )

def save_results(results: Dict[str, Any], output_dir: str, problem_id: str) -> None:
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{problem_id}_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run refinement tree exploration for code generation')
    
    parser.add_argument('--problem_path', type=str, required=True,
                        help='Path to the problem JSON file')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--model_type', type=str, default='qwen',
                        choices=['qwen', 'codellama', 'deepseek'],
                        help='Type of code generation model to use')
    
    parser.add_argument('--model_path', type=str,
                        help='Path or name of the model to use for code generation')
    
    parser.add_argument('--reward_model_path', type=str,
                        help='Path to neural reward model (if None, uses execution-only reward)')
    
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='Maximum number of iterations for the REx algorithm')
    
    parser.add_argument('--max_depth', type=int, default=5,
                        help='Maximum depth of the refinement tree')
    
    parser.add_argument('--exploration_coefficient', type=float, default=0.5,
                        help='Exploration coefficient for UCB calculation')
    
    parser.add_argument('--min_reward_threshold', type=float, default=0.8,
                        help='Minimum reward threshold to consider a problem solved')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for code generation')
    
    parser.add_argument('--save_tree', action='store_true',
                        help='Save the full refinement tree in the results')
    
    return parser.parse_args()

def create_code_generator(model_type: str, model_path: Optional[str], temperature: float):
    """Create a code generator based on model type and path."""
    if model_type == 'qwen':
        model_path = model_path or "Qwen/Qwen2.5-1.5B-Coder-Instruct"
        return QwenCodeGenerator(model_name_or_path=model_path, temperature=temperature)
    elif model_type == 'codellama':
        model_path = model_path or "codellama/CodeLlama-7b-Instruct-hf"
        return CodeLlamaGenerator(model_name_or_path=model_path, temperature=temperature)
    elif model_type == 'deepseek':
        model_path = model_path or "deepseek-ai/deepseek-coder-1.3b-instruct"
        return DeepseekCoderGenerator(model_name_or_path=model_path, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load the problem
    problem = load_problem(args.problem_path)
    logger.info(f"Loaded problem: {problem.id}")
    
    # Create the code generator
    code_generator = create_code_generator(
        model_type=args.model_type,
        model_path=args.model_path,
        temperature=args.temperature
    )
    
    # Create the reward model
    if args.reward_model_path:
        # Use hybrid reward model with both execution and neural models
        neural_reward = NeuralRewardModel(model_path=args.reward_model_path)
        reward_model = HybridRewardModel(
            execution_model=ExecutionRewardModel(),
            neural_model=neural_reward
        )
    else:
        # Use execution-only reward model
        reward_model = ExecutionRewardModel()
    
    # Configure the refinement tree
    config = RefinementTreeConfig(
        max_depth=args.max_depth,
        exploration_coefficient=args.exploration_coefficient,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        min_reward_threshold=args.min_reward_threshold
    )
    
    # Create the refinement tree
    refinement_tree = RefinementTree(
        problem=problem,
        code_generator=code_generator,
        reward_model=reward_model,
        config=config
    )
    
    # Run the REx algorithm
    results = refinement_tree.run_REx()
    
    # Add additional information to results
    results['problem_id'] = problem.id
    results['config'] = config.__dict__
    results['model_type'] = args.model_type
    results['model_path'] = args.model_path or "default"
    
    # Save the full tree if requested
    if not args.save_tree:
        results.pop('tree', None)
    
    # Save results
    save_results(results, args.output_dir, problem.id)
    
    # Print the best solution
    if results['best_program']:
        print("\n" + "="*80)
        print(f"Best solution (reward: {results['best_reward']:.4f}):")
        print("="*80)
        print(results['best_program'])
        print("="*80)
    else:
        print("\nNo solution found.")

if __name__ == "__main__":
    main() 