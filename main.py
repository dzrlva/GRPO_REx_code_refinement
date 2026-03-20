#!/usr/bin/env python3
"""
Main entry point for TGPR tree search.

Runs Thompson Sampling-guided tree search for code refinement
using Qwen2.5-7B-Instruct as the policy model.
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from refinement_tree import RefinementTree, TreeConfig, CodeProblem
from reward_model import ExecutionRewardModel, NeuralRewardModel, HybridRewardModel
from code_generator import QwenCodeGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tgpr_run.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_POLICY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REWARD_MODEL = "Qwen/Qwen2.5-1.5B"


def load_problem(problem_path: str) -> CodeProblem:
    """Load a code problem from a JSON file."""
    with open(problem_path, "r") as f:
        data = json.load(f)

    return CodeProblem(
        problem_id=data.get("problem_id", "unknown"),
        prompt=data.get("prompt", ""),
        tests=data.get("tests", []),
        reference_solution=data.get("reference_solution", None),
        buggy_solution=data.get("buggy_solution", None),
    )


def save_results(results: Dict[str, Any], output_dir: str, problem_id: str) -> None:
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{problem_id}_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def parse_arguments():
    """Parse command line arguments matching paper hyperparameters."""
    parser = argparse.ArgumentParser(
        description="TGPR: Thompson Sampling-guided tree search for code refinement"
    )

    parser.add_argument(
        "--problem_path", type=str, required=True,
        help="Path to the problem JSON file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_POLICY_MODEL,
        help="Path to policy model (default: Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--reward_model_path", type=str, default=DEFAULT_REWARD_MODEL,
        help="Path to reward model (default: Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=20,
        help="Maximum tree search iterations",
    )
    parser.add_argument(
        "--max_depth", type=int, default=5,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--branching_factor", type=int, default=4,
        help="Children per node",
    )
    parser.add_argument(
        "--exploration_coefficient", type=float, default=2.0,
        help="Thompson Sampling coefficient C",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--save_tree", action="store_true",
        help="Save full refinement tree in results",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    problem = load_problem(args.problem_path)
    logger.info(f"Loaded problem: {problem.problem_id}")

    code_generator = QwenCodeGenerator(
        model_name_or_path=args.model_path,
        temperature=args.temperature,
    )

    neural_reward = NeuralRewardModel(model_path=args.reward_model_path)
    reward_model = HybridRewardModel(
        execution_model=ExecutionRewardModel(),
        neural_model=neural_reward,
    )

    config = TreeConfig(
        max_depth=args.max_depth,
        branching_factor=args.branching_factor,
        ts_coefficient=args.exploration_coefficient,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
    )

    tree = RefinementTree(
        model=code_generator.model,
        tokenizer=code_generator.tokenizer,
        reward_model=reward_model,
        config=config,
    )

    best_program, metrics = tree.search(problem)

    results = {
        "problem_id": problem.problem_id,
        "best_code": best_program.code,
        "best_reward": best_program.reward,
        "config": {
            "model_path": args.model_path,
            "reward_model_path": args.reward_model_path,
            "max_depth": config.max_depth,
            "branching_factor": config.branching_factor,
            "ts_coefficient": config.ts_coefficient,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        },
        **metrics,
    }

    if not args.save_tree:
        results.pop("tree", None)

    save_results(results, args.output_dir, problem.problem_id)

    print(f"\n{'='*80}")
    print(f"Best solution (reward: {best_program.reward:.4f}):")
    print(f"{'='*80}")
    print(best_program.code)
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
