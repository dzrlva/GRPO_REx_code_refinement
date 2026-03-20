#!/usr/bin/env python3
"""
Ablation study for TGPR framework.

Compares three configurations (Section 4, Appendix):
    - GRPO (Base): standard GRPO with sparse binary reward
    - GRPO+DR: GRPO with dense hybrid reward, no tree search
    - TGPR (Full): dense reward + Thompson Sampling tree search

Evaluates on MBPP and APPS (pass@1, averaged over 5 seeds).
"""

import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Any
from datasets import load_from_disk

from evaluate import (
    load_model,
    load_dataset,
    evaluate_single_problem,
    pass_at_k,
    bootstrap_ci,
    SEEDS,
    TEMP_PASS1,
    NUM_SAMPLES,
)
from reward_model import (
    ExecutionRewardModel,
    HybridRewardModel,
    NeuralRewardModel,
    CODEBLEU_WEIGHT,
    TEST_PASS_RATE_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ABLATION_CONFIGS = {
    "GRPO (Base)": {
        "description": "Standard GRPO with sparse binary reward",
        "use_dense_reward": False,
        "use_tree_search": False,
    },
    "GRPO+DR": {
        "description": "GRPO with dense hybrid reward, no tree search",
        "use_dense_reward": True,
        "use_tree_search": False,
    },
    "TGPR (Full)": {
        "description": "Dense reward + Thompson Sampling tree search",
        "use_dense_reward": True,
        "use_tree_search": True,
    },
}

BENCHMARKS = ["mbpp", "apps"]


def get_reward_model(config: Dict) -> Any:
    """Create reward model based on ablation config."""
    if config["use_dense_reward"]:
        return HybridRewardModel(
            execution_model=ExecutionRewardModel(),
            codebleu_weight=CODEBLEU_WEIGHT,
            test_pass_weight=TEST_PASS_RATE_WEIGHT,
        )
    else:
        return ExecutionRewardModel()


def evaluate_config(
    model_path: str,
    dataset_path: str,
    benchmark: str,
    config_name: str,
    config: Dict,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Evaluate a single ablation configuration on one benchmark."""
    model, tokenizer = load_model(model_path)
    dataset = load_dataset(dataset_path)

    logger.info(f"Evaluating {config_name} on {benchmark} ({len(dataset)} problems)")

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        problem_scores = []
        for problem in dataset:
            result = evaluate_single_problem(
                model, tokenizer, problem, num_samples, max_new_tokens=1024,
            )
            problem_scores.append(result["pass@1"])

        seed_mean = np.mean(problem_scores) * 100
        seed_results.append(seed_mean)

    mean_pass1 = np.mean(seed_results)
    ci = bootstrap_ci(seed_results)

    return {
        "config": config_name,
        "benchmark": benchmark,
        "pass@1_mean": round(mean_pass1, 1),
        "pass@1_ci": [round(ci[0], 1), round(ci[1], 1)],
        "seed_results": [round(s, 1) for s in seed_results],
    }


def run_ablation(
    model_paths: Dict[str, str],
    dataset_paths: Dict[str, str],
    output_dir: str = "ablation_results",
) -> Dict[str, Any]:
    """
    Run full ablation study.

    Args:
        model_paths: {"GRPO (Base)": path, "GRPO+DR": path, "TGPR (Full)": path}
        dataset_paths: {"mbpp": path, "apps": path}
        output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for config_name, config in ABLATION_CONFIGS.items():
        model_path = model_paths[config_name]
        for benchmark in BENCHMARKS:
            dataset_path = dataset_paths[benchmark]

            result = evaluate_config(
                model_path=model_path,
                dataset_path=dataset_path,
                benchmark=benchmark,
                config_name=config_name,
                config=config,
            )
            all_results.append(result)

            logger.info(
                f"{config_name} | {benchmark} | "
                f"pass@1={result['pass@1_mean']}% "
                f"CI={result['pass@1_ci']}"
            )

    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    compute_attribution(all_results, output_dir)
    return all_results


def compute_attribution(results: List[Dict], output_dir: str):
    """
    Decompose total improvement into component contributions.

    Dense reward: GRPO+DR - GRPO Base
    Tree search: TGPR Full - GRPO+DR
    Total: TGPR Full - GRPO Base
    """
    attribution = {}

    for benchmark in BENCHMARKS:
        bench_results = {r["config"]: r for r in results if r["benchmark"] == benchmark}

        base = bench_results["GRPO (Base)"]["pass@1_mean"]
        dr = bench_results["GRPO+DR"]["pass@1_mean"]
        full = bench_results["TGPR (Full)"]["pass@1_mean"]

        total_gain = full - base
        dr_gain = dr - base
        tree_gain = full - dr

        if total_gain > 0:
            dr_pct = round(dr_gain / total_gain * 100)
            tree_pct = round(tree_gain / total_gain * 100)
        else:
            dr_pct = 0
            tree_pct = 0

        attribution[benchmark] = {
            "total_gain": round(total_gain, 1),
            "dense_reward_gain": round(dr_gain, 1),
            "dense_reward_pct": dr_pct,
            "tree_search_gain": round(tree_gain, 1),
            "tree_search_pct": tree_pct,
        }

        logger.info(
            f"{benchmark}: total +{total_gain:.1f} pp | "
            f"dense reward +{dr_gain:.1f} ({dr_pct}%) | "
            f"tree search +{tree_gain:.1f} ({tree_pct}%)"
        )

    attr_path = os.path.join(output_dir, "attribution.json")
    with open(attr_path, "w") as f:
        json.dump(attribution, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TGPR Ablation Study")
    parser.add_argument("--grpo_base_path", type=str, required=True)
    parser.add_argument("--grpo_dr_path", type=str, required=True)
    parser.add_argument("--tgpr_full_path", type=str, required=True)
    parser.add_argument("--mbpp_dataset", type=str, required=True)
    parser.add_argument("--apps_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="ablation_results")

    args = parser.parse_args()

    model_paths = {
        "GRPO (Base)": args.grpo_base_path,
        "GRPO+DR": args.grpo_dr_path,
        "TGPR (Full)": args.tgpr_full_path,
    }
    dataset_paths = {
        "mbpp": args.mbpp_dataset,
        "apps": args.apps_dataset,
    }

    run_ablation(model_paths, dataset_paths, args.output_dir)
