#!/usr/bin/env python3
"""
Evaluation script for TGPR framework.

Evaluation protocol from the paper:
    - pass@1 (temperature 0.2) and pass@10 (temperature 0.8)
    - 200 samples per problem for unbiased pass@10 estimator
    - 5 random seeds: 42, 123, 456, 789, 999
    - 95% bootstrap confidence intervals
    - 10s timeout per test case
"""

import os
import argparse
import torch
import numpy as np
import json
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from reward_model import HybridRewardModel, ExecutionRewardModel, compute_codebleu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 999]
TEMP_PASS1 = 0.2
TEMP_PASS10 = 0.8
TOP_P = 0.95
TOP_K = 50
NUM_SAMPLES = 200
TIMEOUT_PER_TEST = 10


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k (Chen et al., 2021).
    n = total samples, c = correct samples, k = k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """95% bootstrap confidence interval across seeds."""
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def parse_args():
    parser = argparse.ArgumentParser(description="TGPR Evaluation")

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Display name for the model",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to test dataset (JSON or parquet)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="humaneval",
        choices=["humaneval", "mbpp", "apps", "codeforces", "livecodebench"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_results",
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples", type=int, default=NUM_SAMPLES,
        help="Samples per problem for pass@10 estimator (default: 200)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="Maximum tokens to generate",
    )

    return parser.parse_args()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSON or parquet."""
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")
    else:
        with open(path, "r") as f:
            return json.load(f)


def load_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model in bfloat16 with auto device mapping."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    n: int,
    temperature: float,
    max_new_tokens: int = 1024,
) -> List[str]:
    """Generate n samples with given temperature, top_p=0.95, top_k=50."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length - max_new_tokens,
    ).to(model.device)

    samples = []
    batch_size = min(n, 10)

    for start in range(0, n, batch_size):
        current_batch = min(batch_size, n - start)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=TOP_P,
                top_k=TOP_K,
                num_return_sequences=current_batch,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
        for ids in gen_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            samples.append(text)

    return samples


def build_prompt(problem: Dict[str, Any]) -> str:
    """Build refinement prompt from problem dict."""
    desc = problem.get("prompt", problem.get("problem_description", ""))
    buggy = problem.get("buggy_solution", "")

    if buggy:
        return (
            f"You are an expert programmer. Fix the bugs in this Python code.\n\n"
            f"Problem:\n{desc}\n\n"
            f"Buggy code:\n```python\n{buggy}\n```\n\n"
            f"Fixed code:\n```python\n"
        )
    return (
        f"You are an expert programmer. Write Python code to solve this problem.\n\n"
        f"Problem:\n{desc}\n\n"
        f"Solution:\n```python\n"
    )


def evaluate_single_problem(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: Dict[str, Any],
    num_samples: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Evaluate one problem: generate samples at both temperatures, compute pass@k."""
    prompt = build_prompt(problem)
    tests = problem.get("tests", [])
    reference = problem.get("canonical_solution", problem.get("reference_solution", ""))
    task_id = problem.get("task_id", problem.get("problem_id", "unknown"))
    reward_model = ExecutionRewardModel(timeout=TIMEOUT_PER_TEST)

    results = {"task_id": task_id}

    for metric_name, temp, k in [("pass@1", TEMP_PASS1, 1), ("pass@10", TEMP_PASS10, 10)]:
        n = num_samples if k > 1 else min(num_samples, 20)
        samples = generate_samples(model, tokenizer, prompt, n, temp, max_new_tokens)

        correct = 0
        for sample in samples:
            reward = reward_model.get_reward(problem="", solution=sample, tests=tests)
            if reward >= 1.0:
                correct += 1

        results[metric_name] = pass_at_k(len(samples), correct, k)
        results[f"{metric_name}_n"] = len(samples)
        results[f"{metric_name}_correct"] = correct

    if reference:
        best_sample = samples[0] if samples else ""
        results["codebleu"] = compute_codebleu(best_sample, reference)

    return results


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    seed: int,
) -> List[Dict[str, Any]]:
    """Evaluate model on full dataset with given seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    results = []
    for problem in tqdm(dataset, desc=f"Seed {seed}"):
        result = evaluate_single_problem(
            model, tokenizer, problem, args.num_samples, args.max_new_tokens,
        )
        results.append(result)

    return results


def aggregate_results(all_seed_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate results across 5 seeds with bootstrap CIs."""
    seed_pass1 = []
    seed_pass10 = []

    for seed_results in all_seed_results:
        p1_values = [r["pass@1"] for r in seed_results]
        p10_values = [r["pass@10"] for r in seed_results]
        seed_pass1.append(np.mean(p1_values) * 100)
        seed_pass10.append(np.mean(p10_values) * 100)

    p1_mean = np.mean(seed_pass1)
    p10_mean = np.mean(seed_pass10)
    p1_ci = bootstrap_ci(seed_pass1)
    p10_ci = bootstrap_ci(seed_pass10)

    return {
        "pass@1_mean": round(p1_mean, 1),
        "pass@1_ci": [round(p1_ci[0], 1), round(p1_ci[1], 1)],
        "pass@10_mean": round(p10_mean, 1),
        "pass@10_ci": [round(p10_ci[0], 1), round(p10_ci[1], 1)],
        "num_seeds": len(all_seed_results),
        "seeds": SEEDS[:len(all_seed_results)],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_path)
    model, tokenizer = load_model(args.model_path)
    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/"))

    logger.info(f"Evaluating {model_name} on {args.benchmark} ({len(dataset)} problems)")
    logger.info(f"Seeds: {SEEDS}, samples per problem: {args.num_samples}")

    all_seed_results = []
    for seed in SEEDS:
        logger.info(f"Running seed {seed}")
        seed_results = evaluate_model(model, tokenizer, dataset, args, seed)
        all_seed_results.append(seed_results)

    summary = aggregate_results(all_seed_results)
    summary["model"] = model_name
    summary["benchmark"] = args.benchmark

    logger.info(f"pass@1: {summary['pass@1_mean']}% CI {summary['pass@1_ci']}")
    logger.info(f"pass@10: {summary['pass@10_mean']}% CI {summary['pass@10_ci']}")

    output_path = os.path.join(args.output_dir, f"{model_name}_{args.benchmark}.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
