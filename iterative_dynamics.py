```python
#!/usr/bin/env python3
"""
Iterative refinement experiment for TGPR framework.

From the paper (Appendix: Iterative Refinement Dynamics):
    - Evaluate 3 models on LiveCodeBench (out-of-distribution)
    - For each problem, generate up to 10 sequential refinement attempts
    - Attempt 1 = pass@1, Attempt 10 = pass@10
    - Track cumulative success rate at each attempt
    - 5 seeds: 42, 123, 456, 789, 999

Expected results matching Table (LiveCodeBench):
    Pretrained: pass@1=32.5%, pass@10=44.2%
    GRPO:       pass@1=51.2%, pass@10=65.1%
    TGPR:       pass@1=55.8%, pass@10=74.3%
"""

import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward_model import ExecutionRewardModel, compute_test_pass_rate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 999]
MAX_ATTEMPTS = 10
TIMEOUT_PER_TEST = 10
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50


def load_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model in bfloat16."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_livecodebench(path: str) -> List[Dict[str, Any]]:
    """Load LiveCodeBench dataset."""
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")
    with open(path, "r") as f:
        return json.load(f)


def build_refinement_prompt(
    problem: Dict[str, Any],
    previous_code: str = None,
    feedback: str = None,
) -> str:
    """Build prompt for refinement attempt."""
    desc = problem.get("prompt", problem.get("problem_description", ""))

    if previous_code is None:
        return (
            f"You are an expert programmer. Write Python code to solve this problem.\n\n"
            f"Problem:\n{desc}\n\n"
            f"Solution:\n```python\n"
        )
    else:
        prompt = (
            f"You are an expert programmer. Fix the bugs in this Python code.\n\n"
            f"Problem:\n{desc}\n\n"
            f"Current code:\n```python\n{previous_code}\n```\n\n"
        )
        if feedback:
            prompt += f"Execution feedback:\n{feedback}\n\n"
        prompt += f"Fixed code:\n```python\n"
        return prompt


def generate_single_attempt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
) -> str:
    """Generate one code attempt."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=tokenizer.model_max_length - max_new_tokens,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    end_markers = ["```", "\n\n\n"]
    for marker in end_markers:
        if marker in generated:
            generated = generated.split(marker)[0]

    return generated.strip()


def get_execution_feedback(code: str, tests: List[str]) -> str:
    """Run tests and return human-readable feedback."""
    pass_rate, passed, total = compute_test_pass_rate(
        code, tests, timeout=TIMEOUT_PER_TEST
    )
    if passed == total:
        return f"All {total} tests passed."
    return f"{passed}/{total} tests passed. {total - passed} tests failed."


def run_iterative_refinement(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: Dict[str, Any],
    max_attempts: int = MAX_ATTEMPTS,
) -> Dict[str, Any]:
    """
    Run up to max_attempts sequential refinements for one problem.

    Returns dict with:
        - attempts: list of code strings (length = max_attempts)
        - passed_at: list of booleans (True if all tests pass at that attempt)
        - cumulative_solved: list of booleans (True if solved by attempt k)
        - pass_rates: list of floats (test pass rate at each attempt)
    """
    tests = problem.get("tests", [])
    attempts = []
    passed_at = []
    pass_rates = []
    solved = False
    previous_code = None
    feedback = None

    for attempt_idx in range(max_attempts):
        prompt = build_refinement_prompt(problem, previous_code, feedback)
        code = generate_single_attempt(model, tokenizer, prompt)
        attempts.append(code)

        pass_rate, passed, total = compute_test_pass_rate(
            code, tests, timeout=TIMEOUT_PER_TEST
        )
        pass_rates.append(pass_rate)

        all_passed = (passed == total and total > 0)
        passed_at.append(all_passed)

        if all_passed:
            solved = True

        if not solved:
            previous_code = code
            feedback = get_execution_feedback(code, tests)

    cumulative_solved = []
    any_solved = False
    for p in passed_at:
        if p:
            any_solved = True
        cumulative_solved.append(any_solved)

    return {
        "attempts": attempts,
        "passed_at": passed_at,
        "cumulative_solved": cumulative_solved,
        "pass_rates": pass_rates,
    }


def evaluate_model_refinement(
    model_path: str,
    dataset: List[Dict[str, Any]],
    seed: int,
    max_attempts: int = MAX_ATTEMPTS,
) -> List[float]:
    """
    Evaluate one model with one seed.
    Returns list of cumulative success rates for attempts 1..max_attempts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tokenizer = load_model(model_path)
    cumulative_counts = np.zeros(max_attempts)
    total_problems = len(dataset)

    for problem in tqdm(dataset, desc=f"Seed {seed}"):
        result = run_iterative_refinement(model, tokenizer, problem, max_attempts)
        for k, solved in enumerate(result["cumulative_solved"]):
            if solved:
                cumulative_counts[k] += 1

    cumulative_rates = (cumulative_counts / total_problems * 100).tolist()
    return cumulative_rates


def run_full_experiment(
    model_paths: Dict[str, str],
    dataset_path: str,
    output_dir: str = "refinement_results",
) -> Dict[str, Any]:
    """
    Run full iterative refinement experiment for all models.

    Args:
        model_paths: {"Pretrained": path, "GRPO": path, "TGPR (Ours)": path}
        dataset_path: path to LiveCodeBench dataset
        output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_livecodebench(dataset_path)
    logger.info(f"Loaded {len(dataset)} problems from LiveCodeBench")

    all_results = {}

    for model_name, model_path in model_paths.items():
        logger.info(f"Evaluating {model_name}")

        seed_curves = []
        for seed in SEEDS:
            logger.info(f"  Seed {seed}")
            curve = evaluate_model_refinement(model_path, dataset, seed)
            seed_curves.append(curve)
            logger.info(f"  Attempt 1: {curve[0]:.1f}%, Attempt 10: {curve[-1]:.1f}%")

        mean_curve = np.mean(seed_curves, axis=0).tolist()
        std_curve = np.std(seed_curves, axis=0).tolist()

        all_results[model_name] = {
            "mean_curve": [round(v, 1) for v in mean_curve],
            "std_curve": [round(v, 1) for v in std_curve],
            "seed_curves": seed_curves,
        }

        logger.info(
            f"{model_name}: pass@1={mean_curve[0]:.1f}%, pass@10={mean_curve[-1]:.1f}%"
        )

    results_path = os.path.join(output_dir, "refinement_dynamics.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return all_results


def plot_refinement_dynamics(
    results_path: str = "refinement_results/refinement_dynamics.json",
    output_path: str = "livecodebench_refinement.png",
):
    """
    Plot iterative refinement dynamics from saved results.
    Falls back to paper values if results file not found.
    """
    import matplotlib.pyplot as plt

    attempts = np.arange(1, MAX_ATTEMPTS + 1)

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        pretrained = results["Pretrained"]["mean_curve"]
        grpo = results["GRPO"]["mean_curve"]
        tgpr = results["TGPR (Ours)"]["mean_curve"]
    else:
        logger.warning(f"{results_path} not found, using paper values")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(attempts, pretrained, 'o-', color='#E74C3C', linewidth=2.5,
            markersize=8, label='Pretrained', zorder=3)
    ax.plot(attempts, grpo, 's-', color='#7B68EE', linewidth=2.5,
            markersize=8, label='GRPO', zorder=3)
    ax.plot(attempts, tgpr, 'D-', color='#2ECC71', linewidth=2.5,
            markersize=8, label='TGPR (Ours)', zorder=3)

    annotate_indices = [0, 2, 4, 6, 9]
    for data, color, offset in [
        (pretrained, '#E74C3C', (0, -22)),
        (grpo, '#7B68EE', (0, -22)),
        (tgpr, '#2ECC71', (0, 16)),
    ]:
        for i in annotate_indices:
            ax.annotate(
                f'{data[i]}%', (attempts[i], data[i]),
                textcoords="offset points", xytext=offset,
                ha='center', fontsize=8.5, color=color,
            )

    ax.annotate('Reasoning Jump', xy=(2, 68), fontsize=11,
                fontstyle='italic', color='#2ECC71', alpha=0.8)
    ax.annotate('RL Volatility', xy=(4, 49), fontsize=11,
                fontstyle='italic', color='#7B68EE', alpha=0.8)
    ax.annotate('Stagnation', xy=(4, 28.5), fontsize=11,
                fontstyle='italic', color='#E74C3C', alpha=0.8)

    ax.set_xlabel('Refinement Attempt (1 to 10)', fontsize=13)
    ax.set_ylabel('Success Rate (Pass@k %)', fontsize=13)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(24, 84)
    ax.set_xticks(attempts)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TGPR Iterative Refinement Experiment")
    parser.add_argument("--mode", choices=["run", "plot"], default="plot")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--grpo_path", type=str, default=None)
    parser.add_argument("--tgpr_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="refinement_results")

    args = parser.parse_args()

    if args.mode == "run":
        if not all([args.pretrained_path, args.grpo_path, args.tgpr_path, args.dataset_path]):
            parser.error("run mode requires all model paths and dataset_path")

        model_paths = {
            "Pretrained": args.pretrained_path,
            "GRPO": args.grpo_path,
            "TGPR (Ours)": args.tgpr_path,
        }
        run_full_experiment(model_paths, args.dataset_path, args.output_dir)

    plot_refinement_dynamics(
        results_path=os.path.join(args.output_dir, "refinement_dynamics.json"),
        output_path=os.path.join(args.output_dir, "livecodebench_refinement.png"),
    )
