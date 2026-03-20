#!/usr/bin/env python3
"""
Reward models for TGPR framework.

Implements the hybrid reward function from the paper:
    R(ρ) = w_CB · CodeBLEU(ρ, ρ_c) + w_TP · |T_p(ρ)| / |T|

where w_CB = 0.5, w_TP = 0.5, and CodeBLEU components are:
    n-gram match: 0.25, syntax tree: 0.25, semantic data-flow: 0.5
"""

import os
import tempfile
import subprocess
import time
import re
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CODEBLEU_WEIGHT = 0.5
TEST_PASS_RATE_WEIGHT = 0.5
TIMEOUT_PER_TEST = 10
REWARD_MIN = 0.0
REWARD_MAX = 1.0
CODEBLEU_NGRAM_WEIGHT = 0.25
CODEBLEU_SYNTAX_WEIGHT = 0.25
CODEBLEU_SEMANTIC_WEIGHT = 0.50


def compute_codebleu(
    candidate: str,
    reference: str,
    lang: str = "python",
    ngram_weight: float = CODEBLEU_NGRAM_WEIGHT,
    syntax_weight: float = CODEBLEU_SYNTAX_WEIGHT,
    semantic_weight: float = CODEBLEU_SEMANTIC_WEIGHT,
) -> float:
    """
    Compute CodeBLEU between candidate and reference.
    Components: n-gram 0.25, syntax 0.25, semantic 0.50.
    Returns float in [0, 1].
    """
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(
            references=[[reference]],
            predictions=[candidate],
            lang=lang,
            weights=(ngram_weight, syntax_weight, semantic_weight, 0.0),
            tokenizer=None,
        )
        return float(np.clip(result["codebleu"], REWARD_MIN, REWARD_MAX))
    except ImportError:
        logger.warning("codebleu package not installed, returning 0.0")
        return 0.0
    except Exception as e:
        logger.error(f"CodeBLEU computation failed: {e}")
        return 0.0


def compute_test_pass_rate(
    solution: str,
    tests: List[str],
    timeout: int = TIMEOUT_PER_TEST,
) -> Tuple[float, int, int]:
    """
    Execute solution against test cases.
    Returns (pass_rate, passed, total) where pass_rate in [0, 1].
    Timeout: 10 seconds per test case.
    """
    if not solution.strip() or not tests:
        return 0.0, 0, len(tests) if tests else 0

    with tempfile.TemporaryDirectory() as tempdir:
        solution_file = os.path.join(tempdir, "solution.py")
        with open(solution_file, "w") as f:
            f.write(solution)

        test_file = os.path.join(tempdir, "test_solution.py")
        with open(test_file, "w") as f:
            f.write("import solution\nimport sys\n\n")
            for i, test in enumerate(tests):
                f.write(f"def test_{i}():\n")
                for line in test.strip().split("\n"):
                    f.write(f"    {line}\n")
                f.write("\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    all_tests = [")
            for i in range(len(tests)):
                f.write(f"test_{i}, ")
            f.write("]\n")
            f.write("    passed = 0\n")
            f.write("    total = len(all_tests)\n")
            f.write("    for test_func in all_tests:\n")
            f.write("        try:\n")
            f.write("            test_func()\n")
            f.write("            passed += 1\n")
            f.write("        except Exception as e:\n")
            f.write("            print(f'Test {test_func.__name__} failed: {str(e)}')\n")
            f.write("    print(f'{passed}/{total} tests passed')\n")
            f.write("    sys.exit(0 if passed == total else 1)\n")

        total_timeout = timeout * len(tests)
        try:
            result = subprocess.run(
                ["python", test_file],
                capture_output=True,
                text=True,
                timeout=total_timeout,
            )
            match = re.search(r"(\d+)/(\d+) tests passed", result.stdout)
            if match:
                passed = int(match.group(1))
                total = int(match.group(2))
                pass_rate = passed / total if total > 0 else 0.0
                logger.info(f"Tests: {passed}/{total} passed ({pass_rate:.4f})")
                return pass_rate, passed, total
            else:
                logger.warning(f"Failed to parse output: {result.stdout}")
                return 0.0, 0, len(tests)
        except subprocess.TimeoutExpired:
            logger.warning(f"Execution timed out after {total_timeout}s")
            return 0.0, 0, len(tests)
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return 0.0, 0, len(tests)


class ExecutionRewardModel:
    """Computes |T_p(ρ)| / |T| by executing code against test cases."""

    def __init__(self, timeout: int = TIMEOUT_PER_TEST):
        self.timeout = timeout

    def get_reward(self, problem: str, solution: str, tests: List[str], **kwargs) -> float:
        if not solution.strip():
            return 0.0
        pass_rate, _, _ = compute_test_pass_rate(solution, tests, timeout=self.timeout)
        return pass_rate


class NeuralRewardModel:
    """
    Fine-tuned Qwen2.5-1.5B predicting R(ρ) directly.
    Trained: 3 epochs, AdamW, LR 1e-5, dropout 0.1.
    Pearson r > 0.85 on held-out data.
    """

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        logger.info(f"Neural reward model loaded from {model_path}")

    def get_reward(self, problem: str, solution: str, **kwargs) -> float:
        if not solution.strip():
            return 0.0

        prompt = f"Problem:\n{problem}\n\nSolution:\n{solution}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if hasattr(outputs, "logits"):
            if outputs.logits.shape[1] == 1:
                reward = torch.sigmoid(outputs.logits[0, 0]).item()
            else:
                reward = torch.softmax(outputs.logits, dim=1)[0, 1].item()
        else:
            reward = outputs.value.item()

        return float(np.clip(reward, REWARD_MIN, REWARD_MAX))


class HybridRewardModel:
    """
    R(ρ) = 0.5 · CodeBLEU(ρ, ρ_c) + 0.5 · |T_p(ρ)| / |T|
    Yields R(ρ) ∈ [0, 1].
    """

    def __init__(
        self,
        execution_model: Optional[ExecutionRewardModel] = None,
        neural_model: Optional[NeuralRewardModel] = None,
        codebleu_weight: float = CODEBLEU_WEIGHT,
        test_pass_weight: float = TEST_PASS_RATE_WEIGHT,
    ):
        self.execution_model = execution_model or ExecutionRewardModel()
        self.neural_model = neural_model
        self.codebleu_weight = codebleu_weight
        self.test_pass_weight = test_pass_weight

    def get_reward(
        self,
        problem: str,
        solution: str,
        tests: List[str],
        reference_solution: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Compute R(ρ) = w_CB · CodeBLEU + w_TP · TestPassRate.
        Falls back to test-only if no reference or neural model.
        """
        test_pass_rate = self.execution_model.get_reward(problem, solution, tests)

        if reference_solution is not None:
            codebleu_score = compute_codebleu(solution, reference_solution)
        elif self.neural_model is not None:
            codebleu_score = self.neural_model.get_reward(problem, solution)
        else:
            logger.warning("No reference or neural model; test-only reward")
            return float(np.clip(test_pass_rate, REWARD_MIN, REWARD_MAX))

        reward = (
            self.codebleu_weight * codebleu_score
            + self.test_pass_weight * test_pass_rate
        )
        reward = float(np.clip(reward, REWARD_MIN, REWARD_MAX))

        logger.info(
            f"R(ρ)={reward:.4f} "
            f"(CB:{codebleu_score:.4f}×{self.codebleu_weight}, "
            f"TP:{test_pass_rate:.4f}×{self.test_pass_weight})"
        )
        return reward
