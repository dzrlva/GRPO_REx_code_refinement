#!/usr/bin/env python3
"""
GRPO-compatible reward model wrapper.
Imports core reward logic from reward_model.py.
"""

from typing import List, Optional
from reward_model import HybridRewardModel, ExecutionRewardModel, NeuralRewardModel


class GRPORewardModel:
    """
    Batch interface for GRPO training pipeline.
    Wraps HybridRewardModel with __call__ accepting
    lists of completions.
    """

    def __init__(
        self,
        reward_model: Optional[HybridRewardModel] = None,
    ):
        self.reward_model = reward_model or HybridRewardModel()

    def __call__(
        self,
        prompts: Optional[List[str]] = None,
        completions: Optional[List[str]] = None,
        reference_codes: Optional[List[str]] = None,
        test_cases: Optional[List[List[str]]] = None,
    ) -> List[float]:
        """Batch reward computation for GRPO."""
        if completions is None:
            return []

        rewards = []
        for i, completion in enumerate(completions):
            problem = prompts[i] if prompts else ""
            reference = reference_codes[i] if reference_codes else None
            tests = test_cases[i] if test_cases else []

            reward = self.reward_model.get_reward(
                problem=problem,
                solution=completion,
                tests=tests,
                reference_solution=reference,
            )
            rewards.append(reward)

        return rewards

    @staticmethod
    def from_pretrained(model_path: str, **kwargs) -> "GRPORewardModel":
        """Compatibility with HuggingFace trainer interface."""
        return GRPORewardModel()
