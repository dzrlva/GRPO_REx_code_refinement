#!/usr/bin/env python3
"""
GRPO trainer with Thompson Sampling-guided tree search for TGPR.

Implements Group Relative Policy Optimization (Section 3):
    - Group size: 8 trajectories per problem
    - KL coefficient: 0.01
    - Clip range: 0.2
    - GAE λ: 0.95
    - 4 policy update epochs per batch
    - AdamW optimizer, LR 5e-6 with cosine decay
    - Total steps: 8000, warmup: 100
    - bfloat16 precision on 2× A100

Tree search generates `branching_factor` children per node
using Thompson Sampling (Beta distribution) for node selection.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from refinement_tree import RefinementTree, TreeConfig, CodeProblem, Program
from reward_model import HybridRewardModel, ExecutionRewardModel, NeuralRewardModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """
    GRPO + tree search configuration matching paper hyperparameters.
    """
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    reward_model_name: str = "Qwen/Qwen2.5-1.5B"
    output_dir: str = "outputs/tgpr"

    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 100
    total_steps: int = 8000
    gradient_clip: float = 1.0

    effective_batch_size: int = 64
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    group_size: int = 8
    kl_coefficient: float = 0.01
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 1.0
    ppo_epochs: int = 4

    max_depth: int = 5
    branching_factor: int = 4
    ts_coefficient: float = 2.0
    beta_epsilon: float = 1e-6

    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 512

    seed: int = 42


def compute_grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    GRPO advantage: normalize rewards within each group.
    A_i = (r_i - mean(r)) / std(r)
    """
    mean = rewards.mean()
    std = rewards.std()
    if std < 1e-8:
        return torch.zeros_like(rewards)
    return (rewards - mean) / std


def compute_policy_loss(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """
    Clipped surrogate objective.
    L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    """
    ratio = torch.exp(logprobs_new - logprobs_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss.mean()


def compute_kl_penalty(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
) -> torch.Tensor:
    """KL(π_new || π_old) approximation."""
    return (logprobs_old - logprobs_new).mean()


class GRPORefinementTrainer:
    """
    GRPO trainer with Thompson Sampling-guided tree search.

    Training loop:
    1. For each problem, generate `group_size` trajectories via tree search
    2. Compute hybrid rewards R(ρ) = 0.5·CodeBLEU + 0.5·TestPassRate
    3. Compute GRPO advantages (group-normalized)
    4. Update policy with clipped surrogate + KL penalty
    5. Repeat for `total_steps`
    """

    def __init__(
        self,
        config: GRPOConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading policy model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        logger.info(f"Loading reward model: {config.reward_model_name}")
        neural_reward = NeuralRewardModel(model_path=config.reward_model_name)
        self.reward_model = HybridRewardModel(
            execution_model=ExecutionRewardModel(),
            neural_model=neural_reward,
        )

        tree_config = TreeConfig(
            max_depth=config.max_depth,
            branching_factor=config.branching_factor,
            ts_coefficient=config.ts_coefficient,
            beta_epsilon=config.beta_epsilon,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
        )
        self.tree = RefinementTree(
            model=self.model,
            tokenizer=self.tokenizer,
            reward_model=self.reward_model,
            config=tree_config,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )

        os.makedirs(config.output_dir, exist_ok=True)

    def _problem_from_example(self, example: Dict[str, Any]) -> CodeProblem:
        return CodeProblem(
            problem_id=example.get("task_id", "unknown"),
            prompt=example.get("problem_description", example.get("prompt", "")),
            tests=example.get("tests", []),
            reference_solution=example.get("canonical_solution", None),
            buggy_solution=example.get("buggy_solution", None),
        )

    def _generate_group_trajectories(
        self, problem: CodeProblem
    ) -> List[Tuple[str, str, float]]:
        """
        Generate `group_size` trajectories for one problem via tree search.
        Returns list of (prompt, completion, reward).
        """
        trajectories = []

        for _ in range(self.config.group_size):
            best_program, _ = self.tree.search(problem)

            prompt = self.tree._build_prompt(
                problem.buggy_solution or "", problem
            )
            trajectories.append((prompt, best_program.code, best_program.reward))

        return trajectories

    def _get_logprobs(
        self, model: AutoModelForCausalLM, prompt: str, completion: str
    ) -> torch.Tensor:
        """Compute log-probabilities of completion tokens given prompt."""
        full_text = prompt + completion
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True
        ).to(self.device)

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(**inputs)

        logits = outputs.logits[:, prompt_len - 1:-1, :]
        target_ids = inputs.input_ids[:, prompt_len:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        return token_logprobs.sum()

    def train(self) -> Dict[str, Any]:
        if self.train_dataset is None:
            raise ValueError("Training dataset required")

        logger.info(f"Starting GRPO training: {self.config.total_steps} steps")

        self.model.train()
        global_step = 0
        metrics_log = []

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        dataset_indices = list(range(len(self.train_dataset)))

        while global_step < self.config.total_steps:
            np.random.shuffle(dataset_indices)

            for idx in dataset_indices:
                if global_step >= self.config.total_steps:
                    break

                example = self.train_dataset[idx]
                problem = self._problem_from_example(example)
                trajectories = self._generate_group_trajectories(problem)

                prompts = [t[0] for t in trajectories]
                completions = [t[1] for t in trajectories]
                rewards = torch.tensor(
                    [t[2] for t in trajectories], dtype=torch.float32
                ).to(self.device)

                advantages = compute_grpo_advantages(rewards)

                for ppo_epoch in range(self.config.ppo_epochs):
                    total_loss = torch.tensor(0.0, device=self.device)

                    for i in range(len(trajectories)):
                        logprobs_new = self._get_logprobs(
                            self.model, prompts[i], completions[i]
                        )
                        logprobs_old = self._get_logprobs(
                            self.ref_model, prompts[i], completions[i]
                        )

                        policy_loss = compute_policy_loss(
                            logprobs_new, logprobs_old,
                            advantages[i], self.config.clip_range,
                        )
                        kl_penalty = compute_kl_penalty(logprobs_new, logprobs_old)
                        loss = policy_loss + self.config.kl_coefficient * kl_penalty
                        total_loss = total_loss + loss

                    total_loss = total_loss / len(trajectories)
                    total_loss.backward()

                    if (ppo_epoch + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                global_step += 1
                mean_reward = rewards.mean().item()

                if global_step % 50 == 0:
                    logger.info(
                        f"Step {global_step}/{self.config.total_steps} "
                        f"reward={mean_reward:.4f} lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )

                metrics_log.append({
                    "step": global_step,
                    "reward_mean": mean_reward,
                    "reward_std": rewards.std().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                })

                if global_step % 1000 == 0:
                    ckpt_dir = os.path.join(
                        self.config.output_dir, f"checkpoint-{global_step}"
                    )
                    self.model.save_pretrained(ckpt_dir)
                    self.tokenizer.save_pretrained(ckpt_dir)

        final_dir = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)

        logger.info(f"Training complete. Model saved to {final_dir}")
        return {"metrics": metrics_log}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TGPR: GRPO + Tree Search Training")
    parser.add_argument("--model_name", type=str, default="Qwen/
