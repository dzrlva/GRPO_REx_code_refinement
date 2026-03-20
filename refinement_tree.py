#!/usr/bin/env python3
"""
Thompson Sampling-guided tree search for TGPR framework.

Implements the refinement tree from Section 3.2:
    - Node selection via Beta(α, β) Thompson Sampling
    - α = 1 + C · R_norm
    - β = max(1 + C · (1 - R_norm) + N_ρ, ε)
    - R_norm = clip(R / R_max, 0, 1)
    - C = 2.0, ε = 1e-6
    - Max depth: 5, branching factor: 4
"""

import numpy as np
import logging
import torch
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TS_COEFFICIENT = 2.0
BETA_EPSILON = 1e-6
MAX_TREE_DEPTH = 5
BRANCHING_FACTOR = 4
REWARD_MAX = 1.0
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_P = 0.95
GENERATION_TOP_K = 50


@dataclass
class Program:
    """A program node in the refinement tree."""
    code: str
    parent: Optional['Program'] = None
    reward: float = 0.0
    depth: int = 0
    children: List['Program'] = field(default_factory=list)

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if not isinstance(other, Program):
            return False
        return self.code == other.code

    def __repr__(self):
        return f"Program(depth={self.depth}, reward={self.reward:.3f}, code={self.code[:50]}...)"


@dataclass
class CodeProblem:
    """A code problem to solve."""
    problem_id: str
    prompt: str
    tests: List[str]
    reference_solution: Optional[str] = None
    buggy_solution: Optional[str] = None

    def initial_program(self) -> Program:
        if self.buggy_solution:
            return Program(code=self.buggy_solution, reward=0.0)
        return Program(code="", reward=0.0)


@dataclass
class TreeConfig:
    """
    Configuration matching paper hyperparameters (Section 3.2, Appendix).
    """
    max_depth: int = MAX_TREE_DEPTH
    branching_factor: int = BRANCHING_FACTOR
    ts_coefficient: float = TS_COEFFICIENT
    beta_epsilon: float = BETA_EPSILON
    temperature: float = GENERATION_TEMPERATURE
    top_p: float = GENERATION_TOP_P
    top_k: int = GENERATION_TOP_K
    max_tokens: int = 512
    max_iterations: int = 20


class RefinementTree:
    """
    Thompson Sampling-guided tree search for code refinement.

    Each node is a program variant. At each iteration, the tree:
    1. Selects a node via Thompson Sampling (Beta distribution)
    2. Generates `branching_factor` children via LLM refinement
    3. Evaluates children using the hybrid reward model
    4. Updates failure counts for Beta distribution parameters
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_model: Any,
        config: Optional[TreeConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config or TreeConfig()

    def _normalize_reward(self, reward: float) -> float:
        """R_norm = clip(R / R_max, 0, 1)"""
        return float(np.clip(reward / REWARD_MAX, 0.0, 1.0))

    def _compute_alpha(self, r_norm: float) -> float:
        """α = 1 + C · R_norm"""
        return 1.0 + self.config.ts_coefficient * r_norm

    def _compute_beta(self, r_norm: float, n_failed: int) -> float:
        """β = max(1 + C · (1 - R_norm) + N_ρ, ε)"""
        raw = 1.0 + self.config.ts_coefficient * (1.0 - r_norm) + n_failed
        return max(raw, self.config.beta_epsilon)

    def _thompson_sample(self, program: Program, n_failed: int) -> float:
        """Sample θ ~ Beta(α, β) for node selection."""
        r_norm = self._normalize_reward(program.reward)
        alpha = self._compute_alpha(r_norm)
        beta = self._compute_beta(r_norm, n_failed)
        return np.random.beta(alpha, beta)

    def _select_node(
        self, programs: Set[Program], failed_counts: Dict[Program, int]
    ) -> Program:
        """Select node with highest Thompson Sample: ρ_next = argmax θ_ρ."""
        if len(programs) == 1:
            return next(iter(programs))
        return max(
            programs,
            key=lambda p: self._thompson_sample(p, failed_counts[p]),
        )

    def _generate_refinement(self, program: Program, problem: CodeProblem) -> str:
        """Generate a single refined version of the program via LLM."""
        prompt = self._build_prompt(program.code, problem)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        refined = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        end_markers = ["```", "\n\n\n"]
        for marker in end_markers:
            if marker in refined:
                refined = refined.split(marker)[0]

        return refined.strip()

    def _generate_children(
        self, program: Program, problem: CodeProblem
    ) -> List[Program]:
        """
        Generate `branching_factor` children for a node.
        Each child is a refinement of the parent program.
        """
        if program.depth >= self.config.max_depth:
            logger.warning(f"Max depth {self.config.max_depth} reached")
            return []

        children = []
        for _ in range(self.config.branching_factor):
            refined_code = self._generate_refinement(program, problem)
            child = Program(
                code=refined_code,
                parent=program,
                depth=program.depth + 1,
            )
            child.reward = self._evaluate(child, problem)
            children.append(child)

        program.children.extend(children)
        return children

    def _evaluate(self, program: Program, problem: CodeProblem) -> float:
        """
        Evaluate program using hybrid reward:
        R(ρ) = 0.5 · CodeBLEU(ρ, ρ_c) + 0.5 · |T_p(ρ)| / |T|
        """
        reward = self.reward_model.get_reward(
            problem=problem.prompt,
            solution=program.code,
            tests=problem.tests,
            reference_solution=problem.reference_solution,
        )
        return float(np.clip(reward, 0.0, REWARD_MAX))

    def _build_prompt(self, code: str, problem: CodeProblem) -> str:
        """Build refinement prompt for the LLM."""
        if problem.buggy_solution:
            return (
                f"You are an expert programmer. Fix the bugs in this Python code.\n\n"
                f"Problem:\n{problem.prompt}\n\n"
                f"Buggy code:\n```python\n{code}\n```\n\n"
                f"Tests:\n```python\n{self._format_tests(problem.tests)}\n```\n\n"
                f"Fixed code:\n```python\n"
            )
        else:
            prompt = (
                f"You are an expert programmer. Write Python code to solve this problem.\n\n"
                f"Problem:\n{problem.prompt}\n\n"
            )
            if code:
                prompt += f"Current attempt:\n```python\n{code}\n```\n\n"
            prompt += (
                f"Tests:\n```python\n{self._format_tests(problem.tests)}\n```\n\n"
                f"Solution:\n```python\n"
            )
            return prompt

    def _format_tests(self, tests: List[str]) -> str:
        return "\n\n".join(tests)

    def search(self, problem: CodeProblem) -> Tuple[Program, Dict[str, Any]]:
        """
        Run Thompson Sampling-guided tree search.

        At each iteration:
        1. Select node via Thompson Sampling (Beta distribution)
        2. Generate `branching_factor` children
        3. Update failure counts
        4. Return best program found
        """
        initial = problem.initial_program()
        initial.reward = self._evaluate(initial, problem)

        programs: Set[Program] = {initial}
        failed_counts: Dict[Program, int] = defaultdict(int)
        best_program = initial

        metrics = {
            "iterations": 0,
            "total_programs": 1,
            "best_reward": initial.reward,
            "solved": False,
        }

        logger.info(f"Starting tree search for problem {problem.problem_id}")

        for iteration in range(1, self.config.max_iterations + 1):
            selected = self._select_node(programs, failed_counts)

            logger.info(
                f"Iteration {iteration}: selected node "
                f"(depth={selected.depth}, reward={selected.reward:.4f})"
            )

            children = self._generate_children(selected, problem)

            if not children:
                failed_counts[selected] += 1
                continue

            improved = False
            for child in children:
                programs.add(child)
                metrics["total_programs"] += 1

                if child.reward > best_program.reward:
                    best_program = child
                    metrics["best_reward"] = child.reward
                    improved = True

                if child.reward > 0.99:
                    logger.info(f"Problem solved at iteration {iteration}")
                    metrics["solved"] = True
                    metrics["iterations"] = iteration
                    return best_program, metrics

            if not improved:
                failed_counts[selected] += 1

            metrics["iterations"] = iteration

        logger.info(
            f"Search complete. Best reward: {best_program.reward:.4f} "
            f"after {metrics['iterations']} iterations"
        )
        return best_program, metrics
