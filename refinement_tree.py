#!/usr/bin/env python3
import numpy as np
import logging
import torch
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Program:
    """A program in the refinement tree."""
    code: str
    parent: Optional['Program'] = None
    heuristic_value: float = 0.5  # Default heuristic value
    depth: int = 0  # Depth in the tree
    
    def __hash__(self):
        """Hash based on code content."""
        return hash(self.code)
    
    def __eq__(self, other):
        """Equality based on code content."""
        if not isinstance(other, Program):
            return False
        return self.code == other.code
    
    def __repr__(self):
        """String representation of the program."""
        return f"Program(depth={self.depth}, heuristic={self.heuristic_value:.3f}, code={self.code[:50]}...)"

@dataclass
class CodeProblem:
    """A code problem to solve."""
    problem_id: str
    prompt: str
    tests: List[str]
    original_solution: Optional[str] = None
    buggy_solution: Optional[str] = None
    
    def empty_solution(self) -> Program:
        """Return an empty solution for this problem."""
        if self.buggy_solution:
            return Program(code=self.buggy_solution, heuristic_value=0.1)
        return Program(code="", heuristic_value=0.1)

@dataclass
class RefinementTreeConfig:
    """Configuration for the refinement tree."""
    max_tree_depth: int = 5
    exploration_coef: float = 2.0
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    time_limit: int = 60  # Time limit in seconds
    max_iterations: int = 20  # Maximum number of iterations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_tree_depth": self.max_tree_depth,
            "exploration_coef": self.exploration_coef,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "time_limit": self.time_limit,
            "max_iterations": self.max_iterations,
        }

class RefinementTree:
    """Implementation of the Refinement Exploration (REx) algorithm."""
    
    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_model: Any,
        config: RefinementTreeConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config
        
    def generate_refinement(self, program: Program, problem: CodeProblem) -> str:
        """Generate a refined version of the program."""
        # Construct the prompt for refinement
        prompt = self._construct_refinement_prompt(program.code, problem)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate a continuation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the continuation
        refined_code = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Find the end of the generated code
        # Sometimes the model generates additional text after the code
        end_markers = ["```", "\n\n", "\nHere's", "\nThis", "\nI've", "\nNow", "\nThe"]
        for marker in end_markers:
            if marker in refined_code:
                refined_code = refined_code.split(marker)[0]
        
        # Combine with original buggy code if we're doing bug fixing
        if problem.buggy_solution:
            # Find the common parts and merge them
            refined_code = self._merge_code(problem.buggy_solution, refined_code)
        
        return refined_code
    
    def _construct_refinement_prompt(self, code: str, problem: CodeProblem) -> str:
        """Construct a prompt for refining a program."""
        if problem.buggy_solution:
            # Bug fixing prompt
            prompt = f"""You are an expert programmer. Your task is to fix bugs in the following Python code.

Problem description:
{problem.prompt}

Buggy code:
```python
{code}
```

Tests the code should pass:
```python
{self._format_tests(problem.tests)}
```

Fixed code:
```python
"""
        else:
            # Code generation prompt
            prompt = f"""You are an expert programmer. Your task is to write a Python function that solves the following problem.

Problem description:
{problem.prompt}

"""
            if code:
                prompt += f"""Current solution (may be incomplete or incorrect):
```python
{code}
```

"""
            
            prompt += f"""Tests the code should pass:
```python
{self._format_tests(problem.tests)}
```

Complete solution:
```python
"""
        
        return prompt
    
    def _format_tests(self, tests: List[str]) -> str:
        """Format test cases for inclusion in the prompt."""
        return "\n\n".join(tests)
    
    def _merge_code(self, original_code: str, refined_code: str) -> str:
        """Merge the original code with the refined code."""
        # Simple merging strategy: if the refined code is significantly shorter than original,
        # it might be a partial fix, so prepend the original
        if len(refined_code.strip()) < len(original_code.strip()) / 2:
            # Get original function signature
            lines = original_code.split("\n")
            signature_line = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    signature_line = i
                    break
            
            if signature_line >= 0:
                # Include original function signature and any imports
                prefix = "\n".join(lines[:signature_line+1])
                return f"{prefix}\n{refined_code}"
        
        return refined_code
    
    def evaluate_program(self, program: Program, problem: CodeProblem) -> float:
        """Evaluate a program using the reward model."""
        # Construct the input for the reward model
        inputs = {
            "problem": problem.prompt,
            "solution": program.code,
            "tests": problem.tests
        }
        
        # Get reward score
        reward_score = self.reward_model.get_reward(**inputs)
        
        return reward_score
    
    def is_solved(self, problem: CodeProblem, program: Program) -> bool:
        """Check if a program solves the problem."""
        reward_score = self.evaluate_program(program, problem)
        return reward_score > 0.95  # Consider solved if reward score is high
    
    def refine(self, program: Program, problem: CodeProblem) -> Program:
        """Refine a program to create a new version."""
        # Don't refine beyond max depth
        if program.depth >= self.config.max_tree_depth:
            logger.warning(f"Max tree depth reached for program: {program}")
            return program
        
        # Generate a refinement
        refined_code = self.generate_refinement(program, problem)
        
        # Create a new Program object
        refined_program = Program(
            code=refined_code,
            parent=program,
            depth=program.depth + 1
        )
        
        # Evaluate the refined program
        refined_program.heuristic_value = self.evaluate_program(refined_program, problem)
        
        return refined_program
    
    def run_REx(self, problem: CodeProblem) -> Tuple[Program, Dict[str, Any]]:
        """Run the REx algorithm on a problem."""
        # Start with the empty solution or buggy solution
        programs = {problem.empty_solution()}
        failed_count = defaultdict(lambda: 0)
        iterations = 0
        metrics = {
            "program_count": 0,
            "iterations": 0,
            "best_reward": 0.0,
            "refinement_tree": []
        }
        
        # Add the initial program to the tree
        initial_program = problem.empty_solution()
        initial_program.heuristic_value = self.evaluate_program(initial_program, problem)
        metrics["refinement_tree"].append({
            "code": initial_program.code,
            "depth": initial_program.depth,
            "heuristic": initial_program.heuristic_value,
            "failed_count": 0
        })
        
        logger.info(f"Starting REx for problem {problem.problem_id}")
        
        # Main REx loop
        while iterations < self.config.max_iterations:
            iterations += 1
            
            # Select the program with highest UCB score
            if len(programs) == 1:
                program = next(iter(programs))
            else:
                program = max(programs, key=lambda p: np.random.beta(
                    1 + self.config.exploration_coef * p.heuristic_value,
                    1 + self.config.exploration_coef * (1 - p.heuristic_value) + failed_count[p]
                ))
            
            logger.info(f"Iteration {iterations}: Selected program with heuristic={program.heuristic_value:.3f}")
            
            # Refine the program
            new_program = self.refine(program, problem)
            
            # Update metrics
            metrics["program_count"] += 1
            metrics["iterations"] = iterations
            metrics["best_reward"] = max(metrics["best_reward"], new_program.heuristic_value)
            
            # Add to tree visualization
            metrics["refinement_tree"].append({
                "code": new_program.code,
                "depth": new_program.depth,
                "heuristic": new_program.heuristic_value,
                "failed_count": failed_count[program],
                "parent_index": metrics["refinement_tree"].index(next(
                    item for item in metrics["refinement_tree"] 
                    if item["code"] == program.code and item["depth"] == program.depth
                )) if program.code in [item["code"] for item in metrics["refinement_tree"]] else -1
            })
            
            # Check if the problem is solved
            if self.is_solved(problem, new_program):
                logger.info(f"Problem solved after {iterations} iterations!")
                metrics["solved"] = True
                return new_program, metrics
            
            # Update the failed count
            failed_count[program] += 1
            
            # Add the new program to the set
            programs.add(new_program)
        
        # If we reach here, return the best program found
        best_program = max(programs, key=lambda p: p.heuristic_value)
        logger.info(f"Max iterations reached. Best program has heuristic={best_program.heuristic_value:.3f}")
        metrics["solved"] = False
        
        return best_program, metrics 