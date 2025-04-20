import os
import re
import json
import copy
import time
import torch
import logging
import inspect
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset, DatasetDict
from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RefinementProgram:
    """Represents a program in the refinement tree."""
    code: str
    reward: float = 0.0
    heuristic_value: float = 0.0
    test_results: List[bool] = field(default_factory=list)
    parent: Optional['RefinementProgram'] = None
    depth: int = 0
    
    def __hash__(self):
        return hash(self.code)
    
    def __eq__(self, other):
        if not isinstance(other, RefinementProgram):
            return False
        return self.code == other.code

@dataclass
class CodeRefinementProblem:
    """Represents a code refinement problem."""
    task_id: str
    problem_description: str
    buggy_solution: str
    canonical_solution: Optional[str] = None
    tests: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None
    
    def empty_solution(self) -> RefinementProgram:
        """Return the initial (buggy) solution as a RefinementProgram."""
        return RefinementProgram(code=self.buggy_solution)

@dataclass
class GRPOConfig:
    """Configuration for GRPO training with refinement tree."""
    model_name: str
    output_dir: str = "outputs/grpo-refinement"
    
    # Training parameters
    num_train_epochs: int = 1
    train_batch_size: int = 16
    learning_rate: float = 1e-5
    
    # Refinement tree parameters
    max_tree_depth: int = 5
    exploration_coefficient: float = 3.0
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    
    # PPO parameters
    ppo_config: Optional[Dict] = None
    
    # Model paths
    reward_model_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default PPO configuration if not provided."""
        if self.ppo_config is None:
            self.ppo_config = {
                "model_name": self.model_name,
                "steps": 20000,
                "learning_rate": self.learning_rate,
                "batch_size": self.train_batch_size,
                "mini_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "optimize_cuda_cache": True,
                "early_stopping": False,
                "target_kl": 0.1,
                "ppo_epochs": 4,
                "seed": 42,
                "init_kl_coef": 0.2,
                "adap_kl_ctrl": True,
            }

def execute_tests(
    code: str, 
    tests: List[str], 
    entry_point: Optional[str] = None
) -> Tuple[bool, List[bool], str]:
    """
    Execute tests on the given code.
    
    Args:
        code: The code to test
        tests: List of test functions or assertions
        entry_point: Optional name of the main function
        
    Returns:
        Tuple of (all_passed, test_results, error_msg)
    """
    import sys
    import io
    import traceback
    from contextlib import redirect_stdout, redirect_stderr
    
    # Create a temporary file to execute the code
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name
    
    # Prepare the test environment
    test_results = []
    error_msg = ""
    all_passed = False
    
    try:
        # Redirect stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Create a namespace for the code execution
            namespace = {}
            
            # Execute the code
            with open(temp_path, 'r') as f:
                exec(f.read(), namespace)
            
            # Run each test
            for i, test in enumerate(tests):
                try:
                    if entry_point and entry_point in namespace:
                        # If we have an entry point, modify the test to use it
                        test = test.replace("solution_func(", f"{entry_point}(")
                    
                    # Create a test namespace with access to the original namespace
                    test_namespace = dict(namespace)
                    exec(test, test_namespace)
                    test_results.append(True)
                except Exception as e:
                    test_results.append(False)
                    error_msg += f"Test {i+1} failed: {str(e)}\n"
        
        # Check if all tests passed
        all_passed = all(test_results)
        
        # Add stdout and stderr to the error message if there's content
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        if stdout_content:
            error_msg += f"\nStandard output:\n{stdout_content}"
        
        if stderr_content:
            error_msg += f"\nStandard error:\n{stderr_content}"
            
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}\n{traceback.format_exc()}"
        test_results = [False] * len(tests)
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return all_passed, test_results, error_msg

def calculate_reward(
    program: RefinementProgram,
    target_program: Optional[RefinementProgram] = None,
    reward_model: Optional[PreTrainedModel] = None,
    reward_tokenizer: Optional[AutoTokenizer] = None,
    test_weight: float = 0.8,
    model_weight: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    Calculate the reward for a program based on test results and optionally a reward model.
    
    Args:
        program: The program to evaluate
        target_program: The target (canonical) program for comparison
        reward_model: Optional pre-trained reward model
        reward_tokenizer: Tokenizer for the reward model
        test_weight: Weight for the test-based reward
        model_weight: Weight for the model-based reward
        device: Device to run the model on
        
    Returns:
        The calculated reward
    """
    # Calculate test-based reward
    if program.test_results:
        test_reward = sum(program.test_results) / len(program.test_results)
    else:
        test_reward = 0.0
    
    # Calculate model-based reward if a reward model is provided
    model_reward = 0.0
    if reward_model is not None and reward_tokenizer is not None:
        try:
            # Prepare input for the reward model
            inputs = reward_tokenizer(
                program.code,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            
            # Run the model to get the reward prediction
            with torch.no_grad():
                outputs = reward_model(**inputs)
                model_reward = outputs.logits.item()
            
            # Normalize the reward to [0, 1]
            model_reward = max(0.0, min(1.0, model_reward))
            
        except Exception as e:
            logger.error(f"Error calculating model-based reward: {str(e)}")
            model_reward = 0.0
    
    # Combine rewards
    total_reward = (test_weight * test_reward) + (model_weight * model_reward)
    
    # For early stopping: if all tests pass, give maximum reward
    if program.test_results and all(program.test_results):
        total_reward = 1.0
    
    return total_reward

def refine_code(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    problem: CodeRefinementProblem,
    program: RefinementProgram,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> RefinementProgram:
    """
    Generate a refined version of the code using the model.
    
    Args:
        model: The pre-trained model to use for generation
        tokenizer: The tokenizer for the model
        problem: The problem being solved
        program: The current program to refine
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum number of tokens to generate
        device: Device to run the model on
        
    Returns:
        A new RefinementProgram with the refined code
    """
    try:
        # Create a prompt for refinement
        prompt = f"""Fix the following buggy Python code to make it pass all tests.

Problem description:
{problem.problem_description}

Buggy code:
{program.code}

Test cases:
"""
        for i, test in enumerate(problem.tests):
            prompt += f"{test}\n"
            
        prompt += "\nFixed code:\n"
        
        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate the refined code
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        # Decode the generated text
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the fixed code part
        fixed_code_match = re.search(r"Fixed code:\n(.*)", full_output, re.DOTALL)
        if fixed_code_match:
            fixed_code = fixed_code_match.group(1).strip()
        else:
            fixed_code = program.code  # Fallback to the original code
        
        # Create a new RefinementProgram with the refined code
        new_program = RefinementProgram(
            code=fixed_code,
            parent=program,
            depth=program.depth + 1
        )
        
        # Run tests on the new program
        all_passed, test_results, _ = execute_tests(
            fixed_code, problem.tests, problem.entry_point
        )
        new_program.test_results = test_results
        
        # Calculate reward
        reward = calculate_reward(new_program)
        new_program.reward = reward
        
        # Set heuristic value (proportion of tests passed)
        new_program.heuristic_value = sum(test_results) / len(test_results) if test_results else 0.0
        
        return new_program
        
    except Exception as e:
        logger.error(f"Error refining code: {str(e)}")
        # If refinement fails, return a copy of the original program
        return RefinementProgram(
            code=program.code,
            reward=program.reward,
            heuristic_value=program.heuristic_value,
            test_results=program.test_results,
            parent=program,
            depth=program.depth + 1
        )

def refinement_exploration(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    problem: CodeRefinementProblem,
    config: GRPOConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> RefinementProgram:
    """
    Implement the REx (Refinement EXploration) algorithm for code refinement.
    
    Args:
        model: The pre-trained model for code refinement
        tokenizer: The tokenizer for the model
        problem: The code refinement problem
        config: Configuration for the refinement process
        device: Device to run on
        
    Returns:
        The best RefinementProgram found
    """
    # Initialize the set of programs with the initial (buggy) solution
    initial_program = problem.empty_solution()
    
    # Run tests on the initial program
    all_passed, test_results, _ = execute_tests(
        initial_program.code, problem.tests, problem.entry_point
    )
    initial_program.test_results = test_results
    initial_program.heuristic_value = sum(test_results) / len(test_results) if test_results else 0.0
    
    # Check if the initial program already passes all tests
    if all_passed:
        logger.info(f"Initial program already passes all tests for {problem.task_id}")
        return initial_program
    
    # Set up the REx algorithm
    programs = {initial_program}
    failed_cnt = defaultdict(lambda: 0)
    best_program = initial_program
    
    for _ in range(config.max_tree_depth):
        # Select the program to refine using UCB1 formula
        C = config.exploration_coefficient
        program = max(
            programs, 
            key=lambda p: np.random.beta(
                1 + C * p.heuristic_value,
                1 + C * (1 - p.heuristic_value) + failed_cnt[p]
            )
        )
        
        # Refine the program
        new_program = refine_code(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            program=program,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            device=device
        )
        
        # Update the best program if this one is better
        if new_program.heuristic_value > best_program.heuristic_value:
            best_program = new_program
        
        # Check if all tests pass (problem solved)
        if all(new_program.test_results):
            logger.info(f"Found solution that passes all tests for {problem.task_id}")
            return new_program
        
        # Update the failed count for the refined program
        failed_cnt[program] += 1
        
        # Add the new program to the set
        programs.add(new_program)
    
    logger.info(f"Reached max depth for {problem.task_id}. Best program passes {sum(best_program.test_results)}/{len(best_program.test_results)} tests")
    return best_program

def generate_ppo_dataset(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    config: GRPOConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dataset:
    """
    Generate a dataset for PPO training by running refinement exploration.
    
    Args:
        model: The pre-trained model for code refinement
        tokenizer: The tokenizer for the model
        dataset: The dataset of code problems
        config: Configuration for the refinement process
        device: Device to run on
        
    Returns:
        A dataset ready for PPO training
    """
    # Create the PPO dataset
    ppo_data = []
    
    for example in tqdm(dataset, desc="Generating PPO dataset"):
        # Create a problem from the example
        problem = CodeRefinementProblem(
            task_id=example["task_id"],
            problem_description=example["problem_description"],
            buggy_solution=example["buggy_solution"],
            canonical_solution=example.get("canonical_solution"),
            tests=example["tests"],
            entry_point=example.get("entry_point")
        )
        
        # Run refinement exploration
        best_program = refinement_exploration(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            config=config,
            device=device
        )
        
        # Create a trail of refinements from the best program back to the initial one
        refinement_trail = []
        current = best_program
        while current is not None:
            refinement_trail.append(current)
            current = current.parent
        
        # Reverse to get the sequence from initial to best
        refinement_trail.reverse()
        
        # Create prompt-completion pairs for PPO training
        for i in range(len(refinement_trail) - 1):
            # Current program
            current_program = refinement_trail[i]
            
            # Next (better) program
            next_program = refinement_trail[i + 1]
            
            # Skip if the next program doesn't improve the reward
            if next_program.heuristic_value <= current_program.heuristic_value:
                continue
            
            # Create the prompt
            prompt = f"""Fix the following buggy Python code to make it pass all tests.

Problem description:
{problem.problem_description}

Buggy code:
{current_program.code}

Test cases:
"""
            for j, test in enumerate(problem.tests):
                prompt += f"{test}\n"
                
            prompt += "\nFixed code:\n"
            
            # The completion is the next (better) program's code
            completion = next_program.code
            
            # Add to PPO dataset
            ppo_data.append({
                "task_id": problem.task_id,
                "prompt": prompt,
                "completion": completion,
                "reward": next_program.reward,
                "heuristic_value": next_program.heuristic_value,
                "tests_passed": sum(next_program.test_results),
                "total_tests": len(next_program.test_results)
            })
    
    # Create a Hugging Face dataset
    ppo_dataset = Dataset.from_dict({
        key: [example[key] for example in ppo_data]
        for key in ppo_data[0].keys()
    })
    
    return ppo_dataset

class PPODataset(TorchDataset):
    """Dataset for PPO training."""
    
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        return {
            "prompt": example["prompt"],
            "completion": example["completion"],
            "reward": example["reward"]
        }

class SaveBestModelCallback(TrainerCallback):
    """
    Callback to save the best model during training based on reward metrics.
    """
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.best_reward = -float('inf')
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Save the model if it's the best so far based on average reward."""
        if metrics and "eval/rewards/mean" in metrics:
            mean_reward = metrics["eval/rewards/mean"]
            
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                
                # Save the model
                best_model_dir = os.path.join(self.output_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                
                # Get the model from kwargs
                if "model" in kwargs:
                    model = kwargs["model"]
                    model.save_pretrained(best_model_dir)
                    logger.info(f"Saved best model with mean reward {mean_reward} to {best_model_dir}")

class GRPORefinementTrainer:
    """
    Trainer for GRPO with refinement tree algorithm.
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        reward_model: Optional[PreTrainedModel] = None,
        reward_tokenizer: Optional[AutoTokenizer] = None
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load the model and tokenizer if not provided
        if model is None or tokenizer is None:
            logger.info(f"Loading model and tokenizer from {config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
            
            # Ensure the model has a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.model = model
            self.tokenizer = tokenizer
        
        # Load the reward model if specified
        if reward_model is None and config.reward_model_name:
            logger.info(f"Loading reward model from {config.reward_model_name}")
            self.reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
            self.reward_model = AutoModelForCausalLM.from_pretrained(config.reward_model_name)
            self.reward_model.to(self.device)
        else:
            self.reward_model = reward_model
            self.reward_tokenizer = reward_tokenizer
        
        # Initialize the PPO trainer
        self.ppo_config = PPOConfig(**config.ppo_config)
        self.ppo_trainer = None
        
        # Create the output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def prepare_ppo_dataset(self) -> Dataset:
        """
        Prepare the dataset for PPO training.
        
        Returns:
            Dataset ready for PPO training
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset must be provided")
        
        logger.info("Preparing PPO dataset...")
        
        # Generate the PPO dataset by running refinement exploration
        ppo_dataset = generate_ppo_dataset(
            model=self.model.pretrained_model,
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,
            config=self.config,
            device=self.device
        )
        
        # Save the PPO dataset
        ppo_dataset_path = os.path.join(self.config.output_dir, "ppo_dataset")
        ppo_dataset.save_to_disk(ppo_dataset_path)
        logger.info(f"Saved PPO dataset to {ppo_dataset_path}")
        
        return ppo_dataset
    
    def train(self, ppo_dataset: Optional[Dataset] = None) -> dict:
        """
        Train the model using PPO.
        
        Args:
            ppo_dataset: Optional pre-prepared PPO dataset
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare the PPO dataset if not provided
        if ppo_dataset is None:
            ppo_dataset = self.prepare_ppo_dataset()
        
        logger.info("Starting PPO training...")
        
        # Create a PyTorch dataset
        ppo_torch_dataset = PPODataset(ppo_dataset, self.tokenizer)
        
        # Initialize the PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=ppo_torch_dataset,
            data_collator=None  # We'll handle batching ourselves
        )
        
        # Add the save best model callback
        self.ppo_trainer.add_callback(SaveBestModelCallback(self.config.output_dir))
        
        # Extract prompts and completions
        prompts = [example["prompt"] for example in ppo_dataset]
        completions = [example["completion"] for example in ppo_dataset]
        rewards = torch.tensor([example["reward"] for example in ppo_dataset])
        
        # Initialize statistics
        stats = {
            "env/reward_mean": [],
            "env/reward_std": [],
            "env/reward_min": [],
            "env/reward_max": [],
            "ppo/policy_loss": [],
            "ppo/value_loss": [],
            "ppo/entropy": [],
            "ppo/approxkl": []
        }
        
        # Initialize response length sampling
        response_length_sampler = LengthSampler(
            min_value=32, max_value=min(256, self.tokenizer.model_max_length)
        )
        
        # Training loop
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            # Create batches
            batch_size = self.config.train_batch_size
            for i in range(0, len(prompts), batch_size):
                # Get batch
                batch_prompts = prompts[i:i + batch_size]
                batch_completions = completions[i:i + batch_size]
                batch_rewards = rewards[i:i + batch_size]
                
                # Tokenize batch
                query_tensors = [
                    self.tokenizer.encode(prompt, return_tensors="pt")[0]
                    for prompt in batch_prompts
                ]
                
                # Generate responses
                response_tensors = []
                for query_tensor in query_tensors:
                    max_new_tokens = response_length_sampler()
                    response_tensor = self.ppo_trainer.generate(
                        query_tensor.unsqueeze(0).to(self.device),
                        max_new_tokens=max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True
                    )
                    response_tensors.append(response_tensor[0])
                
                # Decode responses
                batch_responses = [
                    self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                    for response_tensor in response_tensors
                ]
                
                # Extract completions from responses
                batch_generations = []
                for response, prompt in zip(batch_responses, batch_prompts):
                    if prompt in response:
                        completion = response[len(prompt):]
                    else:
                        completion = response
                    batch_generations.append(completion)
                
                # Calculate rewards for the generations
                batch_reward_values = []
                for generation, prompt, reward in zip(batch_generations, batch_prompts, batch_rewards):
                    # Try to extract just the code part
                    code_match = re.search(r"```python\s*(.*?)\s*```", generation, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                    else:
                        code = generation
                    
                    # Create a placeholder RefinementProgram for reward calculation
                    program = RefinementProgram(code=code)
                    
                    # Extract the problem from the prompt
                    problem_match = re.search(r"Problem description:\s*(.*?)\s*Buggy code:", prompt, re.DOTALL)
                    if problem_match:
                        problem_description = problem_match.group(1).strip()
                    else:
                        problem_description = ""
                    
                    buggy_code_match = re.search(r"Buggy code:\s*(.*?)\s*Test cases:", prompt, re.DOTALL)
                    if buggy_code_match:
                        buggy_code = buggy_code_match.group(1).strip()
                    else:
                        buggy_code = ""
                    
                    test_cases_match = re.search(r"Test cases:\s*(.*?)\s*Fixed code:", prompt, re.DOTALL)
                    if test_cases_match:
                        test_cases_str = test_cases_match.group(1).strip()
                        test_cases = [tc.strip() for tc in test_cases_str.split("\n") if tc.strip()]
                    else:
                        test_cases = []
                    
                    # Run tests on the generated code
                    all_passed, test_results, _ = execute_tests(code, test_cases)
                    program.test_results = test_results
                    
                    # Calculate reward
                    reward_value = calculate_reward(
                        program=program,
                        reward_model=self.reward_model,
                        reward_tokenizer=self.reward_tokenizer,
                        device=self.device
                    )
                    
                    batch_reward_values.append(reward_value)
                
                # Convert to tensor
                batch_reward_tensors = torch.tensor(batch_reward_values).to(self.device)
                
                # PPO step
                train_stats = self.ppo_trainer.step(
                    query_tensors, response_tensors, batch_reward_tensors
                )
                
                # Update statistics
                for key, value in train_stats.items():
                    if key in stats:
                        stats[key].append(value)
                
                # Log progress
                if (i // batch_size) % 10 == 0:
                    mean_reward = torch.mean(batch_reward_tensors).item()
                    logger.info(f"Epoch {epoch + 1}, Batch {i // batch_size}, Mean Reward: {mean_reward:.4f}")
        
        # Save the final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Save training metrics
        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved training metrics to {metrics_path}")
        
        return stats
    
    def evaluate(self, dataset: Optional[Dataset] = None) -> dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: The dataset to evaluate on. If None, uses self.eval_dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        if dataset is None:
            if self.eval_dataset is None:
                raise ValueError("Evaluation dataset must be provided")
            dataset = self.eval_dataset
        
        logger.info("Starting evaluation...")
        
        # Metrics to track
        metrics = {
            "reward/mean": 0.0,
            "reward/std": 0.0,
            "tests_passed/mean": 0.0,
            "tests_passed/percentage": 0.0,
            "all_tests_passed": 0.0
        }
        
        all_rewards = []
        all_tests_passed_ratio = []
        all_problems_solved = []
        
        for example in tqdm(dataset, desc="Evaluating"):
            # Create a problem from the example
            problem = CodeRefinementProblem(
                task_id=example["task_id"],
                problem_description=example["problem_description"],
                buggy_solution=example["buggy_solution"],
                canonical_solution=example.get("canonical_solution"),
                tests=example["tests"],
                entry_point=example.get("entry_point")
            )
            
            # Run refinement exploration
            best_program = refinement_exploration(
                model=self.model.pretrained_model,
                tokenizer=self.tokenizer,
                problem=problem,
                config=self.config,
                device=self.device
            )
            
            # Track metrics
            all_rewards.append(best_program.reward)
            tests_passed_ratio = sum(best_program.test_results) / len(best_program.test_results) if best_program.test_results else 0.0
            all_tests_passed_ratio.append(tests_passed_ratio)
            all_problems_solved.append(all(best_program.test_results))
        
        # Calculate summary metrics
        metrics["reward/mean"] = float(np.mean(all_rewards))
        metrics["reward/std"] = float(np.std(all_rewards))
        metrics["tests_passed/mean"] = float(np.mean(all_tests_passed_ratio))
        metrics["tests_passed/percentage"] = float(np.mean(all_tests_passed_ratio) * 100)
        metrics["all_tests_passed"] = float(np.mean(all_problems_solved) * 100)
        
        # Save evaluation metrics
        metrics_path = os.path.join(self.config.output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        # Log summary
        logger.info(f"Evaluation results:")
        logger.info(f"  Mean reward: {metrics['reward/mean']:.4f}")
        logger.info(f"  Mean tests passed: {metrics['tests_passed/mean']:.4f} ({metrics['tests_passed/percentage']:.2f}%)")
        logger.info(f"  Problems with all tests passed: {metrics['all_tests_passed']:.2f}%")
        
        return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model using GRPO with refinement tree")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to train")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-refinement", help="Output directory")
    parser.add_argument("--reward_model_path", type=str, help="Path to the reward model (optional)")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_tree_depth", type=int, default=5, help="Maximum refinement tree depth")
    parser.add_argument("--exploration_coef", type=float, default=3.0, help="Exploration coefficient")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load the dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = DatasetDict.load_from_disk(args.dataset_path)
    
    # Create the configuration
    config = GRPOConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_tree_depth=args.max_tree_depth,
        exploration_coefficient=args.exploration_coef,
        temperature=args.temperature,
        reward_model_name=args.reward_model_path,
    )
    
    # Initialize the trainer
    trainer = GRPORefinementTrainer(
        config=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    metrics = trainer.evaluate()
    
    # Save final metrics to a JSON file
    final_metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved final metrics to {final_metrics_path}") 