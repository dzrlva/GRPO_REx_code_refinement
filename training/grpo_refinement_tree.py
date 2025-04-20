import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import random
import time
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper Classes for Refinement Tree
@dataclass
class RefinementProgram:
    """Class for representing a program node in the refinement tree."""
    code: str  # The program code
    parent: Optional['RefinementProgram'] = None  # Parent node in the tree
    reward: float = 0.0  # Reward score for this program
    heuristic_value: float = 0.5  # Heuristic value for exploration
    depth: int = 0  # Depth in the refinement tree
    failed_refinements: int = 0  # Number of failed refinement attempts
    
    def __hash__(self):
        return hash(self.code)
    
    def __eq__(self, other):
        if not isinstance(other, RefinementProgram):
            return False
        return self.code == other.code

@dataclass
class CodeRefinementProblem:
    """Class for representing a code refinement problem."""
    problem_description: str  # Problem description
    buggy_code: str  # Original buggy code
    tests: List[str] = field(default_factory=list)  # Test cases
    reference_solution: Optional[str] = None  # Reference solution if available
    
    def empty_solution(self) -> RefinementProgram:
        """Get the initial buggy solution as a RefinementProgram."""
        return RefinementProgram(code=self.buggy_code)

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    model_name: str
    output_dir: str
    num_train_epochs: int = 3
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    kl_penalty_coefficient: float = 0.1
    entropy_coefficient: float = 0.01
    clip_range: float = 0.2
    adap_kl_control: bool = True
    target_kl: float = 0.1
    init_kl_coefficient: float = 0.2
    max_length: int = 2048
    max_prompt_length: int = 512
    max_response_length: int = 512
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    exploration_coefficient: float = 10.0  # Coefficient for exploration in REx algorithm
    max_tree_depth: int = 5  # Maximum depth of the refinement tree
    max_tree_size: int = 50  # Maximum number of nodes in the refinement tree
    seed: int = 42
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

def execute_tests(code: str, tests: List[str]) -> Tuple[bool, List[bool], str]:
    """
    Execute tests for a given code solution.
    
    Args:
        code: Python code to test
        tests: List of test functions as strings
        
    Returns:
        Tuple of (all_passed, individual_results, error_message)
    """
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    
    # Create a namespace for execution
    namespace = {}
    
    # Capture standard output and error
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Track if all tests pass
    all_passed = False
    individual_results = []
    error_message = ""
    
    try:
        # Execute the solution code
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)
        
        # Execute each test
        for test in tests:
            try:
                test_namespace = namespace.copy()
                # Add a dummy test result collector
                test_namespace['_test_passed'] = True
                
                # Modify the test to set _test_passed to False on assertion error
                modified_test = test.replace('assert ', 'if not (') + '_test_passed = False'
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(modified_test, test_namespace)
                
                individual_results.append(test_namespace['_test_passed'])
            except Exception as e:
                individual_results.append(False)
        
        # Check if all tests passed
        all_passed = all(individual_results) and len(individual_results) > 0
        
    except Exception as e:
        error_message = f"Error executing code: {str(e)}"
        all_passed = False
        individual_results = [False] * len(tests)
    
    # If stderr has content, add it to the error message
    if stderr_capture.getvalue():
        error_message += f"\nStderr: {stderr_capture.getvalue()}"
    
    return all_passed, individual_results, error_message

def calculate_reward(
    refined_code: str, 
    problem: CodeRefinementProblem,
    reward_model: Optional[PreTrainedModel] = None,
    reward_tokenizer: Optional[PreTrainedTokenizer] = None
) -> float:
    """
    Calculate the reward for a refined code solution.
    
    Args:
        refined_code: Refined code solution
        problem: Code refinement problem
        reward_model: Optional reward model
        reward_tokenizer: Optional tokenizer for reward model
        
    Returns:
        Reward value
    """
    # First, check if the code passes the tests
    if problem.tests:
        passed, individual_results, error_message = execute_tests(refined_code, problem.tests)
        # Calculate test-based reward
        test_reward = sum(individual_results) / len(individual_results) if individual_results else 0.0
        
        # If all tests pass, return maximum reward
        if passed:
            return 1.0
    else:
        test_reward = 0.0
    
    # If a reward model is provided, use it for additional reward calculation
    if reward_model is not None and reward_tokenizer is not None:
        try:
            # Format input for reward model
            prompt = f"Problem description: {problem.problem_description}\n\nBuggy code:\n{problem.buggy_code}\n\nRefined code:\n{refined_code}"
            
            # Tokenize the input
            inputs = reward_tokenizer(prompt, return_tensors="pt").to(reward_model.device)
            
            # Generate reward prediction
            with torch.no_grad():
                outputs = reward_model(**inputs)
                reward_value = torch.sigmoid(outputs.logits[:, -1]).item()
            
            # Combine test reward and model reward
            combined_reward = 0.7 * test_reward + 0.3 * reward_value
            return combined_reward
        except Exception as e:
            logger.warning(f"Error computing reward with model: {e}")
            return test_reward
    
    return test_reward

def refine_code(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    program: RefinementProgram,
    problem: CodeRefinementProblem,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 512
) -> RefinementProgram:
    """
    Generate a refined version of the code using the model.
    
    Args:
        model: Language model for code refinement
        tokenizer: Tokenizer for the model
        program: Current program to refine
        problem: Code refinement problem
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        New RefinementProgram with refined code
    """
    # Format prompt
    prompt = f"Problem description: {problem.problem_description}\n\nBuggy code:\n{program.code}\n\nRefined code:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    # Generate refined code
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Create a new RefinementProgram
    new_program = RefinementProgram(
        code=generated_text.strip(),
        parent=program,
        depth=program.depth + 1
    )
    
    return new_program

def refinement_exploration(
    problem: CodeRefinementProblem,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    exploration_coefficient: float = 10.0,
    max_tree_depth: int = 5,
    max_tree_size: int = 50,
    reward_model: Optional[PreTrainedModel] = None,
    reward_tokenizer: Optional[PreTrainedTokenizer] = None
) -> RefinementProgram:
    """
    Implementation of the Refinement Exploration (REx) algorithm.
    
    Args:
        problem: Code refinement problem
        model: Language model for code refinement
        tokenizer: Tokenizer for the model
        exploration_coefficient: Coefficient C for the beta distribution
        max_tree_depth: Maximum depth of the refinement tree
        max_tree_size: Maximum number of nodes in the refinement tree
        reward_model: Optional reward model
        reward_tokenizer: Optional tokenizer for reward model
        
    Returns:
        Best RefinementProgram found
    """
    # Initialize with the buggy solution
    programs = {problem.empty_solution()}
    failed_cnt = defaultdict(lambda: 0)
    best_program = problem.empty_solution()
    best_reward = 0.0
    
    # Calculate initial reward for the buggy solution
    best_program.reward = calculate_reward(
        best_program.code, problem, reward_model, reward_tokenizer
    )
    
    # Continue refining until termination conditions are met
    num_iterations = 0
    
    while (len(programs) < max_tree_size and 
           num_iterations < max_tree_size * 2 and 
           best_reward < 1.0):  # Stop if we find a perfect solution
        
        num_iterations += 1
        
        # Select the program to refine using UCB-like formula
        selected_program = max(programs, key=lambda p: np.random.beta(
            1 + exploration_coefficient * p.heuristic_value,
            1 + exploration_coefficient * (1 - p.heuristic_value) + failed_cnt[p]
        ))
        
        # Skip if we've reached max depth
        if selected_program.depth >= max_tree_depth:
            failed_cnt[selected_program] += 1
            continue
        
        # Refine the selected program
        new_program = refine_code(model, tokenizer, selected_program, problem)
        
        # Skip if the refined code is the same as the parent
        if new_program.code == selected_program.code:
            failed_cnt[selected_program] += 1
            continue
        
        # Calculate reward for the new program
        new_program.reward = calculate_reward(
            new_program.code, problem, reward_model, reward_tokenizer
        )
        
        # Update heuristic value based on reward
        new_program.heuristic_value = new_program.reward
        
        # Add to programs set
        programs.add(new_program)
        
        # Check if this is the best program so far
        if new_program.reward > best_reward:
            best_program = new_program
            best_reward = new_program.reward
            
            # If this is a perfect solution, we're done
            if best_reward >= 1.0:
                break
        else:
            # Increment failed count for the parent
            failed_cnt[selected_program] += 1
    
    return best_program

def generate_ppo_dataset(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: GRPOConfig,
    reward_model: Optional[PreTrainedModel] = None,
    reward_tokenizer: Optional[PreTrainedTokenizer] = None
) -> Dataset:
    """
    Generate a dataset for PPO training using the refinement tree.
    
    Args:
        dataset: Input dataset with code refinement problems
        model: Language model for code refinement
        tokenizer: Tokenizer for the model
        config: GRPO configuration
        reward_model: Optional reward model
        reward_tokenizer: Optional tokenizer for reward model
        
    Returns:
        Dataset prepared for PPO training
    """
    ppo_examples = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Generating PPO dataset")):
        # Create a code refinement problem
        problem = CodeRefinementProblem(
            problem_description=example["problem_description"],
            buggy_code=example["buggy_solution"],
            tests=example["tests"] if isinstance(example["tests"], list) else [],
            reference_solution=example["canonical_solution"]
        )
        
        # Run refinement exploration
        best_program = refinement_exploration(
            problem=problem,
            model=model,
            tokenizer=tokenizer,
            exploration_coefficient=config.exploration_coefficient,
            max_tree_depth=config.max_tree_depth,
            max_tree_size=config.max_tree_size,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer
        )
        
        # Format the example for PPO training
        ppo_example = {
            "task_id": example["task_id"],
            "prompt": f"Problem description: {problem.problem_description}\n\nBuggy code:\n{problem.buggy_code}\n\nRefined code:",
            "chosen": best_program.code,
            "rejected": problem.buggy_code,  # The original buggy code is the rejected response
            "reward": best_program.reward
        }
        
        ppo_examples.append(ppo_example)
    
    return Dataset.from_list(ppo_examples)

class GRPORefinementTrainer:
    """Trainer for code refinement using GRPO with a refinement tree."""
    
    def __init__(
        self,
        config: GRPOConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        reward_model_path: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: GRPO configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            reward_model_path: Path to a reward model checkpoint
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        if config.use_peft:
            # Initialize LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            # Apply PEFT to model
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Initialize reward model if provided
        self.reward_model = None
        self.reward_tokenizer = None
        
        if reward_model_path:
            try:
                self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
                self.reward_model = AutoModelForCausalLM.from_pretrained(
                    reward_model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                )
                logger.info(f"Loaded reward model from {reward_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load reward model: {e}")
        
        # Initialize PPO config
        self.ppo_config = PPOConfig(
            learning_rate=config.learning_rate,
            batch_size=config.train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            ppo_epochs=config.ppo_epochs,
            max_grad_norm=config.max_grad_norm,
            adap_kl_ctrl=config.adap_kl_control,
            init_kl_coef=config.init_kl_coefficient,
            target_kl=config.target_kl,
            gamma=1.0,
            cliprange=config.clip_range,
            cliprange_value=config.clip_range,
            vf_coef=0.1,
            seed=config.seed,
            log_with=None
        )
    
    def _prepare_ppo_dataset(self) -> Dataset:
        """
        Prepare the dataset for PPO training.
        
        Returns:
            Dataset prepared for PPO training
        """
        logger.info("Preparing PPO dataset using refinement exploration...")
        
        return generate_ppo_dataset(
            dataset=self.train_dataset,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            reward_model=self.reward_model,
            reward_tokenizer=self.reward_tokenizer
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model using GRPO with refinement tree exploration.
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting GRPO training with refinement tree...")
        
        # Prepare PPO dataset
        ppo_dataset = self._prepare_ppo_dataset()
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=ppo_dataset
        )
        
        # Set up a length sampler
        response_sampler = LengthSampler(
            min_value=64, max_value=self.config.max_response_length
        )
        
        # Create generation config
        generation_config = GenerationConfig(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Training loop
        training_metrics = {
            "steps": [],
            "rewards": [],
            "kl": [],
            "losses": []
        }
        
        # Get example prompts for training
        prompts = [example["prompt"] for example in ppo_dataset]
        
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            batch_metrics = []
            
            for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch + 1}"):
                # Get batch of prompts
                batch_prompts = batch["prompt"]
                
                # Tokenize prompts
                prompt_tensors = []
                for prompt in batch_prompts:
                    prompt_tensors.append(
                        self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
                    )
                
                # Sample model-generated refinements
                response_tensors = []
                for prompt_tensor in prompt_tensors:
                    response_length = response_sampler()
                    response_tensor = ppo_trainer.generate(
                        prompt_tensor, 
                        max_new_tokens=response_length,
                        generation_config=generation_config
                    )
                    response_tensors.append(response_tensor)
                
                # Decode the responses
                batch_refinements = []
                for prompt_tensor, response_tensor in zip(prompt_tensors, response_tensors):
                    refinement = self.tokenizer.decode(
                        response_tensor[0][prompt_tensor.shape[1]:], 
                        skip_special_tokens=True
                    )
                    batch_refinements.append(refinement)
                
                # Calculate rewards
                rewards = []
                for i, (prompt, refinement) in enumerate(zip(batch_prompts, batch_refinements)):
                    # Extract problem description and buggy code from prompt
                    parts = prompt.split("\n\nBuggy code:\n")
                    if len(parts) != 2:
                        continue
                        
                    problem_desc = parts[0].replace("Problem description: ", "")
                    buggy_code = parts[1].split("\n\nRefined code:")[0]
                    
                    # Create a refinement problem
                    problem = CodeRefinementProblem(
                        problem_description=problem_desc,
                        buggy_code=buggy_code,
                        tests=batch["tests"][i] if "tests" in batch else []
                    )
                    
                    # Calculate reward
                    reward = calculate_reward(
                        refinement, problem, self.reward_model, self.reward_tokenizer
                    )
                    rewards.append(reward)
                
                # Run PPO step
                stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
                batch_metrics.append(stats)
                
                # Save metrics
                training_metrics["steps"].append(epoch * len(ppo_trainer.dataloader) + step)
                training_metrics["rewards"].append(np.mean(rewards))
                training_metrics["kl"].append(stats["kl"])
                training_metrics["losses"].append(stats["ppo/loss/total"])
                
                # Log metrics
                if step % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}, Step {step}: "
                        f"Reward: {np.mean(rewards):.4f}, "
                        f"KL: {stats['kl']:.4f}, "
                        f"Loss: {stats['ppo/loss/total']:.4f}"
                    )
            
            # Evaluate on validation set if provided
            if self.eval_dataset is not None and len(self.eval_dataset) > 0:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation metrics after epoch {epoch + 1}: {eval_metrics}")
            
            # Save checkpoint
            output_dir = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training metrics
            with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
                json.dump(training_metrics, f)
        
        # Save final model
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        return training_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            logger.warning("No evaluation dataset provided")
            return {}
        
        logger.info("Evaluating model...")
        
        eval_metrics = {
            "reward_mean": 0.0,
            "pass_rate": 0.0,
            "syntactic_correctness": 0.0
        }
        
        rewards = []
        pass_count = 0
        syntax_correct_count = 0
        
        for example in tqdm(self.eval_dataset, desc="Evaluating"):
            # Create the prompt
            prompt = f"Problem description: {example['problem_description']}\n\nBuggy code:\n{example['buggy_solution']}\n\nRefined code:"
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate refined code
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=self.config.max_response_length,
                    temperature=0.2,  # Lower temperature for evaluation
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode the output
            refined_code = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Create a problem object
            problem = CodeRefinementProblem(
                problem_description=example["problem_description"],
                buggy_code=example["buggy_solution"],
                tests=example["tests"] if isinstance(example["tests"], list) else [],
                reference_solution=example["canonical_solution"]
            )
            
            # Calculate reward
            reward = calculate_reward(
                refined_code, problem, self.reward_model, self.reward_tokenizer
            )
            rewards.append(reward)
            
            # Check if the solution passes all tests
            if problem.tests:
                passed, _, _ = execute_tests(refined_code, problem.tests)
                if passed:
                    pass_count += 1
            
            # Check for syntax correctness
            try:
                compile(refined_code, "<string>", "exec")
                syntax_correct_count += 1
            except SyntaxError:
                pass
        
        # Calculate metrics
        if rewards:
            eval_metrics["reward_mean"] = np.mean(rewards)
        
        if len(self.eval_dataset) > 0:
            eval_metrics["pass_rate"] = pass_count / len(self.eval_dataset)
            eval_metrics["syntactic_correctness"] = syntax_correct_count / len(self.eval_dataset)
        
        logger.info(f"Evaluation metrics: {eval_metrics}")
        
        return eval_metrics

if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk
    
    parser = argparse.ArgumentParser(description="Train a code refinement model using GRPO with refinement tree")
    parser.add_argument("--model_name", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--reward_model_path", type=str, help="Path to reward model checkpoint")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_tree_depth", type=int, default=5, help="Maximum depth of refinement tree")
    parser.add_argument("--max_tree_size", type=int, default=50, help="Maximum size of refinement tree")
    parser.add_argument("--exploration_coefficient", type=float, default=10.0, help="Exploration coefficient for REx")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT for fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    
    # Create config
    config = GRPOConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        max_tree_depth=args.max_tree_depth,
        max_tree_size=args.max_tree_size,
        exploration_coefficient=args.exploration_coefficient,
        use_peft=args.use_peft,
        seed=args.seed
    )
    
    # Create trainer
    trainer = GRPORefinementTrainer(
        config=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        reward_model_path=args.reward_model_path
    )
    
    # Train the model
    metrics = trainer.train()
    
    # Save metrics
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    logger.info(f"Training complete. Model saved to {args.output_dir}") 