import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import logging
from typing import Dict, List, Optional, Union, Any

# Import our custom modules
from training.grpo_trainer import GRPOTrainer, GRPOConfig
from training.tree_grpo_trainer import TreeGRPOTrainer
from models.reward_model import CodeRefinementRewardModel, RewardModelWrapper
from data.prepare_data import get_combined_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a code refinement model with GRPO+Tree")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to the prepared dataset. If not provided, data will be prepared.")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save the model checkpoints and logs.")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B-Instruct",
                       help="Path to the base model to fine-tune.")
    parser.add_argument("--reference_model", type=str, default=None,
                       help="Path to the reference model. If None, will use the base model.")
    parser.add_argument("--reward_model_path", type=str, default=None,
                       help="Path to a pretrained reward model. If None, will create a new one.")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs.")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log every N steps during training.")
    parser.add_argument("--save_interval", type=int, default=500,
                       help="Save model every N steps.")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision.")
    
    # GRPO specific arguments
    parser.add_argument("--num_generations", type=int, default=4,
                       help="Number of completions to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation.")
    parser.add_argument("--epsilon", type=float, default=0.2,
                       help="PPO epsilon (clipping parameter).")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                       help="KL penalty coefficient.")
    
    # Tree search arguments
    parser.add_argument("--use_tree", action="store_true",
                       help="Whether to use the refinement tree in training.")
    parser.add_argument("--tree_c_param", type=float, default=3.0,
                       help="C parameter for the refinement tree.")
    parser.add_argument("--max_tree_iter", type=int, default=20,
                       help="Maximum number of iterations for tree search.")
    
    # Reward model arguments
    parser.add_argument("--codebleu_weight", type=float, default=0.5,
                       help="Weight for the CodeBLEU component in the reward.")
    parser.add_argument("--test_weight", type=float, default=0.5,
                       help="Weight for the test execution component in the reward.")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility.")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with smaller dataset.")
    
    return parser.parse_args()

def prepare_dataset(args) -> pd.DataFrame:
    """
    Prepare the dataset for training.
    
    Args:
        args: Command line arguments
        
    Returns:
        pandas DataFrame containing the dataset
    """
    if args.dataset_path and os.path.exists(args.dataset_path):
        logger.info(f"Loading dataset from {args.dataset_path}")
        return pd.read_parquet(args.dataset_path)
    else:
        logger.info("Preparing dataset from scratch")
        dataset = get_combined_dataset()
        
        if args.debug:
            logger.info("Debug mode: using a small subset of the data")
            dataset = dataset.sample(n=min(100, len(dataset)))
            
        # Save the dataset
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        dataset_path = os.path.join(args.output_dir, "prepared_dataset.parquet")
        dataset.to_parquet(dataset_path)
        logger.info(f"Saved prepared dataset to {dataset_path}")
        
        return dataset

def prepare_reward_model(args) -> RewardModelWrapper:
    """
    Prepare the reward model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Reward model wrapped in a compatible interface
    """
    if args.reward_model_path and os.path.exists(args.reward_model_path):
        logger.info(f"Loading reward model from {args.reward_model_path}")
        # In this case, we're actually loading the configuration, not a model
        reward_model = CodeRefinementRewardModel(
            codebleu_weight=args.codebleu_weight,
            test_weight=args.test_weight
        )
    else:
        logger.info("Creating a new reward model")
        reward_model = CodeRefinementRewardModel(
            codebleu_weight=args.codebleu_weight,
            test_weight=args.test_weight
        )
        
    return RewardModelWrapper(reward_model)

def prepare_training_args(args) -> TrainingArguments:
    """
    Prepare training arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        TrainingArguments for the HuggingFace Trainer
    """
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_total_limit=3,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

def prepare_grpo_config(args) -> GRPOConfig:
    """
    Prepare GRPO configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        GRPOConfig for the GRPO Trainer
    """
    return GRPOConfig(
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        epsilon_high=args.epsilon,
        epsilon_low=args.epsilon,
        beta=args.kl_coef,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        output_dir=args.output_dir,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        seed=args.seed,
    )

def convert_dataset_to_hf_format(df: pd.DataFrame, debug: bool = False) -> Dict[str, List]:
    """
    Convert pandas DataFrame to HuggingFace Dataset format.
    
    Args:
        df: pandas DataFrame with the dataset
        debug: Whether to use a smaller subset for debugging
        
    Returns:
        Dict with lists for HuggingFace Dataset
    """
    if debug:
        df = df.sample(n=min(100, len(df)))
    
    return {
        "prompt": df["problem_description"].tolist(),
        "canonical_solution": df["canonical_solution"].tolist(),
        "buggy_solution": df["buggy_solution"].tolist(),
        "test_cases": df["tests"].tolist(),
        "dataset": df["dataset"].tolist(),
        "task_id": df["task_id"].tolist(),
    }

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset
    df = prepare_dataset(args)
    logger.info(f"Dataset contains {len(df)} examples")
    
    # Convert to HuggingFace Dataset format
    from datasets import Dataset
    dataset_dict = convert_dataset_to_hf_format(df, args.debug)
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and evaluation sets (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    
    logger.info(f"Training set: {len(train_dataset)} examples")
    logger.info(f"Evaluation set: {len(eval_dataset)} examples")
    
    # Load tokenizer and model
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare reward model
    reward_model = prepare_reward_model(args)
    
    # Configure the reward function
    def reward_function(prompts, completions, **kwargs):
        """Custom reward function for code refinement."""
        # Extract canonical solutions and test cases
        reference_codes = kwargs.get('canonical_solution', [])
        test_cases = kwargs.get('test_cases', [])
        
        # Calculate rewards
        return reward_model(
            prompts=prompts,
            completions=completions,
            reference_codes=reference_codes,
            test_cases=test_cases
        )
    
    # Prepare training arguments
    training_args = prepare_training_args(args)
    
    # Prepare GRPO config
    grpo_config = prepare_grpo_config(args)
    
    # Initialize the appropriate trainer
    if args.use_tree:
        logger.info("Using TreeGRPOTrainer with refinement tree integration")
        trainer = TreeGRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            tree_c_hyperparameter=args.tree_c_param,
            max_tree_iterations=args.max_tree_iter
        )
    else:
        logger.info("Using standard GRPOTrainer")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer
        )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main() 