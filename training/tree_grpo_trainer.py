import os
import torch
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import transformers
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

# Import the original GRPO trainer
from training.grpo_trainer import GRPOTrainer, GRPOConfig, RewardFunc

# Import the refinement tree
from utils.tree import GRPOTreeIntegration

class TreeGRPOTrainer(GRPOTrainer):
    """
    Extends the GRPOTrainer to integrate a refinement tree for code improvement.
    
    This trainer modifies the generation and evaluation process by using a refinement tree
    to explore and select promising code refinements.
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[Any] = None, 
        tree_c_hyperparameter: float = 3.0,
        max_tree_iterations: int = 20
    ):
        """
        Initialize the TreeGRPOTrainer.
        
        Args:
            model: The policy model to be trained.
            reward_funcs: The reward function(s) to evaluate code quality.
            args: Configuration for the trainer.
            train_dataset: Dataset to use for training.
            eval_dataset: Dataset to use for evaluation.
            processing_class: Processing class used to process the data.
            reward_processing_classes: Processing classes for the reward functions.
            callbacks: List of callbacks to customize the training loop.
            optimizers: A tuple containing the optimizer and the scheduler.
            peft_config: PEFT configuration.
            tree_c_hyperparameter: Controls the exploration vs. exploitation in tree search.
            max_tree_iterations: Maximum iterations for the tree search.
        """
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # Setup tree integration parameters
        self.tree_c_hyperparameter = tree_c_hyperparameter
        self.max_tree_iterations = max_tree_iterations
        self.tree_integration = None  # Will be initialized in _generate_and_score_completions
        
    def _generate_and_score_completions(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Override the generation and scoring process to use the refinement tree.
        
        This method:
        1. Uses the policy model to generate initial completions
        2. Passes these completions through the refinement tree
        3. Selects the best refinements based on the reward model
        4. Evaluates and returns the selected refinements
        
        Args:
            inputs: Input dictionary containing prompts and other data.
            
        Returns:
            Dictionary with the refined completions and their evaluations.
        """
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]
        problem_descriptions = [x.get("problem_description", "") for x in inputs]
        
        # Create a wrapper for the reward model
        def reward_model_wrapper(code_samples: List[str]) -> List[float]:
            # This is a simplified wrapper - in practice, you'll need to format inputs correctly
            batch_rewards = []
            # Group the code samples in batches to avoid OOM issues
            batch_size = min(16, len(code_samples))
            for i in range(0, len(code_samples), batch_size):
                batch = code_samples[i:i+batch_size]
                # Assume reward functions expect code string directly
                # In real implementation, you may need to adapt this based on reward function expectations
                batch_result = []
                for reward_func in self.reward_funcs:
                    if callable(reward_func):
                        result = reward_func(completions=batch)
                    else:  # Assume it's a model
                        with torch.no_grad():
                            inputs = {"text": batch}
                            result = reward_func(**inputs)
                    batch_result.append(result)
                    
                # Combine results from multiple reward functions
                if len(batch_result) > 1:
                    result = [sum(values) for values in zip(*batch_result)]
                else:
                    result = batch_result[0]
                    
                batch_rewards.extend(result)
            
            return batch_rewards
        
        # Create a wrapper for the policy model
        def policy_model_wrapper(problem_desc: str, code_to_refine: str) -> List[str]:
            # Format the input for the policy model
            formatted_input = f"Problem description: {problem_desc}\nCode to refine: {code_to_refine}\nRefined code:"
            
            # Use the model to generate refinements
            inputs = self.processing_class(
                text=[formatted_input], 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.processing_class.model_max_length - self.max_completion_length,
                return_attention_mask=True,
                add_special_tokens=True
            ).to(device)
            
            with torch.no_grad():
                with self.accelerator.unwrap_model(self.model).disable_adapter() if hasattr(self.model, 'disable_adapter') else nullcontext():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_completion_length,
                        num_return_sequences=self.num_generations,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True
                    )
            
            # Decode the outputs
            refinements = self.processing_class.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return refinements
        
        # Initialize the tree integration if not already done
        if self.tree_integration is None:
            self.tree_integration = GRPOTreeIntegration(
                reward_model=reward_model_wrapper,
                policy_model={"generate_refinements": policy_model_wrapper},
                c_hyperparameter=self.tree_c_hyperparameter,
                num_refinements=self.num_generations,
                max_tree_iterations=self.max_tree_iterations
            )
        
        # Process in batches to avoid OOM issues
        batch_size = min(8, len(prompts))
        all_completions = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_descriptions = problem_descriptions[i:i+batch_size]
            
            # Generate initial completions using the original GRPO method
            # We'll use the parent class's method but capture only what we need
            with torch.no_grad():
                # Generate basic completions
                batch_inputs = {
                    "prompt": batch_prompts,
                    "problem_description": batch_descriptions
                }
                initial_results = super()._generate_and_score_completions(batch_inputs)
                initial_completions = self.processing_class.batch_decode(
                    initial_results["completion_ids"],
                    skip_special_tokens=True
                )
            
            # For each problem description, use the tree to refine its completions
            for j, (desc, comps) in enumerate(zip(batch_descriptions, [initial_completions[j:j+self.num_generations] for j in range(0, len(initial_completions), self.num_generations)])):
                refined_completions = self.tree_integration.select_best_refinements(desc, comps)
                all_completions.extend(refined_completions)
        
        # Now that we have the refined completions, prepare them for the GRPO pipeline
        completion_tokens = [self.processing_class.encode(comp, add_special_tokens=False) for comp in all_completions]
        completion_ids = [torch.tensor(tokens, device=device) for tokens in completion_tokens]
        
        # Pad and prepare completions for model input
        from transformers.modeling_utils import pad_sequence
        padded_completion_ids = pad_sequence(
            completion_ids, 
            batch_first=True, 
            padding_value=self.processing_class.pad_token_id
        )
        
        # Process prompt information
        prompts_text = [p for p in prompts for _ in range(self.num_generations)]
        prompt_inputs = self.processing_class(
            text=prompts_text, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left", 
            add_special_tokens=False
        ).to(device)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        
        # Create completion mask
        completion_mask = torch.ones_like(padded_completion_ids)
        
        # Now we need to evaluate the completions with the model and get logprobs
        input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Rest of the function follows the original GRPO implementation for evaluation
        logits_to_keep = padded_completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, input_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, input_ids, attention_mask, logits_to_keep, batch_size
                    )
        
        # Calculate rewards
        completions_text = all_completions
        if hasattr(inputs[0]['prompt'], 'is_conversational') and inputs[0]['prompt'].is_conversational:
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # Calculate rewards for the refined completions
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, torch.nn.Module):
                # Handle model-based reward functions
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                
                if hasattr(inputs[0]['prompt'], 'is_conversational') and inputs[0]['prompt'].is_conversational:
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [self.processing_class.apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                ).to(device)
                
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Handle function-based reward functions
                reward_func_name = reward_func.__name__
                
                # Extract any additional information needed by the reward function
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        # Check for NaN values
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Apply weights and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize rewards for advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        
        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        # Log metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        
        # Log completion lengths
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())
        
        # Log rewards
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, torch.nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            reward_func_names.append(reward_func_name)
        
        # Calculate mean reward per function
        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            from training.grpo_trainer import nanstd
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        # Log textual data
        from accelerate.utils import gather_object
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": padded_completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        } 