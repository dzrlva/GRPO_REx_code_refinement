import os
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import subprocess
import tempfile
from codebleu import calc_codebleu
import re

class CodeRefinementRewardModel:
    """
    Reward model for code refinement that combines CodeBLEU metric and test execution results.
    
    The reward is calculated as a weighted sum of the CodeBLEU score and the ratio of passing tests.
    This provides a comprehensive evaluation of both the structural similarity to reference code
    and the functional correctness of the generated solutions.
    """
    
    def __init__(
        self, 
        codebleu_weight: float = 0.5, 
        test_weight: float = 0.5,
        timeout: int = 30
    ):
        """
        Initialize the reward model.
        
        Args:
            codebleu_weight: Weight for the CodeBLEU component of the reward.
            test_weight: Weight for the test execution component of the reward.
            timeout: Maximum execution time for running tests in seconds.
        """
        self.codebleu_weight = codebleu_weight
        self.test_weight = test_weight
        self.timeout = timeout
        
    def safe_execute(self, code: str, timeout: int = None) -> Tuple[bool, str]:
        """
        Safely execute Python code in an isolated subprocess.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds (overrides instance timeout if provided)
            
        Returns:
            Tuple of (success, output/error message)
        """
        if timeout is None:
            timeout = self.timeout
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
            
        try:
            process = subprocess.Popen(
                ["python", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                if process.returncode == 0:
                    return True, stdout
                else:
                    return False, stderr
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return False, f"Execution timed out after {timeout} seconds"
                
        except Exception as e:
            return False, str(e)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    def run_tests(self, code: str, test_cases: List[str]) -> Tuple[int, int]:
        """
        Run tests on the provided code and count how many pass.
        
        Args:
            code: Python code to test
            test_cases: List of test cases (as assert statements or test functions)
            
        Returns:
            Tuple of (passed_tests, total_tests)
        """
        passed = 0
        total = len(test_cases)
        
        for test in test_cases:
            # Construct a full test script
            full_code = f"{code}\n\n{test}"
            success, _ = self.safe_execute(full_code)
            if success:
                passed += 1
                
        return passed, total
    
    def calculate_codebleu(self, reference_code: str, generated_code: str) -> float:
        """
        Calculate the CodeBLEU score between reference and generated code.
        
        Args:
            reference_code: The reference/canonical code
            generated_code: The generated/refined code
            
        Returns:
            CodeBLEU score between 0 and 1
        """
        try:
            score = calc_codebleu(
                references=[reference_code], 
                hyps=[generated_code],
                lang="python",
                weights=(0.25, 0.25, 0.25, 0.25),
                tokenizer=None
            )
            return score['codebleu']
        except Exception as e:
            print(f"Error calculating CodeBLEU: {e}")
            return 0.0
    
    def compute_reward(
        self, 
        generated_code: str, 
        reference_code: str,
        test_cases: List[str]
    ) -> Dict[str, float]:
        """
        Compute the reward for a generated code snippet.
        
        Args:
            generated_code: The generated/refined code
            reference_code: The reference/canonical code
            test_cases: List of test cases to run against the generated code
            
        Returns:
            Dictionary containing the components and final reward:
            {
                'codebleu_score': float,
                'test_pass_ratio': float,
                'combined_reward': float
            }
        """
        # Calculate CodeBLEU score
        codebleu_score = self.calculate_codebleu(reference_code, generated_code)
        
        # Run tests
        passed, total = self.run_tests(generated_code, test_cases)
        test_pass_ratio = passed / total if total > 0 else 0.0
        
        # Combine scores
        combined_reward = (
            self.codebleu_weight * codebleu_score +
            self.test_weight * test_pass_ratio
        )
        
        return {
            'codebleu_score': codebleu_score,
            'test_pass_ratio': test_pass_ratio,
            'passed_tests': passed,
            'total_tests': total,
            'combined_reward': combined_reward
        }
    
    def __call__(
        self, 
        prompts: Optional[List[str]] = None, 
        completions: List[str] = None,
        reference_codes: Optional[List[str]] = None,
        test_cases: Optional[List[List[str]]] = None
    ) -> List[float]:
        """
        Callable interface for integration with GRPO pipeline.
        
        Args:
            prompts: Problem descriptions (optional)
            completions: Generated code completions
            reference_codes: Reference/canonical solutions
            test_cases: Test cases for each completion
            
        Returns:
            List of reward values for each completion
        """
        rewards = []
        
        for i, completion in enumerate(completions):
            # Get reference code and test cases for this completion
            reference = reference_codes[i] if reference_codes else ""
            tests = test_cases[i] if test_cases else []
            
            # Calculate reward
            reward_info = self.compute_reward(completion, reference, tests)
            rewards.append(reward_info['combined_reward'])
            
        return rewards
        
    @staticmethod
    def from_pretrained(
        model_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> "CodeRefinementRewardModel":
        """
        Create a reward model from a pretrained model.
        
        This is a compatibility method for the HuggingFace trainer interface
        that expects reward models to have a from_pretrained method.
        
        Args:
            model_path: Path to pretrained model (ignored in this implementation)
            device: Device to use (ignored in this implementation)
            **kwargs: Additional arguments passed to the constructor
            
        Returns:
            CodeRefinementRewardModel instance
        """
        return CodeRefinementRewardModel(**kwargs)

class RewardModelWrapper:
    """
    Wrapper for the reward model to align with the interface expected by the GRPO trainer.
    
    This wrapper adapts the reward model to work seamlessly with the GRPO training pipeline,
    by providing methods compatible with the expected interface.
    """
    
    def __init__(self, reward_model: CodeRefinementRewardModel):
        """
        Initialize the wrapper with a reward model.
        
        Args:
            reward_model: The underlying reward model
        """
        self.reward_model = reward_model
        
    def __call__(
        self, 
        prompts: Optional[List[str]] = None,
        completions: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        Process a batch of prompts and completions to calculate rewards.
        
        Args:
            prompts: List of problem descriptions
            completions: List of generated code solutions
            **kwargs: Additional arguments like reference_codes and test_cases
            
        Returns:
            List of reward values
        """
        # Extract reference codes and test cases from kwargs
        reference_codes = kwargs.get('reference_codes', [None] * len(completions))
        test_cases = kwargs.get('test_cases', [[] for _ in range(len(completions))])
        
        # Calculate rewards
        return self.reward_model(
            prompts=prompts,
            completions=completions,
            reference_codes=reference_codes,
            test_cases=test_cases
        )
        
    @staticmethod
    def from_pretrained(
        model_path: str,
        **kwargs
    ) -> "RewardModelWrapper":
        """
        Create a wrapped reward model from a pretrained model.
        
        This compatibility method creates both the underlying reward model
        and wraps it in the appropriate interface.
        
        Args:
            model_path: Path to pretrained model (ignored in this implementation)
            **kwargs: Additional arguments passed to the reward model constructor
            
        Returns:
            RewardModelWrapper instance
        """
        reward_model = CodeRefinementRewardModel.from_pretrained(model_path, **kwargs)
        return RewardModelWrapper(reward_model) 