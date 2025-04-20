#!/usr/bin/env python3
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExecutionRewardModel:
    """Reward model based on executing code against test cases."""
    
    def __init__(self, timeout: int = 5, max_execution_time: int = 30):
        self.timeout = timeout  # Timeout for individual test cases
        self.max_execution_time = max_execution_time  # Max total execution time
    
    def get_reward(self, problem: str, solution: str, tests: List[str], **kwargs) -> float:
        """Get reward for a solution based on test execution."""
        # Skip empty solutions
        if not solution.strip():
            return 0.0
        
        # Create a temporary directory for the code
        with tempfile.TemporaryDirectory() as tempdir:
            # Write the solution to a file
            solution_file = os.path.join(tempdir, "solution.py")
            with open(solution_file, "w") as f:
                f.write(solution)
            
            # Write the test file
            test_file = os.path.join(tempdir, "test_solution.py")
            with open(test_file, "w") as f:
                f.write("import solution\nimport sys\n\n")
                
                # Write each test case as a function
                for i, test in enumerate(tests):
                    f.write(f"def test_{i}():\n")
                    # Indent the test code
                    for line in test.strip().split("\n"):
                        f.write(f"    {line}\n")
                    f.write("\n")
                
                # Write the main function to run tests
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
            
            # Execute the tests
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["python", test_file],
                    capture_output=True,
                    text=True,
                    timeout=self.max_execution_time
                )
                execution_time = time.time() - start_time
                
                # Parse the output to get the number of tests passed
                output = result.stdout
                match = re.search(r"(\d+)/(\d+) tests passed", output)
                if match:
                    passed = int(match.group(1))
                    total = int(match.group(2))
                    
                    # Compute the reward
                    correctness_score = passed / total if total > 0 else 0
                    
                    # Adjust for execution time - faster is better
                    time_factor = min(1.0, self.max_execution_time / (execution_time + 1e-5))
                    
                    # Combine the scores - correctness is the main factor
                    reward = 0.9 * correctness_score + 0.1 * time_factor
                    
                    logger.info(f"Solution passed {passed}/{total} tests with reward {reward:.4f}")
                    return reward
                else:
                    logger.warning(f"Failed to parse test output: {output}")
                    return 0.0
            
            except subprocess.TimeoutExpired:
                logger.warning("Execution timed out")
                return 0.0
            except Exception as e:
                logger.error(f"Error executing solution: {str(e)}")
                return 0.0

class NeuralRewardModel:
    """Reward model based on a fine-tuned neural model."""
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Neural reward model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load neural reward model: {str(e)}")
            raise
    
    def get_reward(self, problem: str, solution: str, **kwargs) -> float:
        """Get reward prediction from the neural model."""
        # Skip empty solutions
        if not solution.strip():
            return 0.0
        
        # Prepare the input
        prompt = f"Problem:\n{problem}\n\nSolution:\n{solution}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        
        # Get the prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the reward score
        if hasattr(outputs, "logits"):
            # Classification or regression model
            if outputs.logits.shape[1] == 1:
                # Regression
                reward = torch.sigmoid(outputs.logits[0, 0]).item()
            else:
                # Classification - use the positive class probability
                reward = torch.softmax(outputs.logits, dim=1)[0, 1].item()
        else:
            # Value prediction model
            reward = outputs.value.item()
        
        return reward

class HybridRewardModel:
    """Combines execution-based and neural reward models."""
    
    def __init__(
        self, 
        execution_model: Optional[ExecutionRewardModel] = None,
        neural_model: Optional[NeuralRewardModel] = None,
        execution_weight: float = 0.7
    ):
        self.execution_model = execution_model or ExecutionRewardModel()
        self.neural_model = neural_model
        self.execution_weight = execution_weight
    
    def get_reward(self, problem: str, solution: str, tests: List[str], **kwargs) -> float:
        """Get combined reward from execution and neural models."""
        # Get execution reward
        execution_reward = self.execution_model.get_reward(problem, solution, tests)
        
        # If no neural model, just return execution reward
        if self.neural_model is None:
            return execution_reward
        
        # Get neural reward
        neural_reward = self.neural_model.get_reward(problem, solution)
        
        # Combine rewards
        combined_reward = (
            self.execution_weight * execution_reward + 
            (1 - self.execution_weight) * neural_reward
        )
        
        logger.info(f"Hybrid reward: {combined_reward:.4f} (execution: {execution_reward:.4f}, neural: {neural_reward:.4f})")
        return combined_reward 