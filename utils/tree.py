import numpy as np
import torch
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
from collections import defaultdict

class Program:
    """
    Represents a program in the refinement tree.
    
    Attributes:
        code (str): The code representing the program.
        heuristic_value (float): A value indicating the quality of the program.
    """
    def __init__(self, code: str, heuristic_value: float = 0.0):
        self.code = code
        self.heuristic_value = heuristic_value
    
    def __hash__(self):
        return hash(self.code)
    
    def __eq__(self, other):
        if not isinstance(other, Program):
            return False
        return self.code == other.code


class RefinementTree:
    """
    Implements the refinement tree algorithm for code refinement.
    
    The tree uses a beta distribution to select programs for refinement,
    balancing between high-reward programs and programs that have failed 
    less often in previous refinement attempts.
    """
    def __init__(self, 
                reward_model: Callable[[List[str]], List[float]],
                refinement_model: Callable[[str, str], List[str]],
                c_hyperparameter: float = 3.0,
                num_candidates: int = 5,
                max_iterations: int = 50):
        """
        Initialize the refinement tree.
        
        Args:
            reward_model: A function that takes a list of code strings and returns a list of reward values.
            refinement_model: A function that takes a problem description and a code to refine, and returns a list of refined code strings.
            c_hyperparameter: Hyperparameter controlling the weight of heuristic value vs. failed attempts.
            num_candidates: Number of best candidates to return.
            max_iterations: Maximum number of iterations to run.
        """
        self.reward_model = reward_model
        self.refinement_model = refinement_model
        self.C = c_hyperparameter
        self.num_candidates = num_candidates
        self.max_iterations = max_iterations
        
    def refine(self, problem_description: str, initial_code: str) -> List[str]:
        """
        Run the refinement tree algorithm to improve code.
        
        Args:
            problem_description: Description of the programming problem.
            initial_code: Initial code to be refined.
            
        Returns:
            List of the best refined code candidates.
        """
        # Initialize the set of programs with the initial code
        initial_program = Program(initial_code)
        # Evaluate the initial program with the reward model
        initial_program.heuristic_value = self.reward_model([initial_code])[0]
        
        programs = {initial_program}
        failed_cnt = defaultdict(lambda: 0)
        
        iter_count = 0
        while iter_count < self.max_iterations:
            # Select program with the highest beta distribution value
            program = max(programs, key=lambda p: np.random.beta(
                1 + self.C * p.heuristic_value,
                1 + self.C * (1 - p.heuristic_value) + failed_cnt[p]
            ))
            
            # Generate refinements
            refined_codes = self.refinement_model(problem_description, program.code)
            
            # If no refinements could be generated, mark as failed and continue
            if not refined_codes:
                failed_cnt[program] += 1
                iter_count += 1
                continue
                
            # Evaluate the refined programs
            heuristic_values = self.reward_model(refined_codes)
            
            # Create Program objects for all refinements
            new_programs = [Program(code, heuristic_value) 
                          for code, heuristic_value in zip(refined_codes, heuristic_values)]
            
            # Add new programs to the set
            programs.update(new_programs)
            
            # Check if any refinement improved significantly
            best_new_program = max(new_programs, key=lambda p: p.heuristic_value)
            if best_new_program.heuristic_value > program.heuristic_value + 0.1:
                # Successful refinement, reset failed count for this path
                failed_cnt[program] = 0
            else:
                # Not a significant improvement, increment failed count
                failed_cnt[program] += 1
                
            iter_count += 1
        
        # Return the top N programs by heuristic value
        sorted_programs = sorted(programs, key=lambda p: p.heuristic_value, reverse=True)
        return [p.code for p in sorted_programs[:self.num_candidates]]


class GRPOTreeIntegration:
    """
    Integrates the refinement tree with GRPO for code refinement.
    
    This class helps connect the refinement tree exploration strategy with
    the Group Relative Policy Optimization training process.
    """
    def __init__(self, 
                reward_model: Any, 
                policy_model: Any,
                c_hyperparameter: float = 3.0,
                num_refinements: int = 5,
                max_tree_iterations: int = 20):
        """
        Initialize the GRPO Tree Integration.
        
        Args:
            reward_model: The model used to evaluate code quality.
            policy_model: The model used to generate refined code.
            c_hyperparameter: Controls the exploration vs. exploitation in the tree search.
            num_refinements: Number of refinement candidates to generate.
            max_tree_iterations: Maximum iterations for the tree search.
        """
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.C = c_hyperparameter
        self.num_refinements = num_refinements
        self.max_tree_iterations = max_tree_iterations
        self.failed_counts = defaultdict(lambda: 0)
        
    def get_code_reward(self, code_samples: List[str]) -> List[float]:
        """
        Get rewards for code samples using the reward model.
        
        Args:
            code_samples: List of code strings to evaluate.
            
        Returns:
            List of reward values for each code sample.
        """
        # This will be implemented based on the specific reward model interface
        return self.reward_model(code_samples)
    
    def generate_refinements(self, problem_desc: str, code_to_refine: str) -> List[str]:
        """
        Generate code refinements using the policy model.
        
        Args:
            problem_desc: Description of the problem.
            code_to_refine: Code to be refined.
            
        Returns:
            List of refined code samples.
        """
        # This will be implemented based on the specific policy model interface
        # The implementation will depend on how the policy model is defined and accessed
        return self.policy_model.generate_refinements(problem_desc, code_to_refine)
    
    def select_best_refinements(self, 
                               problem_desc: str, 
                               code_candidates: List[str]) -> List[str]:
        """
        Select the best refinements using the refinement tree algorithm.
        
        Args:
            problem_desc: Description of the problem.
            code_candidates: List of code candidates to evaluate and refine.
            
        Returns:
            List of the best refined code candidates.
        """
        # Initialize TreeRefinement with the current models
        tree = RefinementTree(
            reward_model=self.get_code_reward,
            refinement_model=lambda prob, code: self.generate_refinements(prob, code),
            c_hyperparameter=self.C,
            num_candidates=self.num_refinements,
            max_iterations=self.max_tree_iterations
        )
        
        # Process each candidate through the tree refinement
        all_refined_candidates = []
        for code in code_candidates:
            refined_candidates = tree.refine(problem_desc, code)
            all_refined_candidates.extend(refined_candidates)
        
        # Return the top N unique candidates
        unique_candidates = list(set(all_refined_candidates))
        unique_candidates_rewards = self.get_code_reward(unique_candidates)
        
        sorted_candidates = [x for _, x in sorted(
            zip(unique_candidates_rewards, unique_candidates), 
            key=lambda pair: pair[0], 
            reverse=True
        )]
        
        return sorted_candidates[:self.num_refinements] 