# SMILES2025: Tree-Augmented RL for Autonomous Repair of AI-Generated Code

This module implements the Iterative Refinement Tree algorithm for code generation with GRPO pipeline. 

## Key Components

- **Refinement Tree**: A tree structure where each node represents a program, and children are refinements of the parent program.
- **Code Generator**: Generates and refines code using large language models (Qwen).
- **Reward Model**: Evaluates code solutions based on test execution and/or neural model predictions.
- **REx Algorithm**: Balances exploration and exploitation using Thompson Sampling-based algorithm.

<img width="832" alt="Снимок экрана 2025-04-21 в 20 12 22" src="https://github.com/user-attachments/assets/644bb231-bde4-4fcd-bd2e-9231c78b112b" />


## Installation

```bash
# Clone the repository
git clone dzrlva/SMILES2025_project
cd SMILES2025_project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the REx Algorithm

```bash
python main.py --problem_path problems/fibonacci.json --model_type qwen
```

### Command Line Arguments

- `--problem_path`: Path to the problem JSON file (required)
- `--output_dir`: Directory to save results (default: 'results')
- `--model_type`: Type of code generation model to use (default: 'qwen')
- `--model_path`: Path or name of the model to use for code generation (default: model-specific)
- `--reward_model_path`: Path to neural reward model (if None, uses execution-only reward)
- `--max_iterations`: Maximum number of iterations for Iterative Refinement Tree algorithm (default: 20)
- `--max_depth`: Maximum depth of the refinement tree (default: 5)
- `--exploration_coefficient`: Exploration coefficient C calculation (default: 0.5)
- `--min_reward_threshold`: Minimum reward threshold to consider a problem solved (default: 0.8)
- `--temperature`: Temperature for code generation (default: 0.7)
- `--save_tree`: Save the full refinement tree in the results (default: False)

### Example

```bash
# Run the Iterative Refinement Tree algorithm on the Fibonacci problem using Qwen model
python main.py --problem_path problems/fibonacci.json --model_type qwen --max_iterations 30 --temperature 0.8
```

## Implementation Details

### Refinement Tree

The refinement tree is implemented in `refinement_tree.py`. It uses a beta distribution to select nodes for refinement based on their reward scores, which balance exploration and exploitation.

### Reward Model

The reward model in `reward_model.py` provides three implementations:
- `ExecutionRewardModel`: Executes code against test cases.
- `NeuralRewardModel`: Uses a fine-tuned neural model to predict code quality.
- `HybridRewardModel`: Combines execution and neural reward models.

### Code Generator

The code generator in `code_generator.py` supports multiple LLM architectures:
- `QwenCodeGenerator`: Uses Qwen models.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Other dependencies listed in requirements.txt 
