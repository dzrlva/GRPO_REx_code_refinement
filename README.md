# SMILES2025 REx: Refinement Tree Exploration for Code Generation

This module implements the Refinement Tree Exploration (REx) algorithm for code generation. REx uses a Monte Carlo Tree Search (MCTS) inspired approach to balance exploration and exploitation when refining code solutions.

## Key Components

- **Refinement Tree**: A tree structure where each node represents a program, and children are refinements of the parent program.
- **Code Generator**: Generates and refines code using large language models (Qwen, CodeLlama, DeepSeek).
- **Reward Model**: Evaluates code solutions based on test execution and/or neural model predictions.
- **REx Algorithm**: Balances exploration and exploitation using Upper Confidence Bound (UCB) scores.

## Installation

```bash
# Clone the repository
git clone <repository-url>
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
- `--model_type`: Type of code generation model to use (choices: 'qwen', 'codellama', 'deepseek', default: 'qwen')
- `--model_path`: Path or name of the model to use for code generation (default: model-specific)
- `--reward_model_path`: Path to neural reward model (if None, uses execution-only reward)
- `--max_iterations`: Maximum number of iterations for the REx algorithm (default: 20)
- `--max_depth`: Maximum depth of the refinement tree (default: 5)
- `--exploration_coefficient`: Exploration coefficient for UCB calculation (default: 0.5)
- `--min_reward_threshold`: Minimum reward threshold to consider a problem solved (default: 0.8)
- `--temperature`: Temperature for code generation (default: 0.7)
- `--save_tree`: Save the full refinement tree in the results (default: False)

### Problem JSON Format

Problems are defined in JSON files with the following structure:

```json
{
  "id": "problem_id",
  "prompt": "Problem description...",
  "tests": [
    "assert function_name(args) == expected_output",
    ...
  ],
  "solutions": [
    "def function_name(args):\n    ...",
    ...
  ]
}
```

### Example

```bash
# Run the REx algorithm on the Fibonacci problem using Qwen model
python main.py --problem_path problems/fibonacci.json --model_type qwen --max_iterations 30 --temperature 0.8

# Run on prime factorization problem using DeepSeek model with custom exploration coefficient
python main.py --problem_path problems/prime_factorization.json --model_type deepseek --exploration_coefficient 0.7
```

## Implementation Details

### Refinement Tree

The refinement tree is implemented in `refinement_tree.py`. It uses a beta distribution to select nodes for refinement based on their UCB scores, which balance exploration and exploitation.

### Reward Model

The reward model in `reward_model.py` provides three implementations:
- `ExecutionRewardModel`: Executes code against test cases.
- `NeuralRewardModel`: Uses a fine-tuned neural model to predict code quality.
- `HybridRewardModel`: Combines execution and neural reward models.

### Code Generator

The code generator in `code_generator.py` supports multiple LLM architectures:
- `QwenCodeGenerator`: Uses Qwen models.
- `CodeLlamaGenerator`: Uses CodeLlama models.
- `DeepseekCoderGenerator`: Uses DeepSeek Coder models.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Other dependencies listed in requirements.txt 