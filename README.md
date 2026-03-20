# TGPR: Tree-Guided Policy Refinement for Code Debugging

This repository contains the implementation of **TGPR** (Tree-Guided Policy Refinement), a framework that integrates Thompson Sampling-guided tree search with Group Relative Policy Optimization (GRPO) to improve iterative code refinement in LLMs.

**Paper:** *TGPR: Thompson Sampling-guided GRPO for Code Refinement*

## Overview

TGPR addresses the exploration bottleneck in RL-based code debugging by using a Thompson Sampling-guided tree search as a **training-time** data augmentation engine. The tree generates diverse debugging trajectories — both successes and informative failures — that are used to update the policy via GRPO. At inference time, the model operates as a standard autoregressive generator with no additional overhead.

### Key Results

| Benchmark | GRPO (pass@1) | TGPR (pass@1) | Δ |
|-----------|:---:|:---:|:---:|
| MBPP | 77.9 | **82.1** | +4.2 |
| APPS | 58.7 | **62.4** | +3.7 |
| HumanEval | 83.3 | **87.1** | +3.8 |
| LiveCodeBench | 51.2 | **55.8** | +4.6 |

## Architecture

<img width="832" alt="TGPR Pipeline" src="https://github.com/user-attachments/assets/644bb231-bde4-4fcd-bd2e-9231c78b112b" />

The pipeline consists of three stages:
1. **Data Collection:** Problem descriptions and canonical solutions are extracted from benchmarks (MBPP, HumanEval, APPS); an LLM generates up to 10 candidate refinements per program.
2. **Data Verification:** Each refinement is executed against test suites and labeled as correct or incorrect.
3. **RL Training:** Verified trajectories are used to train the GRPO agent integrated with the Thompson Sampling-guided tree search.

## Key Components

- **Thompson Sampling Tree** (`refinement_tree.py`): Tree structure where each node is a program variant. Node selection uses Beta distribution with parameters ensuring valid parameterization (α, β > 0).
- **Reward Model** (`reward_model.py`): Hybrid reward combining CodeBLEU (0.5) and test pass rate (0.5), with three implementations:
  - `ExecutionRewardModel`: Executes code against test cases
  - `NeuralRewardModel`: Fine-tuned Qwen2.5-1.5B predicting code quality
  - `HybridRewardModel`: Combines execution and neural rewards
- **Code Generator** (`code_generator.py`): Qwen2.5-7B-Instruct as the base policy model.
- **GRPO Training**: Policy optimization with KL regularization (λ = 0.01), group size 8, clip range 0.2.

## Installation

```bash
git clone https://github.com/dzrlva/TGPR_project.git
cd TGPR_project
pip install -r requirements.txt
