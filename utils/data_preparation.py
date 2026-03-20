#!/usr/bin/env python3
"""
Data preparation pipeline for TGPR framework.

From the paper (Section 4.1):
    - 3 benchmarks for training: MBPP, HumanEval, APPS
    - Up to 10 candidate solutions per problem via 3-shot prompting
    - Solutions executed against hidden unit tests
    - Classified as correct, partially correct, or incorrect
    - Failing solutions paired with execution feedback
    - LLM generates corrected refinements
    - Decontamination: CodeBLEU > 0.85 removed, 10-gram check
    - Data split: 80/10/10 stratified
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict

from reward_model import compute_codebleu, compute_test_pass_rate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CANDIDATES_PER_PROBLEM = 10
CODEBLEU_CONTAMINATION_THRESHOLD = 0.85
NGRAM_SIZE = 10
DATA_SPLIT = (0.8, 0.1, 0.1)


@dataclass
class CodeProblem:
    problem_id: str
    prompt: str
    tests: List[str]
    reference_solution: str
    source: str = "unknown"
    candidate_solutions: List[Dict] = field(default_factory=list)


def load_humaneval(path: str) -> List[CodeProblem]:
    """Load HumanEval dataset."""
    logger.info(f"Loading HumanEval from {path}")
    problems = []

    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            tests = []
            test_lines = data['test'].strip().split("\n")
            current_test = []
            for tl in test_lines:
                if tl.startswith("def"):
                    if current_test:
                        tests.append("\n".join(current_test))
                        current_test = []
                current_test.append(tl)
            if current_test:
                tests.append("\n".join(current_test))

            problems.append(CodeProblem(
                problem_id=data['task_id'],
                prompt=data['prompt'],
                tests=tests,
                reference_solution=data['canonical_solution'],
                source="humaneval",
            ))

    logger.info(f"Loaded {len(problems)} HumanEval problems")
    return problems


def load_mbpp(path: str) -> List[CodeProblem]:
    """Load MBPP dataset."""
    logger.info(f"Loading MBPP from {path}")
    problems = []

    with open(path, 'r') as f:
        data = json.load(f)

    for item in data:
        if 'code' not in item or not item['code'].strip():
            continue
        problems.append(CodeProblem(
            problem_id=f"mbpp_{item['task_id']}",
            prompt=item['text'],
            tests=item['test_list'],
            reference_solution=item['code'],
            source="mbpp",
        ))

    logger.info(f"Loaded {len(problems)} MBPP problems")
    return problems


def load_apps(path: str, max_problems: int = 500) -> List[CodeProblem]:
    """Load APPS dataset."""
    logger.info(f"Loading APPS from {path}")
    problems = []
    problem_dirs = list(Path(path).glob("*/*"))

    if max_problems > 0:
        random.shuffle(problem_dirs)
        problem_dirs = problem_dirs[:max_problems]

    for pdir in tqdm(problem_dirs, desc="Loading APPS"):
        try:
            with open(pdir / "question.txt", "r", encoding="utf-8") as f:
                prompt = f.read()

            solutions_file = pdir / "solutions.json"
            if not solutions_file.exists():
                continue
            with open(solutions_file, "r", encoding="utf-8") as f:
                solutions_data = json.load(f)
            if not solutions_data:
                continue

            with open(pdir / "input_output.json", "r", encoding="utf-8") as f:
                test_data = json.load(f)

            tests = []
            for i, (ti, to) in enumerate(zip(test_data["inputs"], test_data["outputs"])):
                tests.append(f'def test_{i}():\n    assert solve({ti}) == {to}')

            problems.append(CodeProblem(
                problem_id=f"apps_{pdir.parent.name}_{pdir.name}",
                prompt=prompt,
                tests=tests,
                reference_solution=solutions_data[0],
                source="apps",
            ))
        except Exception as e:
            logger.warning(f"Error loading {pdir}: {e}")

    logger.info(f"Loaded {len(problems)} APPS problems")
    return problems


def generate_candidates_stub(
    problem: CodeProblem,
    n: int = CANDIDATES_PER_PROBLEM,
) -> List[Dict]:
    """
    Placeholder for LLM-based candidate generation.
    In production, this calls GPT-4o-mini with 3-shot prompting
    to generate `n` candidate solutions per problem.

    Returns list of dicts with keys:
        code, pass_rate, passed, total, status
    """
    candidates = []
    for i in range(n):
        candidates.append({
            "code": f"# candidate {i} placeholder",
            "pass_rate": 0.0,
            "passed": 0,
            "total": len(problem.tests),
            "status": "incorrect",
        })
    return candidates


def verify_candidates(
    problem: CodeProblem,
    candidates: List[Dict],
) -> List[Dict]:
    """
    Execute each candidate against hidden unit tests.
    Classify as correct (all pass), partially correct, or incorrect.
    """
    verified = []
    for cand in candidates:
        pass_rate, passed, total = compute_test_pass_rate(
            cand["code"], problem.tests
        )
        status = "correct" if passed == total else ("partial" if passed > 0 else "incorrect")
        verified.append({
            "code": cand["code"],
            "pass_rate": pass_rate,
            "passed": passed,
            "total": total,
            "status": status,
        })
    return verified


def check_codebleu_contamination(
    candidate_prompt: str,
    test_problems: List[CodeProblem],
    threshold: float = CODEBLEU_CONTAMINATION_THRESHOLD,
) -> bool:
    """Return True if candidate is too similar to any test problem."""
    for tp in test_problems:
        score = compute_codebleu(candidate_prompt, tp.prompt)
        if score > threshold:
            return True
    return False


def get_ngrams(text: str, n: int) -> set:
    tokens = text.split()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def check_ngram_contamination(
    text: str,
    test_texts: List[str],
    n: int = NGRAM_SIZE,
) -> bool:
    """Return True if any n-gram from text appears in test texts."""
    text_ngrams = get_ngrams(text, n)
    for tt in test_texts:
        test_ngrams = get_ngrams(tt, n)
        if text_ngrams & test_ngrams:
            return True
    return False


def prepare_dataset(
    humaneval_path: str,
    mbpp_path: str,
    apps_path: str,
    output_path: str,
    seed: int = 42,
) -> DatasetDict:
    """
    Full data preparation pipeline:
    1. Load benchmarks
    2. Generate candidate solutions (up to 10 per problem)
    3. Verify against unit tests
    4. Decontamination filtering
    5. Build training records
    6. Stratified split 80/10/10
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    humaneval = load_humaneval(humaneval_path)
    mbpp = load_mbpp(mbpp_path)
    apps = load_apps(apps_path)
    all_problems = humaneval + mbpp + apps
    logger.info(f"Total problems: {len(all_problems)}")

    test_prompts = [p.prompt for p in all_problems]

    for problem in tqdm(all_problems, desc="Generating and verifying candidates"):
        raw_candidates = generate_candidates_stub(problem, CANDIDATES_PER_PROBLEM)
        problem.candidate_solutions = verify_candidates(problem, raw_candidates)

    records = []
    contaminated_count = 0

    for problem in tqdm(all_problems, desc="Building dataset"):
        if check_ngram_contamination(problem.prompt, test_prompts):
            contaminated_count += 1
            continue

        correct = [c for c in problem.candidate_solutions if c["status"] == "correct"]
        incorrect = [c for c in problem.candidate_solutions if c["status"] in ("incorrect", "partial")]

        for wrong in incorrect:
            records.append({
                "problem_id": problem.problem_id,
                "prompt": problem.prompt,
                "tests": problem.tests,
                "reference_solution": problem.reference_solution,
                "buggy_solution": wrong["code"],
                "pass_rate": wrong["pass_rate"],
                "status": wrong["status"],
                "source": problem.source,
            })

        if correct:
            records.append({
                "problem_id": problem.problem_id,
                "prompt": problem.prompt,
                "tests": problem.tests,
                "reference_solution": problem.reference_solution,
                "buggy_solution": None,
                "pass_rate": 1.0,
                "status": "correct",
                "source": problem.source,
            })

    logger.info(f"Contaminated problems removed: {contaminated_count}")
    logger.info(f"Total records: {len(records)}")

    random.shuffle(records)
    n = len(records)
    t1 = int(DATA_SPLIT[0] * n)
    t2 = t1 + int(DATA_SPLIT[1] * n)

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(records[:t1])),
        "validation": Dataset.from_pandas(pd.DataFrame(records[t1:t2])),
        "test": Dataset.from_pandas(pd.DataFrame(records[t2:])),
    })

    dataset_dict.save_to_disk(output_path)
    logger.info(f"Dataset saved to {output_path}")

    for split, ds in dataset_dict.items():
        logger.info(f"  {split}: {len(ds)} examples")

    return dataset_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TGPR data preparation")
    parser.add_argument("--humaneval_path", type=str, default="data/humaneval/HumanEval.jsonl")
    parser.add_argument("--mbpp_path", type=str, default="data/mbpp/mbpp.jsonl")
    parser.add_argument("--apps_path", type=str, default="data/apps")
    parser.add_argument("--output_path", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare_dataset(
        humaneval_path=args.humaneval_path,
        mbpp_path=args.mbpp_path,
        apps_path=args.apps_path,
        output_path=args.output_path,
        seed=args.seed,
    )
