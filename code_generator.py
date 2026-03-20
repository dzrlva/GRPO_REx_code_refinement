#!/usr/bin/env python3
"""
Code generator for TGPR framework.

Uses Qwen2.5-7B-Instruct as the policy model (Section 4.1).
Generation parameters: temperature 0.8, top_p 0.95, top_k 50.
"""

import re
import logging
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50
MAX_NEW_TOKENS = 512


class QwenCodeGenerator:
    """
    Code generator using Qwen2.5-7B-Instruct.
    Provides generate_code and refine_code interfaces
    used by RefinementTree and GRPOTrainer.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        logger.info(f"Loading {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Model loaded")

    def generate_code(self, prompt: str, **kwargs) -> str:
        """Generate code from a prompt."""
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return self._extract_code(generated)

    def refine_code(
        self,
        original_code: str,
        problem_description: str,
        feedback: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Refine code given problem description and optional feedback."""
        prompt = self._build_prompt(original_code, problem_description, feedback)
        return self.generate_code(prompt, **kwargs)

    def _build_prompt(
        self,
        code: str,
        problem: str,
        feedback: Optional[str] = None,
    ) -> str:
        """Build refinement prompt in Qwen chat format."""
        system = "You are an expert programmer. Fix bugs and improve code correctness."
        user = f"Problem:\n{problem}\n\n"
        if feedback:
            user += f"Feedback:\n{feedback}\n\n"
        user += f"Code:\n```python\n{code}\n```\n\nFixed code:\n```python\n"

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n```python\n"
        )

    def _extract_code(self, text: str) -> str:
        """Extract Python code from generated text."""
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)
        if matches:
            return max(matches, key=len).strip()

        end_marker = "<|im_end|>"
        if end_marker in text:
            return text.split(end_marker)[0].strip()

        return text.strip()
