#!/usr/bin/env python3
import time
import logging
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeGenerator:
    """Base class for code generation using language models."""
    
    def __init__(self, 
                 model_name_or_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.1):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        logger.info(f"Loading model {model_name_or_path} on {device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def generate_code(self, prompt: str, **kwargs) -> str:
        """Generate code based on a prompt."""
        # Override generation parameters if provided
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', self.top_p)
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
        repetition_penalty = kwargs.get('repetition_penalty', self.repetition_penalty)
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the output
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        response = output_text[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {len(response)} chars in {generation_time:.2f}s")
        
        return self._extract_code(response)
    
    def _extract_code(self, text: str) -> str:
        """Extract code from generated text, handling different formats."""
        # Look for Python code blocks
        import re
        
        # Try to find code blocks with ```python ... ``` format
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)
        
        if matches:
            # Return the longest code block (assuming it's the most complete)
            return max(matches, key=len).strip()
        
        # If no code blocks found, return the entire text
        return text.strip()
    
    def refine_code(self, 
                    original_code: str, 
                    problem_description: str, 
                    feedback: Optional[str] = None,
                    **kwargs) -> str:
        """Refine code based on original code, problem description, and optional feedback."""
        # Construct the prompt for refinement
        prompt = self._construct_refinement_prompt(original_code, problem_description, feedback)
        
        # Generate the refined code
        refined_code = self.generate_code(prompt, **kwargs)
        
        return refined_code
    
    def _construct_refinement_prompt(self, 
                                    original_code: str, 
                                    problem_description: str, 
                                    feedback: Optional[str] = None) -> str:
        """Construct a prompt for code refinement."""
        prompt = f"# Problem Description\n{problem_description}\n\n"
        
        if feedback:
            prompt += f"# Feedback on Previous Solution\n{feedback}\n\n"
        
        prompt += f"# Original Code\n```python\n{original_code}\n```\n\n"
        prompt += "# Refined Solution\nRefine the code to address the issues mentioned in the feedback and improve its correctness and efficiency.\n```python\n"
        
        return prompt


class QwenCodeGenerator(CodeGenerator):
    """Code generator using Qwen models."""
    
    def __init__(self, 
                 model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Coder-Instruct",
                 **kwargs):
        super().__init__(model_name_or_path, **kwargs)
    
    def _construct_refinement_prompt(self, 
                                    original_code: str, 
                                    problem_description: str, 
                                    feedback: Optional[str] = None) -> str:
        """Construct a prompt for code refinement using Qwen's format."""
        prompt = "<|im_start|>system\nYou are a helpful AI assistant. You excel at writing clean, efficient, and correct Python code.\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n# Problem Description\n{problem_description}\n\n"
        
        if feedback:
            prompt += f"# Feedback on Previous Solution\n{feedback}\n\n"
        
        prompt += f"# Original Code\n```python\n{original_code}\n```\n\n"
        prompt += "Please refine the code to improve its correctness, efficiency, and readability. Provide only the refined Python code without explanations.\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n```python\n"
        
        return prompt
    
    def _extract_code(self, text: str) -> str:
        """Extract code from Qwen's response format."""
        # First try to extract code blocks
        import re
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)
        
        if matches:
            return max(matches, key=len).strip()
        
        # If no code blocks, return everything up to the end marker if present
        end_marker = "<|im_end|>"
        if end_marker in text:
            return text.split(end_marker)[0].strip()
        
        return text.strip()


class CodeLlamaGenerator(CodeGenerator):
    """Code generator using CodeLlama models."""
    
    def __init__(self, 
                 model_name_or_path: str = "codellama/CodeLlama-7b-Instruct-hf",
                 **kwargs):
        super().__init__(model_name_or_path, **kwargs)
    
    def _construct_refinement_prompt(self, 
                                    original_code: str, 
                                    problem_description: str, 
                                    feedback: Optional[str] = None) -> str:
        """Construct a prompt for code refinement using CodeLlama's format."""
        prompt = "<s>[INST] "
        prompt += f"# Problem Description\n{problem_description}\n\n"
        
        if feedback:
            prompt += f"# Feedback on Previous Solution\n{feedback}\n\n"
        
        prompt += f"# Original Code\n```python\n{original_code}\n```\n\n"
        prompt += "Please refine the code to improve its correctness, efficiency, and readability. Provide only the refined Python code. [/INST]\n\n```python\n"
        
        return prompt


class DeepseekCoderGenerator(CodeGenerator):
    """Code generator using DeepSeek Coder models."""
    
    def __init__(self, 
                 model_name_or_path: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
                 **kwargs):
        super().__init__(model_name_or_path, **kwargs)
    
    def _construct_refinement_prompt(self, 
                                    original_code: str, 
                                    problem_description: str, 
                                    feedback: Optional[str] = None) -> str:
        """Construct a prompt for code refinement using DeepSeek's format."""
        prompt = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For questions not related to computer science, you will refuse to answer.\n\n"
        prompt += f"User: # Problem Description\n{problem_description}\n\n"
        
        if feedback:
            prompt += f"# Feedback on Previous Solution\n{feedback}\n\n"
        
        prompt += f"# Original Code\n```python\n{original_code}\n```\n\n"
        prompt += "Please refine the code to improve its correctness, efficiency, and readability. Provide only the refined Python code without explanations.\n\n"
        
        prompt += "Assistant: ```python\n"
        
        return prompt 