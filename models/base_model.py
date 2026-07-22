"""
Abstract Base Class - Unified Interface for All Models
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from bench_runner.prompts import SYSTEM_PROMPTS, MCQ_FEWSHOT_EXAMPLES


class BaseModel(ABC):
    """LLM Model Abstract Base Class"""
    
    ACTUARIAL_CHOICE_SYSTEM_PROMPT = SYSTEM_PROMPTS["mcq"]
    ACTUARIAL_SYSTEM_PROMPT = ACTUARIAL_CHOICE_SYSTEM_PROMPT
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 base_url: str,
                 temperature: float = 1,
                 timeout: Optional[int] = None,
                 **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.kwargs = kwargs
        self.last_input_tokens = None
        self.retry_times = kwargs.get("retry_times")

    def _extract_input_tokens_from_response(self, response) -> Optional[int]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return None

        for key in ("prompt_tokens", "input_tokens"):
            value = getattr(usage, key, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

        if hasattr(usage, "model_dump"):
            try:
                data = usage.model_dump()
                for key in ("prompt_tokens", "input_tokens"):
                    if data.get(key) is not None:
                        return int(data[key])
            except Exception:
                return None
        return None
        
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate response
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt, uses default ACTUARIAL_SYSTEM_PROMPT if not provided
            **kwargs: Additional arguments
            
        Returns:
            Generated text from the model
        """
        pass
    
    def generate_with_retry(self, prompt: str, retry_times: int = 3, **kwargs) -> Optional[str]:
        """
        Generate with retry mechanism
        
        Args:
            prompt: Input prompt
            retry_times: Number of retries
            **kwargs: Additional arguments
            
        Returns:
            Generated text from the model, None if failed
        """
        retry_limit = int(self.retry_times or retry_times)
        for attempt in range(retry_limit):
            try:
                response = self.generate(prompt, **kwargs)
                return response
            except Exception as e:
                print(f"  Warning: Attempt {attempt + 1}/{retry_limit} failed: {str(e)}")
                if attempt < retry_limit - 1:
                    wait_time = 2 ** attempt
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  Max retries reached, giving up")
                    return None
        return None

    def generate_with_details(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, str]:
        """Generate with best-effort separation of reasoning and final response.

        Default behavior:
        - call `generate()` to get a single text
        - split into (reasoning, final) via `split_reasoning_and_final()` as a fallback

        Model wrappers can override this to use native fields (e.g., thinking/reasoning blocks).
        """

        raw = self.generate(prompt, system_prompt=system_prompt, **kwargs) or ""
        reasoning, final_text = self.split_reasoning_and_final(raw)
        return {
            "raw_response": raw,
            "reasoning": reasoning,
            "final_response": final_text,
            "input_tokens": self.last_input_tokens,
        }

    def supports_spreadsheet_attachments(self) -> bool:
        """Whether this model wrapper can receive a real `.xlsx` file."""
        return False

    def spreadsheet_attachment_support_message(self) -> str:
        return (
            f"{self.__class__.__name__} does not implement direct `.xlsx` attachment "
            "delivery in the current API wrapper."
        )

    def generate_with_retry_details(self, prompt: str, retry_times: int = 3, **kwargs) -> Dict[str, str]:
        """Retry wrapper for `generate_with_details()`.

        Returns empty strings on total failure.
        """

        retry_limit = int(self.retry_times or retry_times)
        for attempt in range(retry_limit):
            try:
                return self.generate_with_details(prompt, **kwargs)
            except Exception as e:
                print(f"  Warning: Attempt {attempt + 1}/{retry_limit} failed: {str(e)}")
                if attempt < retry_limit - 1:
                    wait_time = 2 ** attempt
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  Max retries reached, giving up")
                    return {"raw_response": "", "reasoning": "", "final_response": ""}

        return {"raw_response": "", "reasoning": "", "final_response": ""}
    
    def format_fewshot_prompt(self, 
                             train_examples: List[Dict],
                             test_question: Dict,
                             num_fewshot: int = 2) -> str:
        """
        Build Few-shot prompt
        
        Args:
            train_examples: List of training examples
            test_question: Test question
            num_fewshot: Number of few-shot examples
            
        Returns:
            Complete prompt string
        """
        prompt = ""

        if num_fewshot > 0:
            prompt += "Here are some example questions and their answers:\n\n"

            if not train_examples:
                train_examples = MCQ_FEWSHOT_EXAMPLES
            
            for i, example in enumerate(train_examples[:num_fewshot]):
                prompt += f"Example {i+1}:\n"
                prompt += f"{example['question']}\n\n"
                prompt += "Options:\n"
                
                options = example['options']
                if isinstance(options, dict):
                    for key, value in options.items():
                        prompt += f"{key}) {value}\n"
                elif isinstance(options, list):
                    for j, option in enumerate(options):
                        prompt += f"{chr(65+j)}) {option}\n"
                
                if 'explanation' in example and example['explanation']:
                    prompt += f"\nAnalysis: {example['explanation']}\n"
                
                prompt += f"\n**Answer: {example['answer']}**\n"
                prompt += "=" * 50 + "\n\n"
            
            prompt += "Now solve this question following the same format:\n\n"
        
        prompt += f"{test_question['question']}\n\n"
        prompt += "Options:\n"
        
        options = test_question['options']
        if isinstance(options, dict):
            for key, value in options.items():
                prompt += f"{key}) {value}\n"
        elif isinstance(options, list):
            for j, option in enumerate(options):
                prompt += f"{chr(65+j)}) {option}\n"
        
        prompt += "\n**Answer (respond with exactly one option letter):**"
        
        return prompt

    def split_reasoning_and_final(self, response: str) -> tuple[str, str]:
        """Split a model response into (reasoning_text, final_text).

        Many reasoning models output internal thoughts in tags like <think>...</think>.
        If there is no explicit reasoning section, reasoning_text is "" and final_text is the full response.
        """

        import re

        if not response:
            return "", ""

        text = str(response).strip()

        # Common explicit thinking tags.
        for tag in ("think", "thinking"):
            m = re.search(rf"(?is)<{tag}>\s*(.*?)\s*</{tag}>", text)
            if m:
                reasoning = m.group(1).strip()
                final_text = (text[: m.start()] + text[m.end() :]).strip()
                return reasoning, final_text

        # Reasoning/final markers used by several reasoning models.
        m = re.search(
            r"(?is)^(?:reasoning|thoughts?|analysis)\s*:?\s*(.+?)\n+\s*(?:final|answer)\s*:?\s*(.+)$",
            text,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()

        # A final marker without a reasoning label still gives a useful split.
        m = re.search(r"(?is)\n\s*(?:final|answer)\s*:?\s*(.+)$", text)
        if m:
            final_text = m.group(1).strip()
            reasoning = text[: m.start()].strip()
            return reasoning, final_text

        return "", text
    
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract answer (A/B/C/D/E) from model output
        
        Args:
            response: Model output text
            
        Returns:
            Extracted answer letter, None if failed
        """
        import re
        
        if not response:
            return None
        
        response = response.strip().upper()

        if len(response) <= 5:
            for char in response:
                if char in 'ABCDE':
                    return char
            return None

        match = re.search(r'\*\*([ABCDE])\*\*\s*$', response)
        if match:
            return match.group(1)

        match = re.search(r'\n+([ABCDE])\s*$', response)
        if match:
            return match.group(1)

        match = re.search(r'(?:ANSWER)\s*:?\s*([ABCDE])', response)
        if match:
            return match.group(1)

        matches = list(re.finditer(r'\b([ABCDE])\b', response))
        if matches:
            return matches[-1].group(1)

        for char in response:
            if char in 'ABCDE':
                return char
        
        return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
