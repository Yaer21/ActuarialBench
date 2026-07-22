"""
DeepSeek Model (Using OpenAI Compatible Interface)
"""
from openai import OpenAI
from .base_model import BaseModel


class DeepSeekModel(BaseModel):
    """DeepSeek API Model"""
    
    def __init__(self, model_name: str = "deepseek-v4-flash", api_key: str = None, 
                 base_url: str = "https://api.deepseek.com", **kwargs):
        self.enable_thinking = kwargs.pop("enable_thinking", None)
        self.reasoning_effort = kwargs.pop("reasoning_effort", None)
        super().__init__(model_name, api_key, base_url, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=600.0
        )
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Call DeepSeek API to generate response
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt, uses default ACTUARIAL_SYSTEM_PROMPT if not provided
            **kwargs: Additional arguments
                - return_reasoning: bool, whether to return reasoning process when provided by the API
            
        Returns:
            Generated text from the model
        """
        sys_prompt = system_prompt if system_prompt is not None else self.ACTUARIAL_SYSTEM_PROMPT
        
        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
            }

            enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
            if enable_thinking is not None:
                completion_params["extra_body"] = {
                    "thinking": {
                        "type": "enabled" if enable_thinking else "disabled"
                    }
                }
                reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
                if enable_thinking and reasoning_effort:
                    completion_params["reasoning_effort"] = reasoning_effort

            # DeepSeek's thinking mode ignores sampling parameters. Keep temperature
            # only for non-thinking calls or when no explicit thinking mode is set.
            if enable_thinking is not True:
                completion_params["temperature"] = kwargs.get('temperature', self.temperature)

            response = self.client.chat.completions.create(**completion_params)
            self.last_input_tokens = self._extract_input_tokens_from_response(response)
            
            message = response.choices[0].message
            content = message.content.strip()
            
            reasoning = getattr(message, 'reasoning_content', None)
            if reasoning and kwargs.get('return_reasoning', False):
                return f"[Reasoning Process]\n{reasoning}\n\n[Final Answer]\n{content}"
            
            return content
            
        except Exception as e:
            raise Exception(f"DeepSeek API call failed: {str(e)}")
