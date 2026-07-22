"""
Kimi Model (Moonshot AI, uses OpenAI compatible interface)
"""
import time
from typing import Optional
from openai import OpenAI
from .base_model import BaseModel


class KimiModel(BaseModel):
    """Kimi API Model"""
    
    def __init__(self, model_name: str = "kimi-latest", api_key: str = None,
                 base_url: str = "https://api.moonshot.cn/v1", **kwargs):
        self.deep_think = kwargs.pop('deep_think', False)
        self.force_stream = kwargs.pop('force_stream', False)
        self.stream_timeout_returns_partial = kwargs.pop('stream_timeout_returns_partial', False)
        super().__init__(model_name, api_key, base_url, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(kwargs.get("timeout", 600.0) or 600.0)
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Call Kimi API to generate response
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Extra arguments
                - deep_think: bool, whether to enable deep thinking mode
            
        Returns:
            Generated text from the model
        """
        sys_prompt = system_prompt if system_prompt is not None else self.ACTUARIAL_SYSTEM_PROMPT
        
        try:
            api_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
            }

            is_k2_6_family = self.model_name.startswith(("kimi-k2.6", "kimi-k2.5"))
            if not is_k2_6_family:
                api_params["temperature"] = kwargs.get('temperature', self.temperature)
            
            if kwargs.get('use_search', False):
                api_params['use_search'] = True
                
            # Moonshot Kimi K2.6 thinking control is passed as provider-specific
            # extra_body. Keep the config field name `deep_think` for compatibility.
            deep_think = kwargs.get('deep_think') if 'deep_think' in kwargs else getattr(self, 'deep_think', None)
            if deep_think is not None:
                api_params['extra_body'] = {
                    'thinking': {
                        'type': 'enabled' if deep_think else 'disabled'
                    }
                }
            if is_k2_6_family:
                api_params["max_tokens"] = kwargs.get("max_tokens", 32768)
            if kwargs.get("force_stream", self.force_stream):
                pieces = []
                try:
                    stream = self.client.chat.completions.create(stream=True, **api_params)
                    started = time.perf_counter()
                    for chunk in stream:
                        if self.timeout and time.perf_counter() - started >= float(self.timeout):
                            break
                        if not getattr(chunk, "choices", None):
                            continue
                        delta = getattr(chunk.choices[0], "delta", None)
                        content = getattr(delta, "content", None)
                        if content:
                            pieces.append(content)
                    return "".join(pieces).strip()
                except Exception:
                    if kwargs.get("stream_timeout_returns_partial", self.stream_timeout_returns_partial) and pieces:
                        return "".join(pieces).strip()
                    raise
            response = self.client.chat.completions.create(**api_params)
            self.last_input_tokens = self._extract_input_tokens_from_response(response)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f'Kimi API call failed: {str(e)}')
