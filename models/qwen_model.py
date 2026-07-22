"""
Qwen Model (Tongyi Qianwen, uses OpenAI compatible interface)
"""
from typing import Optional
from openai import OpenAI
from .base_model import BaseModel


class QwenModel(BaseModel):
    """Qwen API Model"""
    
    def __init__(self, model_name: str = "qwen-plus", api_key: str = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", **kwargs):
        self.enable_thinking = kwargs.pop('enable_thinking', False)
        self.thinking_tokens = kwargs.pop('thinking_tokens', None)
        super().__init__(model_name, api_key, base_url, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=600.0
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Call Qwen API to generate response
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Extra arguments
                - enable_thinking: bool, whether to enable thinking mode
            
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
                "temperature": kwargs.get('temperature', self.temperature),
            }

            extra_body = {}

            # Official DashScope OpenAI-compatible usage passes Qwen thinking controls
            # through `extra_body`.
            enable_thinking = self.enable_thinking or kwargs.get('enable_thinking', False)
            if enable_thinking:
                extra_body['enable_thinking'] = True
            elif kwargs.get('enable_thinking', None) is False or self.enable_thinking is False:
                extra_body['enable_thinking'] = False

            # Keep the existing config field name but map it to Qwen's
            # official `thinking_budget` parameter.
            thinking_tokens = kwargs.get('thinking_tokens') or getattr(self, 'thinking_tokens', None)
            if thinking_tokens:
                extra_body['thinking_budget'] = thinking_tokens

            if extra_body:
                api_params['extra_body'] = extra_body
            
            # DashScope thinking usage is reported most reliably in streaming mode.
            if 'qwq' in self.model_name.lower() or enable_thinking:
                api_params['stream'] = True
                api_params['stream_options'] = {'include_usage': True}
                try:
                    response = self.client.chat.completions.create(**api_params)
                    full_content = ""
                    self.last_input_tokens = None
                    for chunk in response:
                        token_count = self._extract_input_tokens_from_response(chunk)
                        if token_count is not None:
                            self.last_input_tokens = token_count
                        choices = getattr(chunk, 'choices', None) or []
                        if not choices:
                            continue
                        delta = getattr(choices[0], 'delta', None)
                        if delta is not None and hasattr(delta, 'content') and delta.content:
                            full_content += delta.content
                    return full_content.strip() if full_content else ""
                except Exception as stream_error:
                    raise Exception(f"Streaming failed: {str(stream_error)}")
            else:
                response = self.client.chat.completions.create(**api_params)
                self.last_input_tokens = self._extract_input_tokens_from_response(response)
                content = response.choices[0].message.content
                return content.strip() if content else ""
            
        except Exception as e:
            raise Exception(f"Qwen API call failed: {str(e)}")
