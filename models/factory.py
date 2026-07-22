import os
import yaml
from typing import Optional, Dict, Any
from models import (
    OpenAIModel, DeepSeekModel, KimiModel, QwenModel,
    DoubaoModel, ZhipuModel
)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_api_key(provider: str, api_key_env: Optional[str] = None) -> str:
    """Get API key from environment variables."""
    provider = (provider or "").lower()
    if provider == 'ollama':
        return 'ollama'
    if provider == 'vllm':
        return os.getenv(api_key_env or 'VLLM_API_KEY', 'EMPTY')
    
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if api_key:
            return api_key
            
    # Provider-specific fallback when config does not name an env var.
    mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'kimi': 'MOONSHOT_API_KEY',
        'qwen': 'DASHSCOPE_API_KEY',
        'doubao': 'DOUBAO_API_KEY',
        'zhipu': 'ZHIPU_API_KEY',
        'siliconflow': 'SILICONFLOW_API_KEY',
    }
    env_var = mapping.get(provider)
    if env_var:
        return os.getenv(env_var, "")
    return ""

def create_model(model_key: str, config: Dict[str, Any]):
    """Factory function to create a model instance based on configuration."""
    if model_key not in config['models']:
        if ":" in model_key:
            print(f"Initializing local Ollama model by name: {model_key}")
            return OpenAIModel(
                model_key,
                "ollama",
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                temperature=1,
                reasoning_effort="high",
            )
        raise ValueError(f"Model '{model_key}' not found in config.")
        
    model_config = config['models'][model_key]
    provider = model_config.get('provider', 'openai').lower()
    api_model_name = model_config.get('model_name', model_key)
    api_key_env = model_config.get('api_key_env')
    api_key = get_api_key(provider, api_key_env)
    
    base_url = model_config.get('base_url') or model_config.get('api_base')
    temperature = model_config.get('temperature', 0)
    platform = model_config.get('platform')
    timeout = model_config.get('timeout')
    retry_times = model_config.get('retry_times')
    
    reasoning_effort = model_config.get('reasoning_effort')
    thinking_level = model_config.get('thinking_level')
    enable_thinking = model_config.get('enable_thinking')
    thinking_tokens = model_config.get('thinking_tokens')
    deep_think = model_config.get('deep_think')

    print(f"Initializing Model: {model_key} ({provider})")

    # Proxy platforms expose several providers through an OpenAI-compatible endpoint.
    if platform in {'ChatAnywhere', 'CaMeL AI', 'KR777'}:
        return OpenAIModel(
            api_model_name,
            api_key,
            base_url,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            thinking_level=thinking_level,
            thinking_tokens=thinking_tokens,
            enable_thinking=enable_thinking,
            thinking_type=model_config.get('thinking_type'),
            output_effort=model_config.get('output_effort'),
            api_mode=model_config.get('api_mode'),
            force_stream=model_config.get('force_stream', False),
            chat_template_kwargs=model_config.get('chat_template_kwargs'),
            timeout=timeout,
            retry_times=retry_times,
        )

    if provider == 'openai':
        return OpenAIModel(
            api_model_name,
            api_key,
            base_url,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            api_mode=model_config.get('api_mode'),
            force_stream=model_config.get('force_stream', False),
            chat_template_kwargs=model_config.get('chat_template_kwargs'),
            timeout=timeout,
            retry_times=retry_times,
        )
    elif provider == 'deepseek':
        return DeepSeekModel(api_model_name, api_key, base_url=base_url, temperature=temperature, enable_thinking=enable_thinking, reasoning_effort=reasoning_effort)
    elif provider == 'kimi':
        return KimiModel(
            api_model_name,
            api_key,
            base_url=base_url,
            temperature=temperature,
            deep_think=deep_think,
            timeout=timeout,
            retry_times=retry_times,
            force_stream=model_config.get('force_stream', False),
            stream_timeout_returns_partial=model_config.get('stream_timeout_returns_partial', False),
        )
    elif provider == 'qwen':
        return QwenModel(api_model_name, api_key, temperature=temperature, enable_thinking=enable_thinking, thinking_tokens=thinking_tokens)
    elif provider == 'doubao':
        return DoubaoModel(api_model_name, api_key, temperature=temperature, enable_thinking=enable_thinking, reasoning_effort=reasoning_effort)
    elif provider == 'zhipu':
        return ZhipuModel(api_model_name, api_key, temperature=temperature, enable_thinking=enable_thinking)
    elif provider == 'ollama':
        return OpenAIModel(
            api_model_name,
            api_key or "ollama",
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    elif provider == 'siliconflow':
        return OpenAIModel(
            api_model_name,
            api_key,
            base_url or "https://api.siliconflow.com/v1",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )
    elif provider == 'vllm':
        return OpenAIModel(
            api_model_name,
            api_key or "EMPTY",
            base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
            chat_template_kwargs=model_config.get('chat_template_kwargs'),
            timeout=timeout,
            retry_times=retry_times,
        )
    else:
        print(f"Warning: Unknown provider '{provider}', falling back to OpenAIModel.")
        return OpenAIModel(api_model_name, api_key, base_url, temperature=temperature, reasoning_effort=reasoning_effort, enable_thinking=enable_thinking)
