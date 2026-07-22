from .base_model import BaseModel

# Optional provider imports: missing third-party SDKs should not block
# unrelated providers (for example running only Qwen).
try:
    from .openai_model import OpenAIModel
except Exception:  # pragma: no cover
    OpenAIModel = None

try:
    from .deepseek_model import DeepSeekModel
except Exception:  # pragma: no cover
    DeepSeekModel = None

try:
    from .kimi_model import KimiModel
except Exception:  # pragma: no cover
    KimiModel = None

try:
    from .qwen_model import QwenModel
except Exception:  # pragma: no cover
    QwenModel = None

try:
    from .doubao_model import DoubaoModel
except Exception:  # pragma: no cover
    DoubaoModel = None

try:
    from .zhipu_model import ZhipuModel
except Exception:  # pragma: no cover
    ZhipuModel = None

__all__ = [
    'BaseModel', 'OpenAIModel', 'DeepSeekModel', 'KimiModel', 'QwenModel',
    'DoubaoModel', 'ZhipuModel'
]
