"""Public surface of the LLM service layer.

Callers import from here only:

    from app.services.llm import (
        get_llm_provider,
        LLMRequest, LLMResponse, LLMMessage, LLMRole,
        LLMProviderError, LLMTimeoutError, LLMRateLimitError, LLMAuthError,
    )
"""

from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import (
    LLMAuthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.services.llm.factory import (
    get_llm_provider,
    register_provider,
    registered_providers,
)
from app.services.llm.schemas import (
    LLMErrorKind,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMTokenUsage,
)

__all__ = [
    # Schemas
    "LLMRequest",
    "LLMResponse",
    "LLMMessage",
    "LLMRole",
    "LLMTokenUsage",
    "LLMErrorKind",
    # Factory
    "get_llm_provider",
    "register_provider",
    "registered_providers",
    # Base
    "BaseLLMProvider",
    # Exceptions
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMAuthError",
]
