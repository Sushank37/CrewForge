"""Model provider abstraction and concrete adapters."""

from crewforge.providers.base import (
    ModelProvider,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    StructuredGenerationResult,
    TextGenerationResult,
)
from crewforge.providers.ollama_provider import OllamaProvider
from crewforge.providers.openai_provider import OpenAIProvider

__all__ = [
    "ModelProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderConfigurationError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "StructuredGenerationResult",
    "TextGenerationResult",
]
