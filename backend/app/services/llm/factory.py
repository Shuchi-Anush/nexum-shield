"""Config-driven LLM provider factory.

Resolution order for `get_llm_provider(provider=None)`:

  1. Explicit `provider` argument        ("gemini", "claude", …)
  2. Settings.LLM_PROVIDER                (from env / .env)
  3. Default: "gemini"

The returned instance is cached per provider name (`functools.lru_cache`)
so callers reuse the same `httpx.AsyncClient` and share the in-flight
semaphore. Tests can clear the cache with `get_llm_provider.cache_clear()`
or register a mock via `register_provider`.

Adding a new provider:

  1. Implement `BaseLLMProvider` in `providers/<name>.py`.
  2. Add one line to `providers/__init__.py` calling `register_provider`.

`factory.py` itself stays untouched.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.llm.base import BaseLLMProvider


_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {}


def register_provider(name: str, cls: type[BaseLLMProvider]) -> None:
    """Register a provider class under a canonical lowercase name."""
    if not name:
        raise ValueError("provider name must be a non-empty string")
    _PROVIDER_REGISTRY[name.lower()] = cls


def registered_providers() -> list[str]:
    """Return the sorted list of currently registered provider names."""
    _ensure_loaded()
    return sorted(_PROVIDER_REGISTRY.keys())


def _ensure_loaded() -> None:
    """Import the providers package so each module self-registers.

    Done lazily to avoid an import cycle: providers/__init__.py imports
    register_provider from this module.
    """
    import app.services.llm.providers  # noqa: F401  (side-effect: registration)


@lru_cache(maxsize=None)
def get_llm_provider(provider: str | None = None) -> BaseLLMProvider:
    """Resolve and instantiate an LLM provider (cached per name).

    Raises:
        ValueError      — unknown provider name
        LLMAuthError    — required API key missing (raised by the provider's
                          constructor — surfaced unchanged)
    """
    _ensure_loaded()

    settings = get_settings()
    name = (provider or settings.LLM_PROVIDER or "gemini").lower()

    cls = _PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown LLM provider: {name!r} "
            f"(registered: {sorted(_PROVIDER_REGISTRY.keys())})"
        )
    return cls.from_settings(settings)
