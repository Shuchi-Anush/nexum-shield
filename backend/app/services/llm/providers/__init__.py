"""Provider registry side-effects.

Importing this package registers every shipped provider with the factory.
Add new providers here on a single line so `factory.py` stays untouched.
"""

from app.services.llm.factory import register_provider
from app.services.llm.providers.gemini import GeminiProvider

register_provider("gemini", GeminiProvider)

__all__ = ["GeminiProvider"]
