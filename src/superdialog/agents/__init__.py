"""Alternative :class:`Agent` implementations (non-DialogMachine brains)."""

from .langchain_agent import LangChainAgent
from .llm_agent import LLMAgent

__all__ = ["LLMAgent", "LangChainAgent"]
