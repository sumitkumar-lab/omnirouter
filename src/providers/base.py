"""
The abstract blueprint every provider must follow
"""
from abc import ABC, abstractmethod
from typing import Optional
from src.schemas import RouterConfig, LLMResponse

class BaseLLMProvider(ABC):
    """
    The strict blueprint that ALL LLM providers must follow.
    If a developer tries to create a provider without an 'async_generate' method,
    Python will throw a TypeError upon instantiation.
    """
    
    def __init__(self, api_key: str):
        # Every provider needs an API key (or a dummy key for local models)
        self.api_key = api_key

    @abstractmethod
    async def async_generate(
        self, 
        prompt: str, 
        config: RouterConfig
    ) -> LLMResponse:
        """
        The core engine method. 
        Takes a string prompt and our strict RouterConfig.
        MUST return our strictly typed LLMResponse.
        """
        pass
        
    @abstractmethod
    def calculate_cost(
        self, 
        prompt_tokens: int, 
        completion_tokens: int, 
        model_name: str
    ) -> float:
        """
        Calculates the estimated cost of the API call.
        Essential for production monitoring.
        """
        pass