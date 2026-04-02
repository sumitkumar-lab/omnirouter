from openai import AsyncOpenAI
import logging

# Import our strict schemas and base blueprint
from src.schemas import RouterConfig, LLMResponse
from src.providers.base import BaseLLMProvider

# Set up logging for professional debugging
logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """
    The concrete implementation for OpenAI's API.
    How it strictly fulfills the contract defined in BaseLLMProvider.
    """
    
    def __init__(self, api_key: str):
        # Call the parent class initialization
        super().__init__(api_key)
        
        # CRITICAL: We initialize the ASYNC client, not the standard synchronous one.
        # This is what allows our router to handle hundreds of concurrent requests.
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def async_generate(self, prompt: str, config: RouterConfig) -> LLMResponse:
        logger.info(f"Routing request to OpenAI using model: {config.model}")
        
        # 1. Execute the Async API Call
        response = await self.client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            # We will add top_p, frequency_penalty, etc. later as needed
        )

        # 2. Extract Data from OpenAI's specific object structure
        content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # 3. Calculate Cost dynamically
        cost = self.calculate_cost(prompt_tokens, completion_tokens, config.model)

        # 4. Standardize the Output
        # We transform OpenAI's proprietary response into our universal LLMResponse schema.
        # Now, the rest of our application doesn't need to know anything about OpenAI's specific formatting.
        return LLMResponse(
            content=content,
            provider_used="openai",
            model_used=config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_estimate=cost
        )

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
        """
        Calculates the exact cost of the API call based on OpenAI's pricing (per 1M tokens).
        This is a massive value-add for your open-source repository.
        """
        # A dictionary acting as a simple pricing database
        pricing = {
            "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
            "gpt-4o": {"prompt": 5.0, "completion": 15.0},
            "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5}
        }
        
        # If they use a model not in our dict, default to 0.0 to prevent crashes
        rates = pricing.get(model_name, {"prompt": 0.0, "completion": 0.0})
        
        # Math: (tokens / 1,000,000) * rate
        cost = (prompt_tokens / 1_000_000) * rates["prompt"] + \
               (completion_tokens / 1_000_000) * rates["completion"]
        
        return round(cost, 6)