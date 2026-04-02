from anthropic import AsyncAnthropic
import logging

from src.schemas import RouterConfig, LLMResponse
from src.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def async_generate(self, prompt: str, config: RouterConfig) -> LLMResponse:
        logger.info(f"Routing request to Anthropic using model: {config.model}")
        
        # Anthropic's API structure is slightly different from OpenAI's
        response = await self.client.messages.create(
            model=config.model,
            max_tokens=1024, # Anthropic requires max_tokens to be explicitly set
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
        )

        content = response.content[0].text
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        
        cost = self.calculate_cost(prompt_tokens, completion_tokens, config.model)

        return LLMResponse(
            content=content,
            provider_used="anthropic",
            model_used=config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_estimate=cost
        )

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
        pricing = {
            "claude-3-opus-20240229": {"prompt": 15.0, "completion": 75.0},
            "claude-3-5-sonnet-20240620": {"prompt": 3.0, "completion": 15.0},
            "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25}
        }
        rates = pricing.get(model_name, {"prompt": 0.0, "completion": 0.0})
        cost = (prompt_tokens / 1_000_000) * rates["prompt"] + \
               (completion_tokens / 1_000_000) * rates["completion"]
        return round(cost, 6)