import asyncio
import logging
from typing import Dict, Any

# Import our schemas and providers
from src.schemas import RouterConfig, LLMResponse
from src.providers.base import BaseLLMProvider
from src.providers.openai_client import OpenAIProvider
from src.providers.anthropic_client import AnthropicProvider

# Set up logging so we can see the retries happening in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OmniRouter:
    """
    The central routing engine. 
    Handles provider selection, retries, and error management.
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        We initialize the router with a dictionary of API keys.
        We then map string names (like 'openai') to their concrete class instances.
        """
        self.providers: Dict[str, BaseLLMProvider] = {}
        
        # If the user passed an OpenAI key, activate the OpenAI provider
        if "openai" in api_keys:
            self.providers["openai"] = OpenAIProvider(api_key=api_keys["openai"])
            
        #---Register Anthropic
        if "anthropic" in api_keys:
            self.providers["anthropic"] = AnthropicProvider(api_key=api_keys["anthropic"])
        # We will add others here later

    async def generate(self, prompt: str, config: RouterConfig) -> LLMResponse:
        """
        The main entry point. Routes the prompt to the correct provider with retries.
        """
        # 1. Check if the requested provider actually exists in our dictionary
        provider = self.providers.get(config.provider)
        if not provider:
            raise ValueError(f"Provider '{config.provider}' is not configured.")

        last_exception = None
        
        # 2. PRIMARY RETRY LOOP
        for attempt in range(config.max_retries):
            try:
                # If this is a retry, log it
                if attempt > 0:
                    logger.info(f"[{config.provider}] Retrying... Attempt {attempt + 1} of {config.max_retries}")
                
                # 3. The actual API call to whatever provider is currently selected
                response = await provider.async_generate(prompt, config)
                return response
                
            except Exception as e:
                # If the API crashes, we catch it here instead of crashing the app
                logger.warning(f"[{config.provider}] Attempt {attempt + 1} failed with error: {str(e)}")
                last_exception = e
                
                # 4. EXPONENTIAL BACKOFF
                # Wait 2^attempt seconds (1s, 2s, 4s, 8s...) before trying again
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before next attempt...")
                await asyncio.sleep(wait_time)

        # 2. FAILOVER LOGIC (The Holy Grail)
        if config.fallback_provider:
            logger.error(f"🚨 Primary provider '{config.provider}' exhausted all retries. Initiating FAILOVER to '{config.fallback_provider}'...")
            
            # Create a new config for the fallback provider
            fallback_config = RouterConfig(
                provider=config.fallback_provider,
                model=config.fallback_model or config.model, # Use specific fallback model if provided
                temperature=config.temperature,
                max_retries=config.max_retries
            )
            
            # Recursively call generate with the new config!
            return await self.generate(prompt, fallback_config)
        
        # 5. If we loop through all max_retries and still fail, crash gracefully
        logger.error(f"All {config.max_retries} attempts failed and no fallback configured.")
        raise last_exception