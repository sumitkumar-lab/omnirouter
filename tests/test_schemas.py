from src.schemas import LLMResponse, RouterConfig


def test_router_config_defaults():
    router = RouterConfig()

    assert router.provider == "openai"
    assert router.model
    assert router.max_retries == 3


def test_llm_response_accepts_required_fields():
    response = LLMResponse(
        content="hello",
        provider_used="groq",
        model_used="llama-3.1-8b-instant",
    )

    assert response.content == "hello"
    assert response.prompt_tokens == 0
    assert response.cost_estimate == 0.0
