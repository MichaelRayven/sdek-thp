from app.services.llm import FakeLLMService, OpenAICompatibleLLMService, LLMService
from app.core.settings import settings


def llm_service_factory(llm_provider: str) -> LLMService:
    if llm_provider == "fake":
        return FakeLLMService()

    if llm_provider == "openai_compatible":
        return OpenAICompatibleLLMService(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    raise ValueError(f"Unsupported LLM provider: {llm_provider}")
