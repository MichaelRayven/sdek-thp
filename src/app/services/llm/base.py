from app.services.llm import FakeLLMService, OpenAICompatibleLLMService
from app.core.settings import settings
from abc import abstractmethod, ABC


class LLMService(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


def llm_service_factory() -> LLMService:
    if settings.llm_provider == "fake":
        return FakeLLMService()

    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLMService(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
