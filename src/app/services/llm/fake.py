from app.services.llm import LLMService


class FakeLLMService(LLMService):
    async def generate(self, messages: list[dict[str, str]]) -> str:
        return "Тестовый ответ."
