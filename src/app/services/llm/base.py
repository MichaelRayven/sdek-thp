from abc import abstractmethod, ABC


class LLMService(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError
