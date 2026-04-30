from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.settings import settings
from app.services.chat import ChatService
from app.services.llm import llm_service_factory, LLMService
from app.services.query_analyzer import QueryAnalyzerService
from app.services.retriever import RetrieverService
from app.workflows.chat_workflow import ChatWorkflow


def get_query_analyzer_service() -> QueryAnalyzerService:
    return QueryAnalyzerService()


def get_retriever_service() -> RetrieverService:
    return RetrieverService(settings.data_dir)


def get_llm_service() -> LLMService:
    return llm_service_factory(settings.llm_provider)


def get_chat_workflow() -> ChatWorkflow:
    return ChatWorkflow(
        query_analyzer_service=get_query_analyzer_service(),
        retriever_service=get_retriever_service(),
        llm_service=get_llm_service(),
    )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(workflow=get_chat_workflow())


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
