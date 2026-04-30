from app.services.llm import llm_service_factory, LLMService
from app.core.settings import settings
from app.services.retriever import RetrieverService
from app.services.query_analyzer import QueryAnalyzerService
from app.workflows.chat_workflow import ChatWorkflow
from typing import Annotated
from fastapi import Depends
from app.services.chat import ChatService


def get_query_analyzer_service():
    return QueryAnalyzerService()


def get_retriever_service():
    return RetrieverService(settings.data_dir)


def get_llm_service():
    return llm_service_factory(settings.llm_provider)


def get_chat_workflow(
    query_analyzer: Annotated[
        QueryAnalyzerService, Depends(get_query_analyzer_service)
    ],
    retriever: Annotated[RetrieverService, Depends(get_retriever_service)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
):
    return ChatWorkflow(
        query_analyzer_service=query_analyzer,
        retriever_service=retriever,
        llm_service=llm_service,
    )


def get_chat_service(workflow: Annotated[ChatWorkflow, Depends(get_chat_workflow)]):
    return ChatService(workflow=workflow)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
