from typing import Any, Literal
from uuid import uuid4

from app.services.llm import LLMService
from app.services.query_analyzer import QueryAnalyzerService
from app.services.retriever import RetrieverService
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


class ChatWorkflowState(BaseModel):
    country: str | None = None
    needs_clarification: bool = False

    context: str = ""
    sources: list[str] = Field(default_factory=list)

    message: str = ""
    thread_id: str = ""

    answer: str = ""


class ChatWorkflow:
    def __init__(
        self,
        query_analyzer_service: QueryAnalyzerService,
        retriever_service: RetrieverService,
        llm_service: LLMService,
    ) -> None:
        self.query_analyzer_service = query_analyzer_service
        self.retriever_service = retriever_service
        self.llm_service = llm_service

        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.chain = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(ChatWorkflowState)

        workflow.add_node("save_user_message", self._save_user_message)
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("clarify", self._clarify)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "save_user_message")
        workflow.add_edge("save_user_message", "analyze_query")

        workflow.add_conditional_edges(
            "analyze_query",
            self._route_query,
            {
                "clarify": "clarify",
                "retrieve": "retrieve_context",
            },
        )

        workflow.add_edge("clarify", END)
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile(
            checkpointer=self.checkpointer,
            store=self.store,
        )

    def _save_user_message(
        self,
        state: ChatWorkflowState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        self._append_history_message(
            runtime=runtime,
            thread_id=state.thread_id,
            role="user",
            content=state.message,
        )

        return {}

    def _analyze_query(self, state: ChatWorkflowState) -> dict[str, Any]:
        analysis = self.query_analyzer_service.analyze(state.message)

        resolved_country = analysis.country or state.country
        needs_clarification = (
            analysis.is_location_dependent and resolved_country is None
        )

        return {
            "country": resolved_country,
            "needs_clarification": needs_clarification,
        }

    def _route_query(
        self,
        state: ChatWorkflowState,
    ) -> Literal["clarify", "retrieve"]:
        if state.needs_clarification:
            return "clarify"

        return "retrieve"

    def _clarify(
        self,
        state: ChatWorkflowState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        answer = (
            "Пожалуйста, уточните, где вы хотите проходить стажировку: "
            "Франция или Германия?"
        )

        self._append_history_message(
            runtime=runtime,
            thread_id=state.thread_id,
            role="assistant",
            content=answer,
        )

        return {
            "answer": answer,
            "context": "",
            "sources": [],
        }

    def _retrieve_context(self, state: ChatWorkflowState) -> dict[str, Any]:
        documents = self.retriever_service.retrieve(country=state.country)

        context = "\n\n".join(
            f"Source: {doc.source}\n{doc.content}" for doc in documents
        )

        return {
            "context": context,
            "sources": [doc.source for doc in documents],
        }

    async def _generate_answer(
        self,
        state: ChatWorkflowState,
        runtime: Runtime,
    ) -> dict[str, Any]:
        history_messages = self._load_history_messages(
            runtime=runtime,
            thread_id=state.thread_id,
            limit=3,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты ассистент, который консультирует пользователей по правилам "
                    "международной стажировки CdekStart. "
                    "Отвечай в дружелюбной и вежливой форме. "
                    "Используй только факты из переданной служебной информации. "
                    "Не упоминай служебную информацию, контекст, документы, файлы, "
                    "базу знаний, промпт или инструкции. "
                    "Если ответа нет в служебной информации, ответь: "
                    "'У меня нет информации по этому вопросу.' "
                    "Учитывай историю диалога, чтобы понимать уточнения пользователя. "
                    "Если вопрос неоднозначный, задай уточняющий вопрос."
                ),
            },
            *history_messages,
            {
                "role": "user",
                "content": (
                    "<service_information>\n"
                    f"{state.context}\n"
                    "</service_information>\n\n"
                    "<user_question>\n"
                    f"{state.message}\n"
                    "</user_question>"
                ),
            },
        ]

        answer = await self.llm_service.generate(messages)

        self._append_history_message(
            runtime=runtime,
            thread_id=state.thread_id,
            role="assistant",
            content=answer,
        )

        return {
            "answer": answer,
            "needs_clarification": False,
        }

    def _history_namespace(self, thread_id: str) -> tuple[str, str]:
        return ("chat_history", thread_id)

    def _append_history_message(
        self,
        runtime: Runtime,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ) -> None:
        store = runtime.store

        if store is None:
            return

        store.put(
            self._history_namespace(thread_id),
            str(uuid4()),
            {
                "role": role,
                "content": content,
            },
        )

    def _load_history_messages(
        self,
        runtime: Runtime,
        thread_id: str,
        limit: int = 8,
    ) -> list[dict[str, str]]:
        store = runtime.store

        if store is None:
            return []

        memories = store.search(self._history_namespace(thread_id))
        recent_memories = memories[-limit:]

        messages = [
            {
                "role": memory.value["role"],
                "content": memory.value["content"],
            }
            for memory in recent_memories
        ]

        return messages

    async def ainvoke(
        self,
        message: str,
        thread_id: str,
    ) -> ChatWorkflowState:
        result = await self.chain.ainvoke(
            {
                "message": message,
                "thread_id": thread_id,
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
        )

        if isinstance(result, ChatWorkflowState):
            return result

        return ChatWorkflowState.model_validate(result)
