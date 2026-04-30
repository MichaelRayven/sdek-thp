from app.services.llm import LLMService
from app.services.retriever import RetrieverService
from app.services.query_analyzer import QueryAnalyzerService
from pydantic import BaseModel

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph


class ChatWorkflowState(BaseModel):
    country: str | None = None
    needs_clarification: bool = False

    context: str = ""
    message: str = ""
    thread_id: str = ""

    answer: str = ""


class ChatWorkflow:
    def __init__(
        self,
        query_analyzer_service: QueryAnalyzerService,
        retriever_service: RetrieverService,
        llm_service: LLMService,
    ):
        self.query_analyzer_service = query_analyzer_service
        self.retriever_service = retriever_service
        self.llm_service = llm_service

        self.checkpointer = InMemorySaver()
        self.chain = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(ChatWorkflowState)

        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("clarify", self._clarify)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "analyze_query")

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

        return workflow.compile(checkpointer=self.checkpointer)

    def _analyze_query(self, state: ChatWorkflowState) -> dict:
        analysis = self.query_analyzer_service.analyze(state.message)

        resolved_country = analysis.country or state.country
        needs_clarification = (
            analysis.is_location_dependent and resolved_country is None
        )

        return {
            "country": resolved_country,
            "needs_clarification": needs_clarification,
        }

    def _route_query(self, state: ChatWorkflowState) -> str:
        if state.needs_clarification:
            return "clarify"

        return "retrieve"

    def _clarify(self, state: ChatWorkflowState) -> dict:
        return {
            "answer": (
                "Пожалуйста, уточните, где вы хотите проходить стажировку: "
                "Франция или Германия?"
            ),
            "context": "",
        }

    def _retrieve_context(self, state: ChatWorkflowState) -> dict:
        documents = self.retriever_service.retrieve(country=state.country)

        context = "\n\n".join(
            f"Source: {doc.source}\n{doc.content}" for doc in documents
        )

        return {"context": context}

    async def _generate_answer(
        self,
        state: ChatWorkflowState,
    ) -> dict:
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
                    "Если вопрос неоднозначный, задай уточняющий вопрос."
                ),
            },
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

        return {"answer": await self.llm_service.generate(messages)}

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
