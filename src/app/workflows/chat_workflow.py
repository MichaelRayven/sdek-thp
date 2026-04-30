from app.services.llm import LLMService
from app.services.retriever import RetrieverService
from app.services.query_analyzer import QueryAnalyzerService
from pydantic import BaseModel

from langgraph.graph import START, END, StateGraph


class ChatWorkflowState(BaseModel):
    country: str | None = None
    needs_clarification: bool

    context: str
    message: str
    tread_id: str

    answer: str


class ChatWorkflow:
    def __init__(
        self,
        query_analyzer: QueryAnalyzerService,
        retriever: RetrieverService,
        llm: LLMService,
    ):
        self._graph = self._build_graph()
        self.query_analyzer = query_analyzer
        self.retriever = retriever
        self.llm = llm

    def _build_graph(self):
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

    def _analyze_query(self, state: ChatWorkflowState) -> dict:
        analysis = self.query_analyzer.analyze(state.message)

        return {
            "country": analysis.country,
            "needs_clarification": analysis.needs_clarification,
        }

    def _route_query(self, state: ChatWorkflowState) -> str:
        if state.needs_clarification:
            return "clarify"

        return "retrieve"

    def _clarify(self, state: ChatWorkflowState) -> dict:
        return {"answer": "Пожалуйста, уточните страну: Франция или Германия"}

    def _retrieve_context(self, state: ChatWorkflowState) -> dict:
        documents = self.retriever.retrieve(country=state.country)

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
                    "Ты RAG ассистент, который консультирует пользователей по правилам международной стажировки. "
                    "Формируй ответы основываясь на предоставленном контексте. "
                    "Если контекст не содержит ответа на заданный вопрос, отвечай "
                    "что база знаний не содержит такой информации. "
                    "Не искажай факты."
                ),
            },
            {
                "role": "user",
                "content": (f"Контекст:\n{state.context}\n\Вопрос:\n{state.message}"),
            },
        ]

        return {"answer": await self.llm.generate(messages)}
