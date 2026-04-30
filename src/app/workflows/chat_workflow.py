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
    def __init__(self, query_analyzer: QueryAnalyzerService):
        self._graph = self._build_graph()
        self.query_analyzer = query_analyzer

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

    def _clarify(self, state: ChatWorkflowState) -> ChatWorkflowState:
        return state

    def _retrieve_context(self, state: ChatWorkflowState) -> ChatWorkflowState:
        return state

    async def _generate_answer(
        self,
        state: ChatWorkflowState,
    ) -> ChatWorkflowState:
        return state
