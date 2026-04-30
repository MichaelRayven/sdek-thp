from app.models.chat import ChatRequest, ChatResponse


class ChatService:
    async def reply(self, request: ChatRequest) -> ChatResponse:
        message = request.message.strip()

        return ChatResponse(
            answer=("TODO: Connect LLM, get context from session, message:" + message),
            sources=[],
            needs_clarification=False,
        )
