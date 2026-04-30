from app.workflows.chat_workflow import ChatWorkflow
from app.models.chat import ChatRequest, ChatResponse


class ChatService:
    def __init__(self, workflow: ChatWorkflow):
        self.workflow = workflow

    async def reply(self, request: ChatRequest) -> ChatResponse:
        message = request.message.strip()

        result = await self.workflow.ainvoke(
            message=message,
            thread_id=request.thread_id,
        )

        return ChatResponse(answer=result.answer)
