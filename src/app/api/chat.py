from app.api.deps import ChatServiceDep
from app.models.chat import ChatResponse, ChatRequest
from fastapi import APIRouter


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    return await service.reply(request)
