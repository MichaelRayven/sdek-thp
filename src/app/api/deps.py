from typing import Annotated
from fastapi import Depends
from app.services.chat import ChatService


def get_chat_service():
    return ChatService()


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
