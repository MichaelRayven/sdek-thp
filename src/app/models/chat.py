from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = Field(default="default")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    needs_clarification: bool = False
