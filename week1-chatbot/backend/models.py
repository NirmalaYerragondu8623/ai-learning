from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    history: list[dict]=[]

class ChatResponse(BaseModel):
    reply: str
