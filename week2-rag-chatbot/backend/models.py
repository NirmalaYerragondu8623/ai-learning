from pydantic import BaseModel

class RagRequest(BaseModel):
    question: str

class RagResponse(BaseModel):
    answer: str
    sources: list[str] = []

