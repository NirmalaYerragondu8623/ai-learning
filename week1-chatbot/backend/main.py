from fastapi import FastAPI
from models import ChatRequest, ChatResponse
from llm import get_reply

app=FastAPI()

@app.get("/health")
async def root():
    return {"status":"OK","model":"gpt-4o-mini"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply, history = await get_reply(req.message, req.history)
    return ChatResponse(reply=reply, history=history)
