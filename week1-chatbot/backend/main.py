from fastapi import FastAPI
from models import ChatRequest, ChatResponse
from llm import get_reply,stream_reply
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

#Add CORS middleware right after creating the app instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def root():
    return {"status":"OK","model":"gpt-4o-mini"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply, history = await get_reply(req.message, req.history)
    return ChatResponse(reply=reply, history=history)

@app.post("/chat/stream")
async def chat_stream(req:ChatRequest):
    return StreamingResponse(
        stream_reply(req.message,req.history),
        media_type="text/event-stream"
    )
