from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import RagRequest, RagResponse
from rag import rag_answer

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
async def health():
    return {"status":"OK","model":"gpt-4o-mini"}

@app.post("/query", response_model=RagResponse)
async def rag_query(req:RagRequest):
    answer,sources=await rag_answer(req.question)
    return RagResponse(answer=answer,sources=sources)

