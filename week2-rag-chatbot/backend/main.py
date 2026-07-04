from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models import RagRequest, RagResponse
from rag import rag_answer, stream_rag_answer
from fastapi import UploadFile, File
import tempfile
import os

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
async def health():
    return {"status":"OK","model":"gemini-2.5-flash"}

@app.post("/query", response_model=RagResponse)
async def rag_query(req:RagRequest):
    answer,sources=await rag_answer(req.question)
    return RagResponse(answer=answer,sources=sources)

@app.post("/query/stream")
async def rag_query_stream(req: RagRequest):
    return StreamingResponse(
        stream_rag_answer(req.question),
        media_type="text/event-stream"
    )

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    # Read uploaded file content
    content = await file.read()
    
    # Save to a temp file so LangChain's TextLoader can read it
    with tempfile.NamedTemporaryFile(
        mode='wb', 
        suffix='.txt', 
        delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Reuse your existing chunking + embedding logic from rag.py
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        loader = TextLoader(tmp_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=30,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        # Add to existing vectorstore — import from rag.py
        from rag import vectorstore, embeddings
        vectorstore.add_documents(chunks)
        
        return {
            "filename": file.filename,
            "chunks_added": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks"
        }
    finally:
        os.unlink(tmp_path)  # Clean up temp file

