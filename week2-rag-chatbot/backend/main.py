import asyncio
import hashlib
import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import RagRequest, RagResponse
from rag import rag_answer, stream_rag_answer, vectorstore, add_documents_with_retry, _is_rate_limited

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests. Please slow down and try again shortly."})

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
CHUNK_READ_SIZE = 1024 * 1024

@app.get("/health")
async def health():
    return {"status":"OK","model":"gpt-4o-mini"}

@app.post("/query", response_model=RagResponse)
@limiter.limit("5/minute")
async def rag_query(request: Request, req: RagRequest):
    answer,sources=await rag_answer(req.question)
    return RagResponse(answer=answer,sources=sources)

@app.post("/query/stream")
@limiter.limit("5/minute")
async def rag_query_stream(request: Request, req: RagRequest):
    return StreamingResponse(
        stream_rag_answer(req.question),
        media_type="text/event-stream"
    )

async def _read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    # Read in bounded chunks rather than file.read() so an oversized upload
    # is rejected without buffering the whole thing into memory first.
    data = bytearray()
    while True:
        chunk = await file.read(CHUNK_READ_SIZE)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {max_bytes // (1024 * 1024)}MB limit"
            )
    return bytes(data)

def _process_and_index(tmp_path: str, filename: str) -> int:
    try:
        loader = TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()
    except (UnicodeDecodeError, RuntimeError) as exc:
        # TextLoader wraps decode failures in a RuntimeError with the original
        # UnicodeDecodeError as __cause__ instead of letting it propagate directly.
        if isinstance(exc, UnicodeDecodeError) or isinstance(exc.__cause__, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="File is not valid UTF-8 text")
        raise

    # TextLoader stamps metadata["source"] with the temp file path; overwrite
    # it with the real filename so retrieved chunks can be traced back to it.
    for doc in documents:
        doc.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Deterministic IDs (filename + position + content) so re-uploading the
    # same file overwrites its existing chunks instead of duplicating them.
    ids = [
        hashlib.sha256(f"{filename}:{i}:{chunk.page_content}".encode("utf-8")).hexdigest()
        for i, chunk in enumerate(chunks)
    ]

    add_documents_with_retry(vectorstore, chunks, ids=ids)
    return len(chunks)

@app.post("/ingest")
@limiter.limit("5/minute")
async def ingest_file(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    content = await _read_upload_within_limit(file, MAX_FILE_SIZE_BYTES)
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        try:
            chunks_added = await loop.run_in_executor(None, _process_and_index, tmp_path, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            if _is_rate_limited(exc):
                raise HTTPException(
                    status_code=503,
                    detail="Embedding service is rate-limited after retries; please try again shortly"
                ) from exc
            raise

        return {
            "filename": file.filename,
            "chunks_added": chunks_added,
            "message": f"Successfully indexed {chunks_added} chunks"
        }
    finally:
        os.unlink(tmp_path)  # Clean up temp file
