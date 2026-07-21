'''Move your LangChain chain logic here. Wrap in async def rag_answer(question: str) -> tuple[str, list[str]]. Return the answer string and a list of source chunk texts. Load the 
vectorstore once at module level — not inside the function, or it re-loads on every request.'''


import os
from typing import AsyncGenerator
from dotenv import load_dotenv,find_dotenv
from openai import RateLimitError
from tenacity import retry, retry_if_exception, wait_random_exponential, stop_after_attempt
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

def _is_rate_limited(exc: Exception) -> bool:
    return isinstance(exc, RateLimitError) or "429" in str(exc) or "rate_limit" in str(exc).lower()

# Shared by startup indexing and the /ingest endpoint so both survive transient
# 429s/timeouts from the embeddings API instead of failing (or partially
# indexing) on the first hiccup.
@retry(
    retry=retry_if_exception(_is_rate_limited),
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
)
def add_documents_with_retry(store: Chroma, docs, ids=None):
    return store.add_documents(docs, ids=ids) if ids is not None else store.add_documents(docs)

# --- Load vectorstore ONCE at module level -------------------------
# This runs when FastAPI import rag.py - not on every request
# If lc_chroma_db doesn't exist yet, it builds and persists it
PERSIST_DIR=os.getenv("PERSIST_DIR", "./lc_chroma_db")
KNOWLEDGE_DIR = "./knowledge_base"
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    #Already indexed - just load from disk
    vectorstore=Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
else:
    # First run - build and persist
    # Load all .txt files from the folder automatically
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n\n","\n",". "," ",""]
    )
    chunks=splitter.split_documents(documents)

    EMBED_BATCH_SIZE = 80
    vectorstore=Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        add_documents_with_retry(vectorstore, batch)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# ---- Prompt -------------------------------------------------------
prompt=ChatPromptTemplate.from_template("""
Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:{context}
Question:{question}
""")

llm=ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

RATE_LIMIT_MESSAGE = "I'm getting rate-limited by the AI provider right now. Please try again in a moment."

# ---- LCEL chain ----------------------------------------------------
chain=(
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

# --- Main function called by FastAPI ---------------------------------------
async def rag_answer(question:str)->tuple[str,list[str]]:
    try:
        #Run the chain - LangChain's invoke is sync, wrap with asyncio
        import asyncio
        loop=asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, chain.invoke,question)

        #Get source chunks separately for citation
        source_docs=retriever.invoke(question)
        sources=[doc.page_content.replace("\n\n"," ").replace("\n"," ").strip() for doc in source_docs]
        sources=list(dict.fromkeys(sources))

        return answer,sources
    except Exception as exc:
        if _is_rate_limited(exc):
            return RATE_LIMIT_MESSAGE, []
        raise

async def stream_rag_answer(question: str) -> AsyncGenerator[str, None]:
    try:
        # Step 1: Retrieve relevant chunks (sync, fast)
        source_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)
    except Exception as exc:
        if _is_rate_limited(exc):
            yield RATE_LIMIT_MESSAGE
            return
        raise

    # Step 2: Build the grounded prompt
    full_prompt = f"""Answer using ONLY the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}"""

    # Step 3: Stream the generation using the chat model directly
    try:
        async for chunk in llm.astream(full_prompt):
            if chunk.content:
                yield chunk.content
    except Exception as exc:
        if _is_rate_limited(exc):
            yield RATE_LIMIT_MESSAGE
            return
        raise

