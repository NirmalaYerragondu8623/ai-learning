'''Move your LangChain chain logic here. Wrap in async def rag_answer(question: str) -> tuple[str, list[str]]. Return the answer string and a list of source chunk texts. Load the 
vectorstore once at module level — not inside the function, or it re-loads on every request.'''


import os
import time
from typing import AsyncGenerator
from dotenv import load_dotenv,find_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

# --- Load vectorstore ONCE at module level -------------------------
# This runs when FastAPI import rag.py - not on every request
# If lc_chroma_db doesn't exist yet, it builds and persists it
PERSIST_DIR="./lc_chroma_db"
KNOWLEDGE_DIR = "./knowledge_base"
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

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
        loader_cls=TextLoader
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n","\n",". "," ",""]
    )
    chunks=splitter.split_documents(documents)

    # Gemini's free tier caps embedding calls at ~100/minute. Feed chunks in
    # small batches with a cooldown so a large first-time index doesn't 429.
    EMBED_BATCH_SIZE = 80
    vectorstore=Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch)
        if i + EMBED_BATCH_SIZE < len(chunks):
            time.sleep(65)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# ---- Prompt -------------------------------------------------------
prompt=ChatPromptTemplate.from_template("""
Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:{context}
Question:{question}
""")

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---- LCEL chain ----------------------------------------------------
chain=(
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

# --- Main function called by FastAPI ---------------------------------------
async def rag_answer(question:str)->tuple[str,list[str]]:
    #Run the chain - LangChain's invoke is sync, wrap with asyncio
    import asyncio
    loop=asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, chain.invoke,question)

    #Get source chunks separately for citation
    source_docs=retriever.invoke(question)
    sources=[doc.page_content.replace("\n\n"," ").replace("\n"," ").strip() for doc in source_docs]
    sources=list(dict.fromkeys(sources))

    return answer,sources

async def stream_rag_answer(question: str) -> AsyncGenerator[str, None]:
    # Step 1: Retrieve relevant chunks (sync, fast)
    source_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_docs)

    # Step 2: Build the grounded prompt
    full_prompt = f"""Answer using ONLY the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}"""

    # Step 3: Stream the generation using the Gemini chat model directly
    async for chunk in llm.astream(full_prompt):
        if chunk.content:
            yield chunk.content

