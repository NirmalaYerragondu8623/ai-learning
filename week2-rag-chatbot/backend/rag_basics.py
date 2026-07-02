from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv,find_dotenv

# ---- Section 1: Embedding + cosine similarity ----

# Load a local model(no API cost) 
model=SentenceTransformer('all-MiniLM-L6-v2')

# Define sentences
sentences=[
    "Email campaign is used to send Emails to a particular set of audience to promote brand or services.",
    "The campaign success rate is measured by its open rate, Click-through rates.",
    "User Engagement is the main goal of email campaigns."
]

# Generate Embeddings for the sentences
embeddings=model.encode(sentences)

# print embedding vectors
# for i,emb in enumerate(embeddings):
#     print(f"Sentence {i+1}: {emb}\n")

# Calculate the cosine similiarity between the first and rest of the sentences
similarity_scores=cosine_similarity(embeddings,embeddings)
print(f"Similarity Scores: {similarity_scores}")

# ---- Section 2: Chroma store + query ----

# Create a ChromaDB client
client = chromadb.Client()

# Creating a collection. Switch 'create_collection' to 'get_or_create_collection' to avoid creating a new collection everytime.
collection=client.get_or_create_collection(name="email_kb")

# Add the sentence to the collection. Switch 'add' to 'upsert' to avoid adding the same documents everytime
collection.upsert(documents=["Email campaign is used to send Emails to a particular set of audience to promote brand or services.",
                          "The campaign success rate is measured by its open rate, Click-through rates.",
                          "User Engagement is the main goal of email campaigns."],
                ids=["1","2","3"] 
            )
#print(collection.count())

result=collection.query(query_texts=["What is RAG?"],n_results=4)
print(result)

# Chunking a file into smaller pieces of text
def chunk_text(file_path, chunk_size=200):
    with open(file_path,'r') as f:
        text=f.read()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=30,
        separators=["\n\n","\n",". "," ",""]
    )
    chunks=splitter.split_text(text)

    #clear whitespaces from each chunk
    chunks=[chunk.replace("\n\n"," ").replace("\n"," ").strip() for chunk in chunks if chunk.strip()]
    return chunks

# print("\nTotal chunks:",end="")
# chunks=chunk_text("knowledge.txt")
# print(f"{len(chunks)}")

# for i in range(3): 
#     print(f"\n--- Chunk {i+1} ---\n{chunks[i]}\n\n")

# ---- Section 3: Full RAG Pipeline ----
# chunk -> Embed -> store -> retrieve -> generate

def create_or_get_collection(client,name:str):
    try:
        collection=client.get_collection(name=name)
    except:
        collection=client.create_collection(name=name)
    return collection

#Creating chunks from knowledge file
chunks=chunk_text("knowledge.txt")

#Embed the chunks and store them in ChromaDB
embed=model.encode(chunks).tolist()

#creating a persistent client to store embeddings in a local directory
persistent_client=chromadb.PersistentClient(path="./chroma_db")
collection2=create_or_get_collection(persistent_client,"email_marketing_kb")
if collection2.count()==0:
    collection2.add(ids=[f"chunk_{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embed,
                metadatas=[{"source":"knowledge.txt"} for _ in chunks]
                )
    print(f"Indexed {len(chunks)} chunks")
else:
    print(f"Collection already has {collection2.count()} chunks -  skipping indexing")

def retrieve(question:str,k=3)->list[str]:
    #embed the question
    question_embed=model.encode([question]).tolist()
    
    #retrieve relevant chunks
    results=collection2.query(
        query_embeddings=question_embed,
        n_results=k
    )
    return results["documents"][0]


chunks_retrieved=retrieve("What is RAG?")
for i,chunk in enumerate(chunks_retrieved):
    print(f"Chunk {i}:\n{chunk}\n\n")


load_dotenv(find_dotenv())
openai_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def rag_answer(question:str)->str:
    chunks=retrieve(question,k=3)
    context="\n\n".join(chunks)

    prompt=f"""Answer using ONLY the context below. \
    If the answer isn't in the context, say "I don't have that information." \
    Context:{context}\n\nQuestion:{question}"""

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role':'user','content':prompt}]
    )
    return response.choices[0].message.content

# async def test():
#     query=input("Enter your question: ")
#     answer=await rag_answer(query)
#     print("\nAnswer:",answer)

# asyncio.run(test())


#section4: RAG with Langchain
import bs4
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1. Load the text file
loader=TextLoader("knowledge.txt")

# 2. Load documents (returns a list of Document objects)
documents=loader.load()

# 3. Inspect the first document
#print(documents[0].page_content)

# Create a splitter (e.g., 500 characters per chunk)
splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n","\n",". "," ",""]
    )

# Split into chunks
chunks=splitter.split_documents(documents)

# print(f"Number of chunks:{len(chunks)}")
# print(chunks[0].page_content)

# 4.Initiialize embeddings (OpenAI)
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

# 5. Build a Chroma vectorstore with a separate persist directory
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./lc_chroma_db' #separate diecrectory for lanchain Chroma DB
)

print('Knowledge base successfully embedded into Chroma (lc_chroma_db.')

# Build retriever
retriever=vectorstore.as_retriever(search_kwargs={"k":3})

# Build Prompt
prompt = ChatPromptTemplate.from_template(
    """Answer using ONLY the below context.
    If the answer isn't in the context, say "I don't have that information."
    Context:{context}
    Question:{question}
    """
) 

#Build LLM
llm=ChatOpenAI(model="gpt-4o-mini")

# Build LECL chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain=(
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

#Test with same 3 questions
user_query=input("Enter your question:")
answer=chain.invoke(user_query)
print("Answer:",answer)