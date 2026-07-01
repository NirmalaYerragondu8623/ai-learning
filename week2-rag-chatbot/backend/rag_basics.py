from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

result=collection.query(query_texts=["What affects open rates?"],n_results=4)
print(result)

# Chunking a file into smaller pieces of text
def chunk_text(file_path, chunk_size=300):
    with open(file_path,'r') as f:
        text=f.read()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n","\n",". "," ",""]
    )
    chunks=splitter.split_text(text)

    #clear whitespaces from each chunk
    chunks=[chunk.replace("\n\n"," ").replace("\n"," ").strip() for chunk in chunks if chunk.strip()]
    return chunks

print("\nTotal chunks:",end="")
chunks=chunk_text("knowledge.txt")
print(f"{len(chunks)}")

for i in range(3): 
    print(f"\n--- Chunk {i+1} ---\n{chunks[i]}\n\n")




