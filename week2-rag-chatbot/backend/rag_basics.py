from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

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






