from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
for i,emb in enumerate(embeddings):
    print(f"Sentence {i+1}: {emb}\n")

# Calculate the cosine similiarity between the first and rest of the sentences
similarity_scores=cosine_similarity(embeddings,embeddings)
print(f"Similarity Scores: {similarity_scores}")