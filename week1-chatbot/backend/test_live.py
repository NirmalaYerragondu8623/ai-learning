import requests

response = requests.post(
    "https://ai-learning-2jdt.onrender.com/chat/stream",
    json={"message": "What is email open rates?", "history": []},
    stream=True
)

print(f"Status: {response.status_code}")
print("Response:")
for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode("utf-8"), end="", flush=True)