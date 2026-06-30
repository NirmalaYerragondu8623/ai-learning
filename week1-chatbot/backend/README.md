# AI Email Marketing Assistant — FastAPI Chatbot

A FastAPI backend that wraps OpenAI's Chat Completions API into a
streaming, multi-turn chatbot specialized in email marketing (Marketo/Eloqua).

## Live URL
https://ai-learning-2jdt.onrender.com

## Endpoints
- `GET /health` — service health check
- `POST /chat` — single-turn or multi-turn chat (client-managed history)
- `POST /chat/stream` — streaming response via Server-Sent Events

## Tech stack
FastAPI · OpenAI API · Docker · Render

## Run locally
1. `pip install -r requirements.txt`
2. Add `.env` with `OPENAI_API_KEY`
3. `uvicorn main:app --reload`

## Run with Docker
```
docker build -t chatbot .
docker run -d --name chatbot -p 8000:8000 --env-file .env chatbot
```

## Example
```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is email open rate?","history":[]}'
```
