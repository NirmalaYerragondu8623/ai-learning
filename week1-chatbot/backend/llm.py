import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = "You are an expert AI email marketing assistant with deep knowledge of Marketo and Eloqua."

async def get_reply(message: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[m for m in messages if m["role"] != "system"]
    )
    return response.content[0].text

async def stream_reply(message: str, history: list):
    messages = list(history)
    messages.append({"role": "user", "content": message})

    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages
    ) as stream:
        async for text in stream.text_stream:
            yield text