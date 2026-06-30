import os
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv())

client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = "You are an expert AI email marketing assistant with deep knowledge of Marketo and Eloqua."
async def get_reply(message:str,history:list) ->str:
    messages=[{'role':'system','content':SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({'role':'user','content':message})

    response=await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    import asyncio

    async def test():
        reply=await get_reply("What is a good subject line for a Diwali sale email?",[])
        print(reply)
    asyncio.run(test())
