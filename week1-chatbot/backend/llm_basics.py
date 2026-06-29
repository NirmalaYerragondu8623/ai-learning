from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from anthropic import Anthropic

load_dotenv(find_dotenv())

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_completion(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are an expert AI email marketing assistant'}, 
                {'role':'user','content':prompt}],
            temperature=0,
            stream=True
        )
        content=""
        for chunk in response:
            delta=chunk.choices[0].delta
            token=delta.content
            if token:
                print(token,end="",flush=True)
                content+=token
            if chunk.choices[0].finish_reason is not None:
                break
        print()
        return content
    except Exception as e:
        error_msg = f"OpenAI API Error: {e}"
        print(error_msg)
        return error_msg


client2= Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_anthropic_completion(prompt, model="claude-3-5-sonnet-20240620"):
    try:
        content = ""
        with client2.messages.stream(
            model=model,
            max_tokens=1000,
            system="You are an expert AI email marketing assistant.",
            messages=[{'role':'user','content':prompt}]
        ) as stream:
            for text in stream.text_stream:
                print(text,end="",flush=True)
                content += text
        print()
        return content
    except Exception as e:
        error_msg = f"Anthropic API Error: {e}"
        print(error_msg)
        return error_msg

user_msg = input()
print("\nOPENAI RESPONSE:\n")
get_openai_completion(user_msg)
print("\nCLAUDE RESPONSE:\n")
get_anthropic_completion(user_msg)
