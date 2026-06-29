from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from anthropic import Anthropic

load_dotenv(find_dotenv())

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

messages=[
    {'role': 'system', 'content': 'You are an expert AI email marketing assistant'}
]

def get_openai_completion(messages, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
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
    

def collect_messages(prompt):
    messages.append({'role':'user','content':f"{prompt}"})
    response=get_openai_completion(messages)
    messages.append({'role':'assistant','content':f"{response}"})
    return response
    

'''client2= Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
        return error_msg '''
while True:
    user_msg = input()
    if user_msg.strip().lower() in ("exit","quit"):
        break
    print("\nAssistant:")
    collect_messages(user_msg)
    print()
    
    #print("\nCLAUDE RESPONSE:\n")
    #get_anthropic_completion(user_msg)
