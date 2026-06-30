from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import json
from anthropic import Anthropic

load_dotenv(find_dotenv())

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools=[
    {
        "type":"function",
        "name":"get_campaign_stats",
        "description":"Get the performance metrics for a Campaign",
        "parameters":{
            "type":"object",
            "properties":{
                "campaign":{
                    "type":"string",
                    "description":"A Campaign name like C001 or C002",
                },
            },
            "required":["campaign"],
        },
    },
]

def get_campaign_stats(campaign):
    return f"{campaign}:\"open_rate\": 0.32, \"ctr\": 0.04"

messages=[
    {'role': 'system', 'content': 'You are an expert AI email marketing assistant'}
]

def get_openai_completion(messages, model="gpt-4o-mini"):
    try:
        response = client.responses.create(
            model=model,
            input=messages,
            tools=tools,
            temperature=0,
            stream=True
        )
        content=""
        function_calls={}
        for event in response:
            if event.type == "response.output_text.delta":
                token = event.delta
                print(token,end="",flush=True)
                content+=token
            elif event.type == "response.output_item.added" and event.item.type == "function_call":
                function_calls[event.item.id] = {
                    "call_id": event.item.call_id,
                    "name": event.item.name,
                    "arguments": ""
                }
            elif event.type == "response.function_call_arguments.delta":
                function_calls[event.item_id]["arguments"] += event.delta
            elif event.type == "response.completed":
                break
        print()

        if function_calls:
            for call in function_calls.values():
                messages.append({
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": call["arguments"]
                })
                args = json.loads(call["arguments"]) if call["arguments"] else {}
                if call["name"] == "get_campaign_stats":
                    result = get_campaign_stats(**args)
                else:
                    result = f"Unknown tool: {call['name']}"
                messages.append({
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": result
                })
            return get_openai_completion(messages, model)

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
