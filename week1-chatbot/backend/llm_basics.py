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
        "function":{
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
    },
]

def get_campaign_stats(campaign):
    return f"{campaign}:\"open_rate\": 0.32, \"ctr\": 0.04"

messages=[
    {'role': 'system', 'content': 'You are an expert AI email marketing assistant'}
]

def get_openai_completion(messages, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0,
            stream=True
        )
        content=""
        tool_calls={}
        for chunk in response:
            delta=chunk.choices[0].delta
            if delta.content:
                print(delta.content,end="",flush=True)
                content+=delta.content
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index not in tool_calls:
                        tool_calls[tc.index]={"id":tc.id,"name":"","arguments":""}
                    if tc.id:
                        tool_calls[tc.index]["id"]=tc.id
                    if tc.function.name:
                        tool_calls[tc.index]["name"]+=tc.function.name
                    if tc.function.arguments:
                        tool_calls[tc.index]["arguments"]+=tc.function.arguments
            if chunk.choices[0].finish_reason is not None:
                break
        print()

        if tool_calls:
            messages.append({
                "role":"assistant",
                "content":content or None,
                "tool_calls":[
                    {
                        "id":call["id"],
                        "type":"function",
                        "function":{"name":call["name"],"arguments":call["arguments"]}
                    } for call in tool_calls.values()
                ]
            })
            for call in tool_calls.values():
                args = json.loads(call["arguments"]) if call["arguments"] else {}
                if call["name"] == "get_campaign_stats":
                    result = get_campaign_stats(**args)
                else:
                    result = f"Unknown tool: {call['name']}"
                messages.append({
                    "role":"tool",
                    "tool_call_id":call["id"],
                    "content":result
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
