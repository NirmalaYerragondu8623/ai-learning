from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
    )

@app.get("/hello")
def say_hello():
    return f" Welcome to the FastAPI learning journey!!"

@app.get("/hello/{name}")
def say_hello(name:str):
    return f"Hi {name}!! Welcome to the FastAPI learning journey!!"

class MessageRequest(BaseModel):
    message:str
    history:list[dict]=[]

@app.post("/chat")
def chat(req:MessageRequest):
    return {
        "reply":f"You said:{req.message}",
        "history_length":len(req.history)
    }

@app.post("/echo")
def echo(req:MessageRequest):
    return {"reply":{req.message}}




