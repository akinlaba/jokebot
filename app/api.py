from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot import JokeBot
import json
import sys
from pathlib import Path
from app.config import config
from app.llm_interface import qwen_generate

llm_path = Path(config["llm_path"])
corpus_path = Path(config["corpus_path"])

sys.path.insert(0, llm_path)

with open(corpus_path, "r") as f:
    corpus = json.load(f)

bot = JokeBot(corpus, generator=qwen_generate)

app = FastAPI()

class QueryInput(BaseModel):
    query: str

class ChatInput(BaseModel):
    query: str

@app.post("/chat")
def chat(input: ChatInput):
    response = bot.get_response(input.query)
    return {"response": response}

@app.get("/")
def root():
    return {"message": "JokeBot is alive!"}