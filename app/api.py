from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot import JokeBot
import json
import sys
from pathlib import Path
from app.config import config
from app.llm_interface import qwen_generate
from app.chat_logger import log_chat
from uuid import uuid4

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
    session_id: str | None = None  # if we want session_id from the client instead

@app.post("/chat")
def chat(input: ChatInput):
    response, retrieved = bot.get_response(input.query)
    
    session_id = getattr(input, "session_id", None) or str(uuid4())

    log_chat(user_input=input.query, 
             bot_response=response, 
             session_id=session_id, 
             retrieved_jokes=retrieved)
    
    return {"response": response}

@app.get("/")
def root():
    return {"message": "JokeBot is alive!"}

