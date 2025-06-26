from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot import JokeBot
import json


with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = [item["quoted_post"] for item in json.load(f)]

bot = JokeBot(corpus)

app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.post("/chat")
def chat(input: QueryInput):
    response = bot.get_response(input.query)
    return {"response": response}