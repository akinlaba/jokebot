from chatbot import JokeBot
import json

with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = [item["quoted_post"] for item in json.load(f)]

bot = JokeBot(corpus)

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break
    print("Bot:", bot.get_response(user_input))