import streamlit as st
from chatbot import JokeBot
import json

# Load corpus and bot
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = [item["quoted_post"] for item in json.load(f)]

bot = JokeBot(corpus)

# UI config
st.set_page_config(page_title="JokeBot", layout="centered")
st.title("JokeBot")
st.write("Type a message and get a joke or witty comeback!")

query = st.text_input("You:", placeholder="Ask me something...")

if query:
    response = bot.get_response(query)
    st.markdown(f"**JokeBot:** {response}")