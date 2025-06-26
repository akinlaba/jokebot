import streamlit as st
from chatbot import JokeBot
import json
from datetime import datetime
import sys
from pathlib import Path
from llm_interface import qwen_generate
from config import config

llm_path = Path(config["llm_path"])
corpus_path = Path(config["corpus_path"])

sys.path.insert(0, llm_path)

with open(corpus_path, "r") as f:
    corpus = json.load(f)

bot = JokeBot(corpus, generator=qwen_generate)

st.set_page_config(page_title="JokeBot 🤖", layout="centered")
st.title("😂 JokeBot")

# Init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Show chat history
for i, (role, message) in enumerate(st.session_state.chat_history):
    with st.chat_message(role):
        st.markdown(message)

        if role == "bot":
            cols = st.columns(2)
            if cols[0].button("Thumbs Up", key=f"up_{i}"):
                st.session_state.feedback.append(("Thumbs Up", message, str(datetime.now())))
                st.success("Thanks for the feedback!")

            if cols[1].button("Thumbs Down", key=f"down_{i}"):
                st.session_state.feedback.append(("Thumbs Down", message, str(datetime.now())))
                st.warning("We'll try to do better!")

# Input: safe method using return value
user_input = st.chat_input("Type your message...")

if user_input:
    bot_response = bot.get_response(user_input)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", bot_response))
    st.rerun()  # refresh UI to reflect new chat bubble

# Clear chat button
if st.button("Clear chat"):
    st.session_state.chat_history = []
    st.session_state.feedback = []
    st.rerun()