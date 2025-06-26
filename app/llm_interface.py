import requests
from app.config import config

qwen_api_url = config["qwen_api_url"]
temperature = config["temperature"]
max_tokens = config["max_tokens"]

def qwen_generate(prompt):
    try:
        res = requests.post(qwen_api_url, json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["User:", "Bot:"]
        })
        return res.json()["choices"][0]["text"].strip()
    except Exception as e:
        return f"[LLM ERROR] {e}"