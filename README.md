# jokebot

**JokeBot** is a multilingual Nigerian-style chatbot designed to deliver humorous, culturally relevant responses using a retrieval-augmented generation (RAG) architecture. It uses semantic search to retrieve relevant joke content and then prompts a local LLM (Qwen) to generate a response. The project includes both a FastAPI backend and a Streamlit-based UI for interactive usage.

## Features
- Semantic search using SentenceTransformers to find relevant jokes
- Prompt generation and response using a locally hosted LLM (Qwen)
- FastAPI endpoint for programmatic interaction
- Streamlit UI for chatting and collecting feedback
- SQLite-based logging of all chat interactions, including:
  - User input
  - Generated response
  - Session ID and timestamp

## Directory Structure
```
.
├── README.md
├── app
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── api.cpython-312.pyc
│   │   ├── chat_logger.cpython-312.pyc
│   │   ├── chatbot.cpython-312.pyc
│   │   ├── config.cpython-312.pyc
│   │   └── llm_interface.cpython-312.pyc
│   ├── api.py
│   ├── chat_logger.py
│   ├── chatbot.py
│   ├── config.py
│   ├── llm_interface.py
│   ├── main.py
│   └── streamlit_app.py
├── data
│   ├── chat_logs.db
│   ├── config.json
│   ├── corpus.json
│   └── embeddings.pt
├── poetry.lock
├── pyproject.toml
└── tests
```

## Setup

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/)
- `torch`, `sentence-transformers`, `fastapi`, `uvicorn`, `streamlit`, `requests`
- Make sure your config.json is properly configured with:
```
{
  "llm_path": "app",
  "corpus_path": "data/corpus.json",
  "embedding_path": "data/embeddings.pt",
  "qwen_api_url": "http://localhost:5000/v1/completions",
  "temperature": n,
  "max_tokens": n,
  "top_k": n
}
```
### Installation
```
poetry run pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

```
poetry install
```

## Running The App
### FastAPI
```
poetry run uvicorn app.api:app --reload
```
curl your request

### Streamlit
```
poetry run streamlit run app/streamlit_app.py
```
This will launch the web-based chatbot interface.

## Logging
All chat interactions are stored in data/chat_logs.db. Each entry includes:
- Timestamp
- Session ID
- User input
- LLM response
- Top 10 retrieved jokes used as context
You can inspect the logs using the SQLite CLI or a tool like DB Browser for SQLite.