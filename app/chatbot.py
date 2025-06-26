from sentence_transformers import SentenceTransformer, util
import random
import torch
from pathlib import Path
from app.config import config

class JokeBot:
    def __init__(self, corpus, generator=None):
        self.corpus = corpus
        self.embedding_path = Path(config["embedding_path"])
        self.top_k = config["top_k"]
        self.generator = generator
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        if self.embedding_path.exists():
            self.embeddings = torch.load(self.embedding_path)
        else:
            embedding_texts = [item["quoted_post"] for item in corpus]
            self.embeddings = self.model.encode(embedding_texts, convert_to_tensor=True)
            self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.embeddings, self.embedding_path)

    def retrieve_responses(self, query, top_k=None):
        top_k = self.top_k
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]

        all_contents = []
        for hit in hits:
            entry = self.corpus[hit["corpus_id"]]
            contents = entry.get("content", [])
            if isinstance(contents, str):
                all_contents.append(contents)
            elif isinstance(contents, list):
                all_contents.extend(contents)
        return all_contents[:10]

    def get_response(self, query, top_k=None):
        top_k = self.top_k
        retrieved = self.retrieve_responses(query, top_k=top_k)
        response = self.generate_response(query, retrieved)
        return response, retrieved

    def generate_response(self, query, retrieved):
        if not self.generator:
            return random.choice(retrieved)

        # retrieved_texts = [r["quoted_post"] for r in retrieved]
        context = "\n".join(f"[{i+1}] {r}" for i, r in enumerate(retrieved))

        prompt = f"""User: I'm looking for a funny Nigerian-style joke. Here are some options:
            {context}

            Assistant: Pick the funniest one and rewrite it as a joke:"""

        response = self.generator(prompt)
        return response