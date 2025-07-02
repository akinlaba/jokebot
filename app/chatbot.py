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
        top_k = top_k or self.top_k
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Filter corpus: high/mid tier + correct label
        intent = self.classify_input(query)
        filtered = [
            (i, item) for i, item in enumerate(self.corpus)
            if item.get("qwen_cleaned_label") == intent
            and item.get("quality_tier") in {"high", "mid"}
        ]

        if not filtered:
            return ["Sorry, no content available."]

        indices, subcorpus = zip(*filtered)
        sub_embeddings = [self.embeddings[i] for i in indices]

        hits = util.semantic_search(query_embedding, sub_embeddings, top_k=100)[0]

        top_entries = [subcorpus[hit["corpus_id"]] for hit in hits]
        top_entries = [entry for entry in top_entries if self.is_viable_joke(entry.get("content", ""))]

        if not top_entries:
            return ["Sorry, no viable jokes found."]

        top_weights = [self.compute_weight(entry) for entry in top_entries]
        sampled = self.weighted_sample(top_entries, top_weights, k=top_k)

        return [entry["content"] for entry in sampled if "content" in entry]

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
    
    @staticmethod
    def is_viable_joke(text):
        text = str(text).lower().strip()
        if len(text) < 100 or len(text) > 500:
            return False
        soft_flags = ["follow me", "@", "whatsapp", "like and share"]
        hard_flags = ["rape", "masturbation", "bastard", "lick a girl", "death", "sperm", "sperms", "sp*rms", 
                    "witchcraft", "girlfriend bobby", "nepa", "kill you", "comment thief"]
        if any(hf in text for hf in hard_flags):
            return False
        if any(sf in text for sf in soft_flags):
            return len(text) > 150
        return True

    @staticmethod
    def compute_weight(entry, alpha=1.0, beta=0.5, gamma=2.0):
        base = alpha * entry.get("likes", 0) + beta * entry.get("shares", 0)
        tier = entry.get("quality_tier", "mid")
        tier_multiplier = {
            "high": 2.0,
            "mid": 1.0,
            "low": 0.3
        }.get(tier, 1.0)
        zscore = entry.get("zscore", 0)
        return base * tier_multiplier + gamma * zscore

    @staticmethod
    def weighted_sample(entries, weights, k=10):
        import numpy as np
        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()
        indices = np.random.choice(len(entries), size=min(k, len(entries)), replace=False, p=probs)
        return [entries[i] for i in indices]

    @staticmethod
    def classify_input(text):
        joke_keywords = [
            "joke", "laugh", "funny", "make me laugh", "tell me something funny",
            "crack me up", "give me joke", "tell me a joke", "i wan laugh"
        ]
        text = text.lower()
        return "joke" if any(kw in text for kw in joke_keywords) else "banter"