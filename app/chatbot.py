from sentence_transformers import SentenceTransformer, util
import random
import numpy as np
import torch
from pathlib import Path
from app.config import config
import re
import unicodedata

class JokeBot:
    def __init__(self, corpus, generator=None):
        self.corpus = corpus
        self.embedding_path = Path(config["embedding_path"])
        self.top_k = config["top_k"]
        self.generator = generator
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.image_flags = config["image_flags"]
        self.soft_flags = config["soft_flags"]
        self.hard_flags = config["hard_flags"]
        self.joke_keywords = config["joke_keywords"]
        self.hallucinated_phrases = config["hallucinated_phrases"]
        self.prompts = config["prompts"]

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

        intent = self.classify_input(query)
        random.shuffle(retrieved)
        fallback_response = retrieved[0] if retrieved else "Sorry, nothing funny found."
        template = self.prompts.get(intent, self.prompts["banter"])

        for i, joke in enumerate(retrieved[:5]):
            joke = joke.replace('"', "'")
            prompt = template.format(joke=joke)
            try:
                response = self.generator(prompt)
            except Exception as e:
                print(f"[Qwen ERROR]: {e}")
                continue
            cleaned = self.postprocess_response(response)
            if cleaned:
                return cleaned

        return fallback_response
    
    def is_viable_joke(self, text):
        text = str(text).lower().strip()
        if len(text) < 100 or len(text) > 500:
            return False
        image_flags = self.image_flags
        # Moderate red flags (allowed, but penalized if needed)
        soft_flags = self.soft_flags
        # Hard red flags (never allow)
        hard_flags = self.hard_flags
        if any(kw in text for kw in image_flags):
            return False
        if any(hf in text for hf in hard_flags):
            return False
        if any(sf in text for sf in soft_flags):
            return len(text) > 150  # Allow longer, well-formed jokes with mild spam
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
        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()
        indices = np.random.choice(len(entries), size=min(k, len(entries)), replace=False, p=probs)
        return [entries[i] for i in indices]

    def classify_input(self, text):
        joke_keywords = self.joke_keywords
        text = text.lower()
        return "joke" if any(kw in text for kw in joke_keywords) else "banter"
    
    def postprocess_response(self, text, min_len=30):
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()

        # Remove echoes or instruction-style hallucinations
        hallucinated_phrases = self.hallucinated_phrases
        if any(p in text.lower() for p in hallucinated_phrases):
            return ""

        # Remove Chinese or emoji-only outputs
        if re.search(r'[\u4e00-\u9fff]', text):
            return ""

        cleaned = ''.join(c for c in text if unicodedata.category(c)[0] != "C").strip()
        if len(cleaned.split()) < 4 or len(cleaned) < min_len:
            return ""

        return re.sub(r'\n{2,}', '\n', cleaned.strip(" \"“”’‘"))