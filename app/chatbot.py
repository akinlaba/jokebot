from sentence_transformers import SentenceTransformer, util
import random
import torch
from pathlib import Path

class JokeBot:
    def __init__(self, corpus, embedding_path="data/embeddings.pt"):
        self.corpus = corpus
        self.embedding_path = Path(embedding_path)
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        if self.embedding_path.exists():
            self.embeddings = torch.load(self.embedding_path)
        else:
            self.embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
            self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.embeddings, self.embedding_path)

    def get_response(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]
        return random.choice([self.corpus[hit["corpus_id"]] for hit in hits])