
from sentence_transformers import CrossEncoder

from rag_eval_project.retriever import load_store


CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_store()
        self.reranker = CrossEncoder(CROSS_ENCODER, device="cpu")

        self.fetch_k = fetch_k
        self.top_k = top_k

    def invoke(self, query):
        # 1. Retrieve more candidates than we ultimately need.
        candidates = self.store.similarity_search(
            query,
            k=self.fetch_k,
        )

        if not candidates:
            return []

        # 2. Score each query-document pair using the cross-encoder.
        pairs = [
            (query, doc.page_content)
            for doc in candidates
        ]

        scores = self.reranker.predict(pairs)

        # 3. Rank by score, highest first.
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # 4. Return only the best-ranked documents.
        return [
            doc for doc, score in ranked[:self.top_k]
        ]
