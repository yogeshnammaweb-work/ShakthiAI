from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore


class Retriever:
    def __init__(self, k=2):
        self.k = k
        self.embedder = Embedder()
        self.store = ChromaStore()

    def retrieve(self, query: str):
        query_embedding = self.embedder.embed(query)

        return self.store.search(
            query_embedding=query_embedding.tolist(),
            k=self.k,
        )
