from embeddings.embedder import Embedder
from vectorstore.sqlite_store import SQLiteStore


class SQLiteRetriever:
    def __init__(self, k=2):
        self.k = k
        self.embedder = Embedder()
        self.store = SQLiteStore()

    def retrieve(self, query: str):
        query_embedding = self.embedder.embed(query)

        return self.store.search(
            query_embedding=query_embedding.tolist(),
            query_text=query,
            k=self.k,
        )

    def close(self):
        self.store.close()