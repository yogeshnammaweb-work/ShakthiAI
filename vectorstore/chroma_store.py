import chromadb


class ChromaStore:
    def __init__(self, path="database/chroma"):
        self.client = chromadb.PersistentClient(path=path)

        self.collection = self.client.get_or_create_collection(
            name="shakthi_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, ids, documents, embeddings, metadatas=None):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding, k=2):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )