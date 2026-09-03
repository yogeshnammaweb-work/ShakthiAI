import json
from pathlib import Path

from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore


DATA = Path(r"data\master_dataset\StudentModules_chunks.jsonl")


def main():
    with open(DATA, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    print("Chunks to index:", len(rows))

    embedder = Embedder()
    store = ChromaStore()

    ids = [row["chunk_id"] for row in rows]
    documents = [row["text"] for row in rows]

    metadatas = [
        {
            "source": "DSERT StudentModules",
            "page": row["page"],
            "chunk_index": row["chunk_index"],
            "chunk_id": row["chunk_id"],
        }
        for row in rows
    ]

    print("Generating embeddings...")

    embeddings = [
        list(embedder.model.embed([text]))[0].tolist()
        for text in documents
    ]

    print("Embeddings generated:", len(embeddings))

    store.add_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Indexed successfully.")
    print("Collection count:", store.collection.count())


if __name__ == "__main__":
    main()