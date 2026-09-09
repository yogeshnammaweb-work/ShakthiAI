import json
from pathlib import Path

from embeddings.embedder import Embedder
from vectorstore.sqlite_store import SQLiteStore


DATA = Path(
    r"data\master_dataset\health_master_chunks.jsonl"
)


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {DATA}"
        )

    with DATA.open("r", encoding="utf-8") as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    print("Chunks to index:", len(rows))

    embedder = Embedder()
    store = SQLiteStore()

    print("Generating embeddings and storing in SQLite...")

    for index, row in enumerate(rows, start=1):
        embedding = list(
            embedder.model.embed([row["text"]])
        )[0]

        metadata_source = row.get(
            "source",
            row.get("dataset", "unknown"),
        )

        chunk_id = row.get(
            "chunk_id",
            row.get("id"),
        )

        if not chunk_id:
            raise ValueError(
                f"Missing chunk_id/id at row {index}: {row}"
            )

        store.add_document(
            document_id=chunk_id,
            text=row["text"],
            embedding=embedding,
            source=metadata_source,
            page=row.get("page"),
            chunk_index=row.get("chunk_index"),
            chunk_id=chunk_id,
        )

        if index % 10 == 0 or index == len(rows):
            print(
                f"Indexed: {index}/{len(rows)}"
            )

    print()
    print("SQLite indexing complete.")
    print("Document count:", store.count())

    store.close()


if __name__ == "__main__":
    main()
