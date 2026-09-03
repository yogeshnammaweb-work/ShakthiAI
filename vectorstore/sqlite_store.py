import sqlite3
from array import array
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


class SQLiteStore:
    def __init__(self, path: str = "database/knowledge.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.path)

        self._create_tables()

    def _create_tables(self):
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                page INTEGER,
                chunk_index INTEGER,
                chunk_id TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                document_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                FOREIGN KEY(document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE
            )
            """
        )

        self.connection.commit()

    def add_document(
        self,
        document_id: str,
        text: str,
        embedding: Iterable[float],
        source: Optional[str] = None,
        page: Optional[int] = None,
        chunk_index: Optional[int] = None,
        chunk_id: Optional[str] = None,
    ):
        embedding_list = list(embedding)

        embedding_bytes = sqlite3.Binary(
            array("f", embedding_list).tobytes()
        )

        self.connection.execute(
            """
            INSERT OR REPLACE INTO documents
            (
                id,
                text,
                source,
                page,
                chunk_index,
                chunk_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                text,
                source,
                page,
                chunk_index,
                chunk_id,
            ),
        )

        self.connection.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (
                document_id,
                embedding,
                dimension
            )
            VALUES (?, ?, ?)
            """,
            (
                document_id,
                embedding_bytes,
                len(embedding_list),
            ),
        )

        self.connection.commit()

    def count(self) -> int:
        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM documents"
        )

        return cursor.fetchone()[0]

    def get_document(self, document_id: str):
        cursor = self.connection.execute(
            """
            SELECT
                d.id,
                d.text,
                d.source,
                d.page,
                d.chunk_index,
                d.chunk_id,
                e.embedding,
                e.dimension
            FROM documents d
            JOIN embeddings e
                ON d.id = e.document_id
            WHERE d.id = ?
            """,
            (document_id,),
        )

        return cursor.fetchone()

    def search(self, query_embedding, k: int = 2):
        rows = self.connection.execute(
            """
            SELECT
                d.id,
                d.text,
                d.source,
                d.page,
                d.chunk_index,
                d.chunk_id,
                e.embedding,
                e.dimension
            FROM documents d
            JOIN embeddings e
                ON d.id = e.document_id
            """
        ).fetchall()

        if not rows:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

        query = np.asarray(query_embedding, dtype=np.float32)

        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("Query embedding has zero norm.")

        results = []

        for row in rows:
            embedding = np.frombuffer(
                row[6],
                dtype=np.float32,
            )

            if len(embedding) != row[7]:
                raise ValueError(
                    f"Embedding dimension mismatch for {row[0]}"
                )

            embedding_norm = np.linalg.norm(embedding)

            if embedding_norm == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(query, embedding)
                    / (query_norm * embedding_norm)
                )

            distance = 1.0 - similarity

            results.append(
                {
                    "id": row[0],
                    "text": row[1],
                    "source": row[2],
                    "page": row[3],
                    "chunk_index": row[4],
                    "chunk_id": row[5],
                    "distance": distance,
                }
            )

        results.sort(key=lambda item: item["distance"])

        top_results = results[:k]

        return {
            "documents": [
                item["text"]
                for item in top_results
            ],
            "metadatas": [
                {
                    "source": item["source"],
                    "page": item["page"],
                    "chunk_index": item["chunk_index"],
                    "chunk_id": item["chunk_id"],
                }
                for item in top_results
            ],
            "distances": [
                item["distance"]
                for item in top_results
            ],
        }

    def close(self):
        self.connection.close()