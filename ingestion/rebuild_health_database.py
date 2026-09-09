import json
import sqlite3
from pathlib import Path

from embeddings.embedder import Embedder


MASTER_PATH = Path(
    "data/master_dataset/health_master_chunks.jsonl"
)

DB_PATH = Path(
    "database/knowledge.db"
)


def load_master_dataset():
    records = []

    with MASTER_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {MASTER_PATH}, "
                    f"line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def create_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.execute("DROP TABLE IF EXISTS documents")

    connection.execute(
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source TEXT,
            page INTEGER,
            chunk_index INTEGER,
            chunk_id TEXT
        )
        """
    )

    connection.commit()

    return connection


def main():
    print("=" * 80)
    print("REBUILDING SQLITE KNOWLEDGE DATABASE")
    print("=" * 80)

    records = load_master_dataset()

    print(f"MASTER RECORDS: {len(records)}")

    if len(records) != 181:
        raise RuntimeError(
            f"Expected 181 records, found {len(records)}"
        )

    ids = [record.get("id") for record in records]

    if len(set(ids)) != len(ids):
        raise RuntimeError(
            "Duplicate IDs detected in master dataset."
        )

    print("Loading embedding model...")
    embedder = Embedder()

    # Verify the actual embedding dimension before rebuilding.
    test_embedding = embedder.embed(
        "ರಕ್ತದಲ್ಲಿರುವ ಹಿಮೋಗ್ಲೋಬಿನ್ ಏನು ಮಾಡುತ್ತದೆ?"
    )

    expected_dimension = len(test_embedding)

    print(
        f"EMBEDDING DIMENSION FROM MODEL: "
        f"{expected_dimension}"
    )

    if expected_dimension != 384:
        raise RuntimeError(
            f"Expected 384-dimensional embeddings, "
            f"found {expected_dimension}"
        )

    print("Creating SQLite database...")
    connection = create_database()

    inserted = 0

    try:
        for index, record in enumerate(records, start=1):
            record_id = record["id"]
            text = record["text"]

            embedding = embedder.embed(text)

            if len(embedding) != expected_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch for "
                    f"{record_id}: "
                    f"expected {expected_dimension}, "
                    f"found {len(embedding)}"
                )

            embedding_bytes = embedding.tobytes()

            connection.execute(
                """
                INSERT INTO documents (
                    id,
                    text,
                    embedding,
                    source,
                    page,
                    chunk_index,
                    chunk_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    text,
                    embedding_bytes,
                    record.get("source"),
                    record.get("page"),
                    record.get("chunk_index"),
                    record.get("chunk_id"),
                ),
            )

            inserted += 1

            if inserted % 10 == 0 or inserted == len(records):
                print(
                    f"Indexed {inserted}/{len(records)}"
                )

        connection.commit()

    finally:
        connection.close()

    verification = sqlite3.connect(DB_PATH)

    try:
        row = verification.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()

        db_count = row[0]

        embedding_row = verification.execute(
            "SELECT embedding FROM documents LIMIT 1"
        ).fetchone()

        if embedding_row is None:
            raise RuntimeError(
                "Database contains no embeddings."
            )

        embedding_bytes = embedding_row[0]

        # Current Embedder returns float64 values.
        # float64 uses 8 bytes per value.
        if len(embedding_bytes) % 8 != 0:
            raise RuntimeError(
                "Stored embedding byte length is not "
                "compatible with float64."
            )

        stored_dimension = len(embedding_bytes) // 8

    finally:
        verification.close()

    print()
    print("=" * 80)
    print("DATABASE REBUILD COMPLETE")
    print("=" * 80)
    print(f"MASTER RECORDS: {len(records)}")
    print(f"INSERTED RECORDS: {inserted}")
    print(f"DATABASE RECORDS: {db_count}")
    print(
        f"MODEL EMBEDDING DIMENSION: "
        f"{expected_dimension}"
    )
    print(
        f"STORED EMBEDDING DIMENSION: "
        f"{stored_dimension}"
    )
    print(f"DATABASE: {DB_PATH}")

    if db_count != 181:
        raise RuntimeError(
            f"Expected 181 database records, "
            f"found {db_count}"
        )

    if stored_dimension != expected_dimension:
        raise RuntimeError(
            f"Stored embedding dimension mismatch: "
            f"expected {expected_dimension}, "
            f"found {stored_dimension}"
        )

    print()
    print("=" * 80)
    print("DATABASE VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
