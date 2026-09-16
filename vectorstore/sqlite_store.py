import sqlite3
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


class SQLiteStore:
    def __init__(self, path: str = "database/knowledge.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self._validate_schema()

    def _validate_schema(self):
        columns = self.connection.execute(
            "PRAGMA table_info(documents)"
        ).fetchall()

        column_names = {column[1] for column in columns}

        required_columns = {
            "id",
            "text",
            "embedding",
            "source",
            "page",
            "chunk_index",
            "chunk_id",
        }

        missing = required_columns - column_names

        if missing:
            raise RuntimeError(
                "SQLite database schema is missing required columns: "
                f"{sorted(missing)}"
            )

    def _create_tables(self):
        pass

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
        embedding_array = np.asarray(
            list(embedding),
            dtype=np.float64,
        )

        if embedding_array.ndim != 1:
            raise ValueError(
                "Embedding must be a one-dimensional vector."
            )

        if len(embedding_array) == 0:
            raise ValueError(
                "Embedding cannot be empty."
            )

        embedding_bytes = sqlite3.Binary(
            embedding_array.tobytes()
        )

        self.connection.execute(
            """
            INSERT OR REPLACE INTO documents
            (
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
                document_id,
                text,
                embedding_bytes,
                source,
                page,
                chunk_index,
                chunk_id,
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
                id,
                text,
                embedding,
                source,
                page,
                chunk_index,
                chunk_id
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        embedding = np.frombuffer(
            row[2],
            dtype=np.float64,
        )

        return (
            row[0],
            row[1],
            row[3],
            row[4],
            row[5],
            row[6],
            embedding,
            len(embedding),
        )

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize(
            "NFC",
            text or "",
        ).lower()

        return "".join(
            char
            if (
                unicodedata.category(char)[0] in {"L", "M", "N"}
                or char.isspace()
            )
            else " "
            for char in text
        )

    def _extract_query_terms(self, query_text: str):
        normalized_query = self._normalize_text(query_text)

        stop_words = {
            "ನನಗೆ",
            "ನಾನು",
            "ನನ್ನ",
            "ನಮ್ಮ",
            "ನಮ್ಮದು",
            "ಮತ್ತು",
            "ಅಥವಾ",
            "ಇದು",
            "ಇದೆ",
            "ಇದ್ದರೆ",
            "ಇದ್ದಾಗ",
            "ಏನು",
            "ಯಾವ",
            "ಯಾವುದು",
            "ಹೇಗೆ",
            "ಏಕೆ",
            "ಎಷ್ಟು",
            "ಯಾರು",
            "ಎಲ್ಲಿ",
            "ಬಗ್ಗೆ",
            "ಎಂಬ",
            "ಎಂದು",
            "ಆಗ",
            "ಗೆ",
            "ನಲ್ಲಿ",
            "ನಿಂದ",
            "ಇಂದ",
            "ಒಂದು",
            "ಮಾತ್ರ",
            "ಅದು",
            "ಅವರು",
            "ಅವರಿಗೆ",
            "ನೀವು",
            "ನಿಮಗೆ",
            "ಮಾಡಿ",
            "ಮಾಡುವುದು",
            "ಮಾಡಬೇಕು",
            "the",
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
            "is",
            "are",
            "a",
            "an",
            "and",
            "or",
            "can",
            "should",
            "does",
            "do",
        }

        return {
            term
            for term in normalized_query.split()
            if len(term) >= 2 and term not in stop_words
        }

    def search(
        self,
        query_embedding,
        query_text: str = "",
        k: int = 2,
    ):
        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        rows = self.connection.execute(
            """
            SELECT
                id,
                text,
                embedding,
                source,
                page,
                chunk_index,
                chunk_id
            FROM documents
            """
        ).fetchall()

        if not rows:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

        query = np.asarray(
            query_embedding,
            dtype=np.float64,
        )

        if query.ndim != 1:
            raise ValueError(
                "Query embedding must be one-dimensional."
            )

        if len(query) == 0:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError(
                "Query embedding has zero norm."
            )

        normalized_query = self._normalize_text(
            query_text.strip()
        )

        query_terms = self._extract_query_terms(
            query_text
        )

        symptom_concepts = {
            "ಸುಸ್ತು": {
                "ದಣಿವು",
                "ದೌರ್ಬಲ್ಯ",
            },
            "ಸುಸ್ತಾಗಿದೆ": {
                "ದಣಿವು",
                "ದೌರ್ಬಲ್ಯ",
            },
            "ಸುಸ್ತಾಗುತ್ತದೆ": {
                "ದಣಿವು",
                "ದೌರ್ಬಲ್ಯ",
            },
            "ಸುಸ್ತಾಗುತ್ತೆ": {
                "ದಣಿವು",
                "ದೌರ್ಬಲ್ಯ",
            },
            "ತಲೆ ಸುತ್ತುತ್ತೆ": {
                "ತಲೆ ಸುತ್ತುವುದು",
                "ತಲೆ ಹಗುರವಾಗಿರುವ ಅನುಭವ",
            },
            "ತಲೆ ಸುತ್ತುತ್ತದೆ": {
                "ತಲೆ ಸುತ್ತುವುದು",
                "ತಲೆ ಹಗುರವಾಗಿರುವ ಅನುಭವ",
            },
            "ತಲೆ ಸುತ್ತುತ್ತಿದೆ": {
                "ತಲೆ ಸುತ್ತುವುದು",
                "ತಲೆ ಹಗುರವಾಗಿರುವ ಅನುಭವ",
            },
            "ತಲೆ ಸುತ್ತು": {
                "ತಲೆ ಸುತ್ತುವುದು",
                "ತಲೆ ಹಗುರವಾಗಿರುವ ಅನುಭವ",
            },
            "ತಲೆ ಹಗುರ": {
                "ತಲೆ ಹಗುರವಾಗಿರುವ ಅನುಭವ",
                "ತಲೆ ಸುತ್ತುವುದು",
            },
        }

        # Controlled health concepts.
        #
        # These variants allow common Kannada inflections
        # to map back to the same health concept.
        health_concepts = {
            "ರಕ್ತಹೀನತೆ": {
                "ರಕ್ತಹೀನತೆ",
                "ರಕ್ತಹೀನತೆಯ",
                "ರಕ್ತಹೀನತೆಗೆ",
                "ರಕ್ತಹೀನತೆಯಿಂದ",
                "ರಕ್ತಹೀನತೆಯನ್ನು",
            },
            "ಕಬ್ಬಿಣ": {
                "ಕಬ್ಬಿಣ",
                "ಕಬ್ಬಿಣದ",
                "ಕಬ್ಬಿಣಕ್ಕೆ",
                "ಕಬ್ಬಿಣದಿಂದ",
                "ಕಬ್ಬಿಣವನ್ನು",
            },
        }

        detected_symptom_concepts = []

        for phrase, canonical_terms in symptom_concepts.items():
            normalized_phrase = self._normalize_text(
                phrase
            )

            if normalized_phrase in normalized_query:
                detected_symptom_concepts.append(
                    canonical_terms
                )

        detected_health_concepts = []

        for canonical_term, variants in health_concepts.items():
            normalized_canonical_term = self._normalize_text(
                canonical_term
            )

            normalized_variants = {
                self._normalize_text(variant)
                for variant in variants
            }

            if (
                normalized_canonical_term in normalized_query
                or any(
                    variant in normalized_query
                    for variant in normalized_variants
                )
            ):
                detected_health_concepts.append(
                    normalized_variants
                )

        results = []

        for row in rows:
            document_id = row[0]
            text = row[1] or ""
            embedding_blob = row[2]

            embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float64,
            )

            if len(embedding) != len(query):
                raise ValueError(
                    "Embedding dimension mismatch for "
                    f"{document_id}: "
                    f"database={len(embedding)}, "
                    f"query={len(query)}"
                )

            embedding_norm = np.linalg.norm(embedding)

            if embedding_norm == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(query, embedding)
                    / (query_norm * embedding_norm)
                )

            text_for_matching = self._normalize_text(
                text
            )

            text_terms = set(
                text_for_matching.split()
            )

            matched_terms = {
                term
                for term in query_terms
                if term in text_terms
            }

            matched_term_count = len(matched_terms)

            lexical_boost = 0.0

            if (
                normalized_query
                and normalized_query in text_for_matching
            ):
                lexical_boost += 0.10

            if query_terms:
                overlap_ratio = (
                    matched_term_count
                    / len(query_terms)
                )

                lexical_boost += (
                    0.10 * overlap_ratio
                )

            combined_score = similarity

            if matched_term_count > 0:
                combined_score += lexical_boost

            matched_symptom_terms = set()

            for canonical_terms in detected_symptom_concepts:
                for canonical_term in canonical_terms:
                    normalized_canonical_term = (
                        self._normalize_text(
                            canonical_term
                        )
                    )

                    if (
                        normalized_canonical_term
                        in text_for_matching
                    ):
                        matched_symptom_terms.add(
                            canonical_term
                        )

            symptom_boost = min(
                0.12,
                0.06 * len(matched_symptom_terms),
            )

            combined_score += symptom_boost

            # Controlled health-concept boost.
            #
            # This helps queries such as:
            #   ರಕ್ತಹೀನತೆಯ ಲಕ್ಷಣಗಳು
            #
            # match records containing:
            #   ರಕ್ತಹೀನತೆ
            #   ರಕ್ತಹೀನತೆಯ
            #
            # without lowering the global confidence threshold
            # or changing the embedding model.
            matched_health_terms = set()

            for concept_variants in detected_health_concepts:
                for variant in concept_variants:
                    if variant in text_for_matching:
                        matched_health_terms.add(
                            variant
                        )
                        break

            health_boost = 0.0

            if matched_health_terms:
                health_boost = 0.10

            combined_score += health_boost

            # Penalize obvious OCR-garbage chunks without removing
            # legitimate DSERT content.
            tokens = text_for_matching.split()

            garbage_tokens = {
                "ಮ",
                "ಗ",
                "ಭ",
                "ಅಜಾ",
                "ಇ",
                "ಇಪ",
                "ಸ",
                "ಜಾ",
                "ಪಾ",
                "ಸಾಜ",
                "ಾರಾ",
                "ಕಾಣಾ",
                "ಎರ",
                "ಯಾ",
                "ಪಜ",
                "ಫರಾ",
                "ಫಲ",
                "ಲಅಹ್ರಾಹಸಾಣ್",
                "ಸಭಾ",
                "ಅಪಘಾ",
            }

            garbage_token_count = sum(
                1
                for token in tokens
                if token in garbage_tokens
            )

            garbage_ratio = (
                garbage_token_count / len(tokens)
                if tokens
                else 0.0
            )

            if (
                garbage_token_count >= 4
                and garbage_ratio >= 0.10
            ):
                combined_score -= min(
                    0.15,
                    0.05 + 0.10 * garbage_ratio,
                )

            combined_score = max(
                0.0,
                min(1.0, combined_score),
            )

            distance = 1.0 - combined_score

            results.append(
                {
                    "id": document_id,
                    "text": text,
                    "source": row[3],
                    "page": row[4],
                    "chunk_index": row[5],
                    "chunk_id": row[6],
                    "distance": distance,
                    "similarity": similarity,
                    "lexical_boost": lexical_boost,
                    "symptom_boost": symptom_boost,
                    "health_boost": health_boost,
                    "matched_terms": matched_terms,
                    "matched_symptom_terms": matched_symptom_terms,
                    "matched_health_terms": matched_health_terms,
                    "matched_term_count": matched_term_count,
                }
            )

        results.sort(
            key=lambda item: item["distance"]
        )

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
                    "matched_term_count": item[
                        "matched_term_count"
                    ],
                    "lexical_boost": item[
                        "lexical_boost"
                    ],
                    "symptom_boost": item[
                        "symptom_boost"
                    ],
                    "health_boost": item[
                        "health_boost"
                    ],
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
