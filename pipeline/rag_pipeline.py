from retrieval.sqlite_retriever import SQLiteRetriever
from inference.llama_model import LlamaModel


class RAGPipeline:
    def __init__(self):
        self.retriever = SQLiteRetriever(k=2)
        self.llama = LlamaModel()

        self.confidence_threshold = 0.65

        # Terms representing the actual DSERT knowledge domain.
        # These are used only for OOD/refusal detection.
        self.domain_anchors = {
            "pocso",
            "ಮಕ್ಕಳ",
            "ಮಗು",
            "ಮಕ್ಕಳು",
            "ಬಾಲ",
            "ಬಾಲಕ",
            "ಬಾಲಕಿ",
            "ಲೈಂಗಿಕ",
            "ಶೋಷಣೆ",
            "ಅಸುರಕ್ಷಿತ",
            "ಸ್ಪರ್ಶ",
            "ಸುರಕ್ಷತೆ",
            "ಸುರಕ್ಷಿತ",
            "ಹಕ್ಕು",
            "ಹಕ್ಕುಗಳು",
            "ಮಕ್ಕಳಹಕ್ಕು",
            "ಶಿಕ್ಷಣ",
            "ಶಾಲೆ",
            "ವಿದ್ಯಾರ್ಥಿ",
            "ವಿದ್ಯಾರ್ಥಿಗಳು",
            "ಶಿಕ್ಷಕ",
            "ಶಿಕ್ಷಕರು",
            "ಶಾಲಾ",
            "ಕಿರುಕುಳ",
            "ಹಿಂಸೆ",
            "ದೌರ್ಜನ್ಯ",
            "ರಕ್ಷಣೆ",
            "ಸಹಾಯ",
            "ಆರೋಗ್ಯ",
            "ಆಹಾರ",
            "ಪೌಷ್ಟಿಕ",
            "ಪೋಷಣೆ",
            "ಪೋಷಕ",
            "ಪೋಷಕಾಂಶ",
            "ಹಣ್ಣು",
            "ತರಕಾರಿ",
            "ಧಾನ್ಯ",
            "ಹಾಲು",
            "ನೀರು",
            "ಸ್ವಚ್ಛತೆ",
            "ರೋಗ",
            "ಸೋಂಕು",
            "ಅತಿಸಾರ",
            "ಜಿಂಕ್",
            "ಒಆರಎಸ್",
            "ಕ್ಷಯರೋಗ",
            "ಟಿಬಿ",
            "ವ್ಯಾಯಾಮ",
            "ಮಾನಸಿಕಆರೋಗ್ಯ",
            "ಸುಸ್ತು",
            "ದಣಿವು",
            "ದೌರ್ಬಲ್ಯ",
            "ತಲೆ",
            "ಸುತ್ತು",
            "ತಲೆ ಸುತ್ತುವುದು",
            "ತಲೆ ಹಗುರ",
            "ಉಸಿರಾಟದ ತೊಂದರೆ",
            "ರಕ್ತಹೀನತೆ",
            "ಕಬ್ಬಿಣ",
            "ಐರನ್",
            "ರಕ್ತ",
            "ರಕ್ತಕಣ",
            "ರಕ್ತ ಕಣ",
            "ಕೆಂಪು ರಕ್ತಕಣ",
            "ಕೆಂಪು ರಕ್ತ ಕಣ",
            "ಬಿಳಿ ರಕ್ತಕಣ",
            "ಬಿಳಿ ರಕ್ತ ಕಣ",
            "ಹಿಮೋಗ್ಲೋಬಿನ್",
            "ಪ್ಲಾಸ್ಮಾ",
            "ಕಿರುತಟ್ಟೆಗಳು",
        }

    def _build_prompt(self, query: str, sources: list) -> str:
        context_parts = []

        for index, source in enumerate(sources, start=1):
            text = source.get("text", "").strip()

            if text:
                context_parts.append(
                    f"ಮಾಹಿತಿ {index}:\n{text}"
                )

        context = "\n\n".join(context_parts)

        prompt = f"""ನೀವು Shakthi AI ಎಂಬ ಆಫ್‌ಲೈನ್ ಶಿಕ್ಷಣ ಮತ್ತು ಮಕ್ಕಳ ಸುರಕ್ಷತಾ ಸಹಾಯಕ.

ಕೆಳಗೆ ನೀಡಿರುವ DSERT ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಬಳಸಿ ವಿದ್ಯಾರ್ಥಿಯ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಿ.

ಕಟ್ಟುನಿಟ್ಟಿನ ನಿಯಮಗಳು:

- ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ನೀಡಿ.
- 1 ರಿಂದ 2 ಪೂರ್ಣ ವಾಕ್ಯಗಳಲ್ಲಿ ಉತ್ತರಿಸಿ.
- ಪ್ರಶ್ನೆಗೆ ನೇರವಾಗಿ ಉತ್ತರಿಸಿ.
- ನೀಡಿರುವ ಮಾಹಿತಿಯಲ್ಲಿರುವ ವಿಷಯವನ್ನು ಮಾತ್ರ ಬಳಸಿ.
- ಹೊಸ ಮಾಹಿತಿಯನ್ನು ಸೇರಿಸಬೇಡಿ.
- ಊಹಿಸಬೇಡಿ.
- ಪ್ರಶ್ನೆಯನ್ನು ಪುನರಾವರ್ತಿಸಬೇಡಿ.
- ಯಾವುದೇ reasoning ಅಥವಾ analysis ನೀಡಬೇಡಿ.
- ಅಪೂರ್ಣ ವಾಕ್ಯ ನೀಡಬೇಡಿ.
- ಇಂಗ್ಲಿಷ್ ಅಥವಾ ಇತರ ಭಾಷೆಯ ಪದಗಳನ್ನು ಅಗತ್ಯವಿಲ್ಲದೆ ಬಳಸಬೇಡಿ.
- ಮೆಟಾ ಪದಗಳು ಅಥವಾ ಸೂಚನೆಗಳನ್ನು ನೀಡಬೇಡಿ.
- ವ್ಯಾಖ್ಯಾನ ಕೇಳಿದರೆ, ಮಾಹಿತಿಯಲ್ಲಿರುವ ವ್ಯಾಖ್ಯಾನವನ್ನು ನೇರವಾಗಿ ನೀಡಿ.
- ವ್ಯಾಖ್ಯಾನಕ್ಕೆ ಉದಾಹರಣೆಗಳಿದ್ದರೆ, ಅಗತ್ಯವಿದ್ದಾಗ 1 ಅಥವಾ 2 ಉದಾಹರಣೆಗಳನ್ನು ನೀಡಿ.
- ಮಾಹಿತಿಯಲ್ಲಿ ಉತ್ತರ ಸಿಗದಿದ್ದರೆ ಮಾತ್ರ:
  ಈ ಪ್ರಶ್ನೆಗೆ ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ ಉತ್ತರ ಸಿಗಲಿಲ್ಲ.

DSERT ಮಾಹಿತಿ:
{context}

ವಿದ್ಯಾರ್ಥಿಯ ಪ್ರಶ್ನೆ:
{query}

ಕನ್ನಡದಲ್ಲಿ ನೇರವಾದ ಪೂರ್ಣ ಉತ್ತರ:
"""

        return prompt

    def _normalize_text(self, text: str) -> str:
        import unicodedata

        text = (text or "").lower()
        text = unicodedata.normalize("NFC", text)

        return "".join(
            char
            for char in text
            if (
                unicodedata.category(char).startswith(("L", "M", "N"))
                or char.isspace()
            )
        )

    def _query_has_domain_anchor(self, query: str) -> bool:
        normalized_query = self._normalize_text(query)

        for anchor in self.domain_anchors:
            normalized_anchor = self._normalize_text(anchor)

            if normalized_anchor and normalized_anchor in normalized_query:
                return True

        return False

    def retrieve_only(self, query: str):
        results = self.retriever.retrieve(query)

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        sources = []

        for document, metadata in zip(
            documents,
            metadatas,
        ):
            sources.append(
                {
                    "text": document,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "matched_term_count": metadata.get(
                        "matched_term_count",
                        0,
                    ),
                    "lexical_boost": metadata.get(
                        "lexical_boost",
                        0.0,
                    ),
                }
            )

        scores = [
            max(
                0.0,
                min(
                    1.0,
                    1.0 - float(distance),
                ),
            )
            for distance in distances
        ]

        return {
            "sources": sources,
            "scores": scores,
            "distances": distances,
        }

    def _passes_relevance_gate(
        self,
        query: str,
        sources: list,
        scores: list,
    ) -> bool:
        if not sources or not scores:
            return False

        best_score = max(scores)

        # The embedding model alone must never establish
        # domain relevance for this application.
        if best_score < self.confidence_threshold:
            return False

        # Query must contain at least one known DSERT
        # education/child-safety domain concept.
        if not self._query_has_domain_anchor(query):
            return False

        return True

    def _clean_answer(self, answer: str) -> str:
        answer = (answer or "").strip()

        unwanted_prefixes = [
            "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಉತ್ತರವು ಕನ್ನಡದಲ್ಲಿ ಏನು:",
            "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಉತ್ತರ:",
            "ಉತ್ತರ:",
            "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರ:",
            "ಕನ್ನಡದಲ್ಲಿ ನೇರವಾದ ಪೂರ್ಣ ಉತ್ತರ:",
            "Answer:",
            "ANSWER:",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in unwanted_prefixes:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    changed = True

        if len(answer) >= 2:
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1].strip()

            elif answer.startswith("“") and answer.endswith("”"):
                answer = answer[1:-1].strip()

        return answer

    def _validate_answer(
        self,
        answer: str,
        sources: list,
    ) -> bool:
        if not answer:
            return False

        forbidden_terms = [
            "comforts",
            "manual touch",
            "discomfort",
            "reasoning",
            "analysis",
            "answer:",
            "source:",
        ]

        answer_lower = answer.lower()

        if any(
            term in answer_lower
            for term in forbidden_terms
        ):
            return False

        foreign_script_ranges = [
            ("\u0400", "\u04FF"),
            ("\u0370", "\u03FF"),
            ("\u0600", "\u06FF"),
            ("\u0900", "\u097F"),
            ("\u0980", "\u09FF"),
            ("\u0A00", "\u0A7F"),
            ("\u0A80", "\u0AFF"),
            ("\u0B00", "\u0B7F"),
            ("\u0B80", "\u0BFF"),
            ("\u0C00", "\u0C7F"),
            ("\u0D00", "\u0D7F"),
        ]

        for start, end in foreign_script_ranges:
            if any(
                start <= char <= end
                for char in answer
            ):
                return False

        meta_terms = [
            "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಉತ್ತರ",
            "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರ",
            "ಕನ್ನಡದಲ್ಲಿ ನೇರವಾದ ಪೂರ್ಣ ಉತ್ತರ",
            "reasoning",
            "analysis",
        ]

        if any(
            term in answer
            for term in meta_terms
        ):
            return False

        kannada_chars = sum(
            1
            for char in answer
            if "\u0C80" <= char <= "\u0CFF"
        )

        if kannada_chars < 5:
            return False

        has_source_text = any(
            source.get("text", "").strip()
            for source in sources
        )

        if not has_source_text:
            return False

        suspicious_legal_tokens = [
            "70050",
            "70೮8೦",
            "70೮8೧",
            "7೦88೦",
            "೭೦೦೫೦",
        ]

        if any(
            token in answer
            for token in suspicious_legal_tokens
        ):
            return False

        return True

    def _refusal_response(
        self,
        sources: list,
        scores: list,
        confidence: float,
    ):
        return {
            "answer": (
                "ಈ ಪ್ರಶ್ನೆಗೆ ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ "
                "ಉತ್ತರ ಸಿಗಲಿಲ್ಲ."
            ),
            "sources": sources,
            "scores": scores,
            "confidence": confidence,
            "refused": True,
        }

    def ask(self, query: str):
        query = (query or "").strip()

        if not query:
            return self._refusal_response(
                sources=[],
                scores=[],
                confidence=0.0,
            )

        retrieval = self.retrieve_only(query)

        sources = retrieval["sources"]
        scores = retrieval["scores"]

        if not sources:
            return self._refusal_response(
                sources=[],
                scores=[],
                confidence=0.0,
            )

        confidence = max(scores) if scores else 0.0

        if not self._passes_relevance_gate(
            query=query,
            sources=sources,
            scores=scores,
        ):
            return self._refusal_response(
                sources=sources,
                scores=scores,
                confidence=confidence,
            )

        prompt = self._build_prompt(
            query=query,
            sources=sources,
        )

        answer = self.llama.generate(
            prompt,
            max_tokens=96,
            temperature=0.0,
        )

        answer = self._clean_answer(answer)

        if not self._validate_answer(
            answer,
            sources,
        ):
            return self._refusal_response(
                sources=sources,
                scores=scores,
                confidence=confidence,
            )

        return {
            "answer": answer,
            "sources": sources,
            "scores": scores,
            "confidence": confidence,
            "refused": False,
        }

    def close(self):
        self.retriever.close()




