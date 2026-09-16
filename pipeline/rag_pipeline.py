from retrieval.sqlite_retriever import SQLiteRetriever
from inference.llama_model import LlamaModel


class RAGPipeline:
    def __init__(self):
        self.retriever = SQLiteRetriever(k=2)
        self.llama = LlamaModel()

        self.confidence_threshold = 0.65

        # Terms representing the actual Shakthi AI knowledge domain.
        # These are used for OOD/refusal detection.
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
            "ಹಿಮೋಗ್ಲೋಬಿನ್",
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

        # Common words that do not provide useful evidence
        # during lexical grounding comparison.
        self.grounding_stop_words = {
            "ಇದು",
            "ಅದು",
            "ಇದರಿಂದ",
            "ಅದರಿಂದ",
            "ಇದರ",
            "ಅದರ",
            "ಇವು",
            "ಅವು",
            "ಇದ್ದರೆ",
            "ಇರುವುದು",
            "ಇರುತ್ತದೆ",
            "ಇರಬಹುದು",
            "ಆಗಬಹುದು",
            "ಮತ್ತು",
            "ಅಥವಾ",
            "ಆದರೆ",
            "ಎಂದು",
            "ಎಂಬ",
            "ಎಂಬುದು",
            "ಯಾವ",
            "ಯಾವುದು",
            "ಎಷ್ಟು",
            "ಏನು",
            "ಹಾಗೂ",
            "ಮಾತ್ರ",
            "ಕೂಡ",
            "ಸಹ",
            "ಈ",
            "ಆ",
            "ಒಂದು",
            "ಕೆಲವು",
            "ಮೇಲೆ",
            "ಕೆಳಗೆ",
            "ನಂತರ",
            "ಮೊದಲು",
            "ತುಂಬಾ",
            "ಹೆಚ್ಚು",
            "ಕಡಿಮೆ",
            "ಎಲ್ಲಾ",
            "ಅನ್ನು",
            "ಅನ್ನ",
            "ಕ್ಕೆ",
            "ಕೆ",
            "ದ",
            "ಯ",
            "ಗೆ",
            "the",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "can",
            "may",
            "this",
            "that",
        }

        # Critical factual concepts.
        #
        # If Gemma uses one of these concepts, the same concept
        # should exist in the retrieved evidence.
        self.critical_grounding_concepts = {
            "ಕಪ್ಪು",
            "ಕೆಮ್ಮು",
            "ರಕ್ತಹೀನತೆ",
            "ಕಬ್ಬಿಣ",
            "ಐರನ್",
            "ಹಿಮೋಗ್ಲೋಬಿನ್",
            "ದಣಿವು",
            "ದೌರ್ಬಲ್ಯ",
            "ತಲೆ ಸುತ್ತುವುದು",
            "ತಲೆ ಹಗುರ",
            "ಉಸಿರಾಟ",
            "ಮಾತ್ರೆ",
            "ರಕ್ತ",
            "ಕೆಂಪು ರಕ್ತಕಣ",
            "ಬಿಳಿ ರಕ್ತಕಣ",
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
- ಯಾವುದೇ ವೈದ್ಯಕೀಯ ಅಥವಾ ಆರೋಗ್ಯ ವಿಷಯದಲ್ಲಿ, ನೀಡಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ ಇರುವ ಸಂಗತಿಗಳನ್ನು ಮಾತ್ರ ಹೇಳಿ.
- ಮೂಲ ಮಾಹಿತಿಯಲ್ಲಿ ಇರುವ ನಿರ್ದಿಷ್ಟ ಪದ ಅಥವಾ ಸಂಗತಿಯನ್ನು ಬದಲಾಯಿಸಿ ಹೊಸ ಅರ್ಥವನ್ನು ಸೃಷ್ಟಿಸಬೇಡಿ.
- ಮೂಲ ಮಾಹಿತಿಯಲ್ಲಿ ಇರುವ ಪ್ರಮುಖ ಆರೋಗ್ಯ ಸಂಬಂಧಿತ ಪದಗಳನ್ನು ಸಾಧ್ಯವಾದಷ್ಟು ನಿಖರವಾಗಿ ಉಳಿಸಿ.

DSERT ಮಾಹಿತಿ:
{context}

ವಿದ್ಯಾರ್ಥಿಯ ಪ್ರಶ್ನೆ:
{query}

ಕನ್ನಡದಲ್ಲಿ ನೇರವಾದ ಪೂರ್ಣ ಉತ್ತರ:
"""

        return prompt

    def _build_grounding_retry_prompt(
        self,
        query: str,
        sources: list,
    ) -> str:
        context_parts = []

        for index, source in enumerate(sources, start=1):
            text = source.get("text", "").strip()

            if text:
                context_parts.append(
                    f"ಮಾಹಿತಿ {index}:\n{text}"
                )

        context = "\n\n".join(context_parts)

        return f"""ನೀವು Shakthi AI ಎಂಬ ಆಫ್‌ಲೈನ್ ಶಿಕ್ಷಣ ಮತ್ತು ಮಕ್ಕಳ ಸುರಕ್ಷತಾ ಸಹಾಯಕ.

ಹಿಂದಿನ ಉತ್ತರವು ನೀಡಿರುವ ಮೂಲ ಮಾಹಿತಿಯಿಂದ ಸಂಪೂರ್ಣವಾಗಿ ಬೆಂಬಲಿತವಾಗಿಲ್ಲ.
ಈಗ ಮೂಲ ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಆಧರಿಸಿ ಹೊಸ ಉತ್ತರವನ್ನು ರಚಿಸಿ.

ಕಟ್ಟುನಿಟ್ಟಿನ ನಿಯಮಗಳು:

- ಕೆಳಗೆ ನೀಡಿರುವ ಮೂಲ ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಬಳಸಿ.
- ಯಾವುದೇ ಹೊಸ ಸಂಗತಿಯನ್ನು ಸೇರಿಸಬೇಡಿ.
- ಊಹಿಸಬೇಡಿ.
- ಮೂಲ ಮಾಹಿತಿಯಲ್ಲಿ ಇಲ್ಲದ ಆರೋಗ್ಯ ಮಾಹಿತಿ ನೀಡಬೇಡಿ.
- ಮೂಲ ಮಾಹಿತಿಯಲ್ಲಿರುವ ಪ್ರಮುಖ ಸಂಗತಿಗಳನ್ನು ಬದಲಾಯಿಸಬೇಡಿ.
- ಮೂಲದಲ್ಲಿರುವ ನಿರ್ದಿಷ್ಟ ಪದಗಳು ಮುಖ್ಯವಾಗಿದ್ದರೆ ಅವುಗಳನ್ನು ನಿಖರವಾಗಿ ಉಳಿಸಿ.
- ಉದಾಹರಣೆಗೆ, ಮೂಲದಲ್ಲಿ "ಕಪ್ಪು ಬಣ್ಣದ ಮಲ" ಎಂದು ಇದ್ದರೆ ಅದನ್ನು ಬೇರೆ ಪದದಿಂದ ಬದಲಾಯಿಸಬೇಡಿ.
- ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ.
- 1 ರಿಂದ 2 ಪೂರ್ಣ ವಾಕ್ಯಗಳಲ್ಲಿ ಉತ್ತರಿಸಿ.
- ಪ್ರಶ್ನೆಗೆ ನೇರವಾಗಿ ಉತ್ತರಿಸಿ.
- reasoning ಅಥವಾ analysis ನೀಡಬೇಡಿ.
- ಪ್ರಶ್ನೆಯನ್ನು ಪುನರಾವರ್ತಿಸಬೇಡಿ.
- ಮೆಟಾ ವಿವರಣೆ ನೀಡಬೇಡಿ.
- ಮಾಹಿತಿಯಲ್ಲಿ ಉತ್ತರ ಸಿಗದಿದ್ದರೆ ಮಾತ್ರ:
  ಈ ಪ್ರಶ್ನೆಗೆ ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ ಉತ್ತರ ಸಿಗಲಿಲ್ಲ.

ಮೂಲ ಮಾಹಿತಿ:
{context}

ವಿದ್ಯಾರ್ಥಿಯ ಪ್ರಶ್ನೆ:
{query}

ಮೂಲ ಮಾಹಿತಿಗೆ ನಿಷ್ಠವಾಗಿರುವ ಕನ್ನಡ ಉತ್ತರ:
"""

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

        # Direct anchor matching.
        for anchor in self.domain_anchors:
            normalized_anchor = self._normalize_text(anchor)

            if normalized_anchor and normalized_anchor in normalized_query:
                return True

        # Conservative Kannada inflection handling.
        inflection_suffixes = (
            "ದನ್ನು",
            "ದಿಂದ",
            "ದಲ್ಲಿ",
            "ದಲಿ",
            "ದ",
            "ಕ್ಕೆ",
            "ಕೆ",
            "ಯನ್ನು",
            "ಯಿಂದ",
            "ಯಲ್ಲಿ",
            "ಯಲಿ",
            "ಯ",
            "ಗಳನ್ನು",
            "ಗಳಿಗೆ",
            "ಗಳಿಂದ",
            "ಗಳಲ್ಲಿ",
            "ಗಳ",
        )

        query_terms = normalized_query.split()

        for term in query_terms:
            for suffix in inflection_suffixes:
                if not term.endswith(suffix):
                    continue

                if len(term) <= len(suffix) + 2:
                    continue

                stem = term[:-len(suffix)]

                for anchor in self.domain_anchors:
                    normalized_anchor = self._normalize_text(anchor)

                    if stem == normalized_anchor:
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
                    "symptom_boost": metadata.get(
                        "symptom_boost",
                        0.0,
                    ),
                    "health_boost": metadata.get(
                        "health_boost",
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
        # domain relevance.
        if best_score < self.confidence_threshold:
            return False

        # Query must contain a known domain concept.
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
            "ಮೂಲ ಮಾಹಿತಿಗೆ ನಿಷ್ಠವಾಗಿರುವ ಕನ್ನಡ ಉತ್ತರ:",
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

    def _extract_grounding_terms(self, text: str) -> set:
        normalized = self._normalize_text(text)

        terms = set()

        for term in normalized.split():
            if not term:
                continue

            if term in self.grounding_stop_words:
                continue

            if len(term) < 2:
                continue

            terms.add(term)

        return terms

    def _term_supported_by_source(
        self,
        term: str,
        source_terms: set,
    ) -> bool:
        if term in source_terms:
            return True

        # Conservative Kannada suffix handling.
        suffixes = (
            "ಗಳನ್ನು",
            "ಗಳಿಗೆ",
            "ಗಳಿಂದ",
            "ಗಳಲ್ಲಿ",
            "ವನ್ನು",
            "ದಿಂದ",
            "ದಲ್ಲಿ",
            "ದಲಿ",
            "ಕ್ಕೆ",
            "ಕೆ",
            "ಯನ್ನು",
            "ಯಿಂದ",
            "ಯಲ್ಲಿ",
            "ಯಲಿ",
            "ದ",
            "ಯ",
            "ಗಳ",
        )

        for suffix in suffixes:
            if term.endswith(suffix):
                if len(term) <= len(suffix) + 2:
                    continue

                stem = term[:-len(suffix)]

                if stem in source_terms:
                    return True

        return False

    def _validate_grounding(
        self,
        answer: str,
        sources: list,
    ) -> bool:
        """
        Conservative evidence-grounding validation.

        This does not determine whether a medical statement is
        medically true. It checks whether the generated answer
        is supported by the retrieved evidence.

        The validator intentionally allows reasonable Kannada
        paraphrasing while blocking obvious unsupported concepts.
        """
        if not answer or not sources:
            return False

        source_texts = [
            str(source.get("text", "")).strip()
            for source in sources
            if isinstance(source, dict)
            and str(source.get("text", "")).strip()
        ]

        if not source_texts:
            return False

        combined_source = " ".join(source_texts)

        answer_normalized = self._normalize_text(answer)
        source_normalized = self._normalize_text(combined_source)

        if not answer_normalized or not source_normalized:
            return False

        # ---------------------------------------------------------
        # 1. Critical factual concept check
        # ---------------------------------------------------------
        #
        # If an important concept appears in the generated answer,
        # it must also occur in the retrieved evidence.
        #
        # This catches substitutions such as:
        #
        # Source: ಕಪ್ಪು ಬಣ್ಣದ ಮಲ
        # Answer: ಕೆಮ್ಮು ಬಣ್ಣದ ಮಲ
        #
        for concept in self.critical_grounding_concepts:
            if concept in answer_normalized:
                if concept not in source_normalized:
                    return False

        # ---------------------------------------------------------
        # 2. Meaningful lexical overlap
        # ---------------------------------------------------------
        answer_terms = self._extract_grounding_terms(answer)
        source_terms = self._extract_grounding_terms(combined_source)

        if not answer_terms:
            return False

        supported_terms = {
            term
            for term in answer_terms
            if self._term_supported_by_source(
                term,
                source_terms,
            )
        }

        overlap_count = len(supported_terms)

        # For very short answers, at least one meaningful term
        # should be grounded.
        if len(answer_terms) <= 3:
            if overlap_count < 1:
                return False

        # For longer answers, require at least two grounded
        # meaningful terms.
        elif overlap_count < 2:
            return False

        # Do not require a strict percentage of identical words.
        # Kannada paraphrasing can legitimately change morphology
        # and function words while preserving the same meaning.
        #
        # However, if the answer contains many content terms,
        # require a reasonable proportion to be supported.
        overlap_ratio = overlap_count / len(answer_terms)

        if len(answer_terms) >= 6 and overlap_ratio < 0.30:
            return False

        return True

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
            "ಮೂಲ ಮಾಹಿತಿಗೆ ನಿಷ್ಠವಾಗಿರುವ ಕನ್ನಡ ಉತ್ತರ",
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

        # ---------------------------------------------------------
        # Relevance gate
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # First Gemma generation
        # ---------------------------------------------------------
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

        first_pass_valid = (
            self._validate_answer(
                answer,
                sources,
            )
            and self._validate_grounding(
                answer,
                sources,
            )
        )

        # ---------------------------------------------------------
        # Strict grounding retry
        # ---------------------------------------------------------
        if not first_pass_valid:
            retry_prompt = self._build_grounding_retry_prompt(
                query=query,
                sources=sources,
            )

            answer = self.llama.generate(
                retry_prompt,
                max_tokens=96,
                temperature=0.0,
            )

            answer = self._clean_answer(answer)

            retry_valid = (
                self._validate_answer(
                    answer,
                    sources,
                )
                and self._validate_grounding(
                    answer,
                    sources,
                )
            )

            # If the second generation is still unsupported,
            # fail safely instead of returning a hallucinated answer.
            if not retry_valid:
                return self._refusal_response(
                    sources=sources,
                    scores=scores,
                    confidence=confidence,
                )

        # ---------------------------------------------------------
        # Final response contract
        # ---------------------------------------------------------
        return {
            "answer": answer,
            "sources": sources,
            "scores": scores,
            "confidence": confidence,
            "refused": False,
        }

    def close(self):
        self.retriever.close()
