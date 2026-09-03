from retrieval.sqlite_retriever import SQLiteRetriever
from inference.llama_model import LlamaModel


class RAGPipeline:
    def __init__(self):
        # Project requirement: retrieve exactly top 2 sources
        self.retriever = SQLiteRetriever(k=2)
        self.llama = LlamaModel()

    def _build_prompt(self, query: str, sources: list) -> str:
        """
        Build a strict grounded prompt for Qwen2.5 3B Instruct.

        The model must answer only from the retrieved DSERT content.
        """

        context_parts = []

        for index, source in enumerate(sources, start=1):
            text = source.get("text", "").strip()

            if text:
                context_parts.append(
                    f"ಮಾಹಿತಿ {index}:\n{text}"
                )

        context = "\n\n".join(context_parts)

        prompt = f"""ನೀವು Shakthi AI ಎಂಬ ಆಫ್‌ಲೈನ್ ಶಿಕ್ಷಣ ಮತ್ತು ಮಕ್ಕಳ ಸುರಕ್ಷತಾ ಸಹಾಯಕರು.

ಕೆಳಗೆ ನೀಡಿರುವ DSERT ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಬಳಸಿ ವಿದ್ಯಾರ್ಥಿಯ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಿ.

ಕಟ್ಟುನಿಟ್ಟಿನ ನಿಯಮಗಳು:

- ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ನೀಡಿ.
- 1 ರಿಂದ 2 ಪೂರ್ಣ ವಾಕ್ಯಗಳಲ್ಲಿ ಉತ್ತರಿಸಿ.
- ಪ್ರಶ್ನೆಗೆ ನೇರವಾಗಿ ಉತ್ತರಿಸಿ.
- ನೀಡಿರುವ ಮಾಹಿತಿಯಲ್ಲಿರುವ ವಿಷಯವನ್ನು ಮಾತ್ರ ಬಳಸಿ.
- ಹೊಸ ಮಾಹಿತಿಯನ್ನು ಸೇರಿಸಬೇಡಿ.
- ಊಹಿಸಬೇಡಿ.
- ಪ್ರಶ್ನೆಯನ್ನು ಪುನರಾವರ್ತಿಸಬೇಡಿ.
- "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಉತ್ತರ", "ಮಾಹಿತಿ", "ಮೂಲ", "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರ", "ನಾನು" ಮುಂತಾದ ಮೆಟಾ ಪದಗಳನ್ನು ಬಳಸಬೇಡಿ.
- ಯಾವುದೇ reasoning ಅಥವಾ analysis ನೀಡಬೇಡಿ.
- ಅಪೂರ್ಣ ವಾಕ್ಯ ನೀಡಬೇಡಿ.
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

    def retrieve_only(self, query: str):
        """
        Retrieve exactly top-k=2 documents from SQLite.
        """

        results = self.retriever.retrieve(query)

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        sources = []

        for document, metadata in zip(documents, metadatas):
            sources.append(
                {
                    "text": document,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                }
            )

        scores = [
            1.0 - float(distance)
            for distance in distances
        ]

        return {
            "sources": sources,
            "scores": scores,
            "distances": distances,
        }

    def _clean_answer(self, answer: str) -> str:
        """
        Remove common Qwen meta-prefixes and formatting artifacts.
        """

        answer = answer.strip()

        unwanted_prefixes = [
            "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಉತ್ತರವು ಕನ್ನಡದಲ್ಲಿ ಏನು:",
            "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಉತ್ತರ:",
            "ಉತ್ತರ:",
            "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರ:",
            "ಕನ್ನಡದಲ್ಲಿ ನೇರವಾದ ಪೂರ್ಣ ಉತ್ತರ:",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in unwanted_prefixes:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    changed = True

        # Remove accidental quotation marks around the answer.
        if len(answer) >= 2:
            if (
                answer.startswith('"')
                and answer.endswith('"')
            ):
                answer = answer[1:-1].strip()

            elif (
                answer.startswith("“")
                and answer.endswith("”")
            ):
                answer = answer[1:-1].strip()

        return answer

    def ask(self, query: str):
        """
        Full RAG pipeline:

        Kannada Query
            -> multilingual embedding
            -> SQLite top-2 retrieval
            -> grounded prompt
            -> local Qwen2.5 3B inference
            -> cleaned Kannada answer
        """

        query = query.strip()

        if not query:
            return {
                "answer": "ಈ ಪ್ರಶ್ನೆಗೆ ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ ಉತ್ತರ ಸಿಗಲಿಲ್ಲ.",
                "sources": [],
                "scores": [],
                "confidence": 0.0,
                "refused": True,
            }

        retrieval = self.retrieve_only(query)

        sources = retrieval["sources"]
        scores = retrieval["scores"]

        # No retrieved knowledge
        if not sources:
            return {
                "answer": "ಈ ಪ್ರಶ್ನೆಗೆ ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ ಉತ್ತರ ಸಿಗಲಿಲ್ಲ.",
                "sources": [],
                "scores": [],
                "confidence": 0.0,
                "refused": True,
            }

        prompt = self._build_prompt(
            query=query,
            sources=sources,
        )

        # Deterministic generation for factual educational QA.
        answer = self.llama.generate(
            prompt,
            max_tokens=128,
            temperature=0.0,
        )

        answer = self._clean_answer(answer)

        # Safety fallback for empty model output.
        if not answer:
            answer = (
                "ಈ ಪ್ರಶ್ನೆಗೆ ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯಲ್ಲಿ "
                "ಉತ್ತರ ಸಿಗಲಿಲ್ಲ."
            )

        return {
            "answer": answer,
            "sources": sources,
            "scores": scores,
            "confidence": max(scores) if scores else 0.0,
            "refused": False,
        }

    def close(self):
        self.retriever.close()