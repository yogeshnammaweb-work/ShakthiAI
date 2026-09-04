import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.rag_pipeline import RAGPipeline


QUERIES = [
    "POCSO ಕಾಯಿದೆ ಎಂದರೇನು?",
    "ಮಕ್ಕಳ ಹಕ್ಕುಗಳು ಯಾವುವು?",
    "ಅಸುರಕ್ಷಿತ ಸ್ಪರ್ಶ ಎಂದರೇನು?",
    "ಶಾಲೆಯಲ್ಲಿ ಮಕ್ಕಳ ಸುರಕ್ಷತೆ ಏಕೆ ಮುಖ್ಯ?",
]


def main():
    pipeline = RAGPipeline()

    try:
        results = []

        for query in QUERIES:
            start = time.perf_counter()

            result = pipeline.ask(query)

            elapsed = time.perf_counter() - start

            results.append(
                {
                    "query": query,
                    "latency_seconds": elapsed,
                    "confidence": result["confidence"],
                    "refused": result["refused"],
                }
            )

            print("\nQuery:", query)
            print("Latency:", f"{elapsed:.3f}s")
            print("Confidence:", result["confidence"])
            print("Refused:", result["refused"])
            print("Answer:", result["answer"])

        latencies = [
            item["latency_seconds"]
            for item in results
        ]

        average = sum(latencies) / len(latencies)
        minimum = min(latencies)
        maximum = max(latencies)

        print("\n===== LATENCY SUMMARY =====")
        print("Average:", f"{average:.3f}s")
        print("Minimum:", f"{minimum:.3f}s")
        print("Maximum:", f"{maximum:.3f}s")


    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
