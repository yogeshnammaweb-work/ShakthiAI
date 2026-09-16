from pipeline.rag_pipeline import RAGPipeline


TEST_QUERIES = [
    "ರಕ್ತಹೀನತೆಯ ಲಕ್ಷಣಗಳು ಯಾವುವು?",
    "ಕಬ್ಬಿಣದ ಕೊರತೆ ಎಂದರೇನು?",
    "ಐರನ್ ಮಾತ್ರೆ ತಿಂದಮೇಲೆ ಕಪ್ಪು ಮಲ ಬಂದರೆ ಹೆದರಬೇಕಾ?",
]


def main():
    pipeline = RAGPipeline()

    print("=" * 80)
    print("SHAKTHI AI - DOMAIN + GROUNDING REGRESSION TEST")
    print("=" * 80)

    for index, query in enumerate(TEST_QUERIES, start=1):
        print(f"\nTEST {index}")
        print("QUERY:", query)

        try:
            result = pipeline.ask(query)

            print("ANSWER:", result["answer"])
            print("CONFIDENCE:", result["confidence"])
            print("REFUSED:", result["refused"])

            print("SOURCES:")
            for source in result["sources"]:
                print(
                    "-",
                    source.get("source"),
                    "|",
                    source.get("chunk_id"),
                )

            print("SCORES:", result["scores"])

        except Exception as exc:
            print("ERROR:", type(exc).__name__, exc)

    pipeline.close()


if __name__ == "__main__":
    main()
