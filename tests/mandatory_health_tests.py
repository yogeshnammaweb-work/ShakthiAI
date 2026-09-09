from pipeline.rag_pipeline import RAGPipeline


TESTS = [
    {
        "name": "TEST 1 - Anemia symptoms",
        "query": "ನನಗೆ ತುಂಬಾ ಸುಸ್ತು ಅನಿಸುತ್ತೆ ಮತ್ತು ತಲೆ ಸುತ್ತುತ್ತೆ, ನನಗೆ ಏನಾಗಿದೆ ಶಕ್ತಿ AI?",
    },
    {
        "name": "TEST 2 - Iron tablet black stool",
        "query": "ಐರನ್ ಮಾತ್ರೆ ತಿಂದಮೇಲೆ ಕಪ್ಪು ಮಲ ಬಂದರೆ ಹೆದರಬೇಕಾ? ಮಾತ್ರೆ ನಿಲ್ಲಿಸಬೇಕಾ?",
    },
    {
        "name": "TEST 3 - Blood cells and hemoglobin",
        "query": "ಮಾನವನ ರಕ್ತದಲ್ಲಿರುವ ಮುಖ್ಯ ಕಣಗಳು ಯಾವುವು ಮತ್ತು ಹಿಮೋಗ್ಲೋಬಿನ್ ಕೆಲಸವೇನು?",
    },
    {
        "name": "TEST 4 - Out of domain",
        "query": "ಕ್ರಿಕೆಟ್ ವಿಶ್ವಕಪ್ ಯಾರು ಗೆದ್ದರು?",
    },
]


def main():
    pipeline = RAGPipeline()

    print("=" * 80)
    print("SHAKTHI AI - MANDATORY HEALTH VALIDATION")
    print("=" * 80)

    for test in TESTS:
        print()
        print("=" * 80)
        print(test["name"])
        print("=" * 80)
        print("QUERY:")
        print(test["query"])
        print()

        try:
            result = pipeline.ask(test["query"])

            print("RETRIEVED TOP-2:")

            for index, source in enumerate(
                result["sources"],
                start=1,
            ):
                print()
                print(f"[{index}] SCORE: {result['scores'][index - 1]:.6f}")
                print("SOURCE:", source.get("source"))
                print("PAGE:", source.get("page"))
                print("TEXT:", source.get("text", "")[:500])

            print()
            print("ANSWER:")
            print(result["answer"])

            print()
            print("CONFIDENCE:", result["confidence"])
            print("REFUSED:", result["refused"])

        except Exception as exc:
            print()
            print("ERROR:")
            print(exc)

    pipeline.close()

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()