from fastembed import TextEmbedding
import chromadb


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"Loading model: {MODEL_NAME}")
embedder = TextEmbedding(model_name=MODEL_NAME)

client = chromadb.EphemeralClient()

collection = client.create_collection(
    name="kannada_test"
)

documents = [
    "ನೀರಿನ ಕುದಿಯುವ ತಾಪಮಾನವು ಸಾಮಾನ್ಯ ವಾತಾವರಣದ ಒತ್ತಡದಲ್ಲಿ 100 ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್ ಆಗಿದೆ.",
    "ಸಸ್ಯಗಳು ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆಯ ಮೂಲಕ ತಮ್ಮ ಆಹಾರವನ್ನು ತಯಾರಿಸುತ್ತವೆ.",
    "ಭಾರತದ ರಾಜಧಾನಿ ನವದೆಹಲಿ.",
    "ಮಾನವ ದೇಹದಲ್ಲಿ ಹೃದಯವು ರಕ್ತವನ್ನು ಪಂಪ್ ಮಾಡುತ್ತದೆ.",
]

embeddings = [
    list(embedder.embed([doc]))[0].tolist()
    for doc in documents
]

collection.add(
    ids=["test1", "test2", "test3", "test4"],
    documents=documents,
    embeddings=embeddings,
)

queries = [
    "ನೀರಿನ ಕುದಿಯುವ ತಾಪಮಾನ ಎಷ್ಟು?",
    "ಸಸ್ಯಗಳು ತಮ್ಮ ಆಹಾರವನ್ನು ಹೇಗೆ ತಯಾರಿಸುತ್ತವೆ?",
    "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?",
    "ಮಾನವ ದೇಹದಲ್ಲಿ ರಕ್ತವನ್ನು ಪಂಪ್ ಮಾಡುವ ಅಂಗ ಯಾವುದು?",
]

for query in queries:
    query_embedding = list(embedder.embed([query]))[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
    )

    print("\nQUERY:", query)

    for i, doc in enumerate(results["documents"][0], 1):
        print(f"{i}. {doc}")

    print("DISTANCES:", results["distances"][0])