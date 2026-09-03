from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore


embedder = Embedder()
store = ChromaStore()

documents = [
    "ನೀರಿನ ಕುದಿಯುವ ತಾಪಮಾನವು ಸಾಮಾನ್ಯ ವಾತಾವರಣದ ಒತ್ತಡದಲ್ಲಿ 100 ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್ ಆಗಿದೆ.",
    "ಸಸ್ಯಗಳು ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆಯ ಮೂಲಕ ತಮ್ಮ ಆಹಾರವನ್ನು ತಯಾರಿಸುತ್ತವೆ.",
    "ಭಾರತದ ರಾಜಧಾನಿ ನವದೆಹಲಿ.",
    "ಮಾನವ ದೇಹದಲ್ಲಿ ಹೃದಯವು ರಕ್ತವನ್ನು ಪಂಪ್ ಮಾಡುತ್ತದೆ.",
]

embeddings = [
    embedder.embed(doc).tolist()
    for doc in documents
]

store.collection.add(
    ids=["test1", "test2", "test3", "test4"],
    documents=documents,
    embeddings=embeddings,
)

query = "ಸಸ್ಯಗಳು ತಮ್ಮ ಆಹಾರವನ್ನು ಹೇಗೆ ತಯಾರಿಸುತ್ತವೆ?"

query_embedding = embedder.embed(query).tolist()

results = store.search(query_embedding, k=2)

print("\nQuery:")
print(query)

print("\nTop 2 results:")

for i, doc in enumerate(results["documents"][0], 1):
    print(f"{i}. {doc}")

print("\nDistances:")
print(results["distances"][0])