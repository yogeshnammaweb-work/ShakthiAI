from pathlib import Path
import json
from fastembed import TextEmbedding
import chromadb

DATA = Path(r"data\master_dataset\StudentModules_chunks.jsonl")
OUTPUT = Path(r"data\master_dataset\retrieval_benchmark_results.json")

queries = [
    "ಮಕ್ಕಳ ಹಕ್ಕುಗಳ ರಕ್ಷಣೆ ಎಂದರೇನು?",
    "ಮಕ್ಕಳನ್ನು ಯಾವ ರೀತಿಯ ಹಿಂಸೆ ಮತ್ತು ಶೋಷಣೆಯಿಂದ ರಕ್ಷಿಸಬೇಕು?",
    "18 ವರ್ಷದೊಳಗಿನ ಮಕ್ಕಳ ರಕ್ಷಣೆಗೆ ಯಾವ ಕಾನೂನು ಜಾರಿಗೆ ತರಲಾಗಿದೆ?",
    "ಶಾಲೆಯಲ್ಲಿ ಮಕ್ಕಳ ಸುರಕ್ಷತೆ ಏಕೆ ಮುಖ್ಯ?",
    "ಮಕ್ಕಳಿಗೆ ಸುರಕ್ಷಿತ ವಾತಾವರಣವನ್ನು ಒದಗಿಸುವ ಜವಾಬ್ದಾರಿ ಯಾರದ್ದು?",
    "ಮಕ್ಕಳ ಮೇಲೆ ನಡೆಯುವ ಲೈಂಗಿಕ ದೌರ್ಜನ್ಯವನ್ನು ತಡೆಯಲು ಏನು ಮಾಡಲಾಗಿದೆ?",
    "ಶಾಲೆಗಳಲ್ಲಿ ಮಕ್ಕಳ ರಕ್ಷಣಾ ನೀತಿ ಯಾವುದು?",
    "ಮಕ್ಕಳಿಗೆ ರಕ್ಷಣೆ, ಭದ್ರತೆ ಮತ್ತು ನ್ಯಾಯ ಒದಗಿಸುವ ಉದ್ದೇಶವೇನು?",
]

with open(DATA, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

documents = [r["text"] for r in rows]
ids = [r["chunk_id"] for r in rows]

models = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

all_results = []

for model_name in models:
    print(f"Testing: {model_name}")

    embedder = TextEmbedding(model_name=model_name)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="benchmark_" + model_name.replace("/", "_").replace("-", "_"),
        metadata={"hnsw:space": "cosine"},
    )

    doc_embeddings = [
        list(embedder.embed([doc]))[0].tolist()
        for doc in documents
    ]

    collection.add(
        ids=ids,
        documents=doc_embeddings if False else documents,
        embeddings=doc_embeddings,
    )

    for query in queries:
        query_embedding = list(embedder.embed([query]))[0].tolist()

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
        )

        matches = []

        for rank in range(2):
            chunk_id = result["ids"][0][rank]
            distance = result["distances"][0][rank]

            row = next(r for r in rows if r["chunk_id"] == chunk_id)

            matches.append({
                "rank": rank + 1,
                "chunk_id": chunk_id,
                "page": row["page"],
                "distance": distance,
                "similarity": 1 - distance,
                "text": row["text"],
            })

        all_results.append({
            "model": model_name,
            "query": query,
            "top_k": 2,
            "results": matches,
        })

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUTPUT}")

