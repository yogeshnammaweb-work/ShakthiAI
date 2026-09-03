import json

path = r"data\master_dataset\retrieval_benchmark_results.json"

with open(path, encoding="utf-8") as f:
    results = json.load(f)

for item in results:
    print("\n" + "=" * 70)
    print("MODEL:", item["model"])
    print("QUERY:", item["query"])

    for r in item["results"]:
        print(
            f"Rank {r['rank']} | "
            f"{r['chunk_id']} | "
            f"page {r['page']} | "
            f"similarity {r['similarity']:.4f}"
        )
