import json

path = r"data\master_dataset\StudentModules_chunks.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("Chunks:", len(rows))
print()

for r in rows[:5]:
    print("---", r["chunk_id"], "| page", r["page"], "---")
    print(r["text"][:700])
    print()
