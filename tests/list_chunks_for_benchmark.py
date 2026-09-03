import json

input_path = r"data\master_dataset\StudentModules_chunks.jsonl"
output_path = r"data\master_dataset\chunk_preview.txt"

with open(input_path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

with open(output_path, "w", encoding="utf-8") as out:
    for r in rows:
        out.write(f'[{r["chunk_id"]}] PAGE {r["page"]}\n')
        out.write(r["text"][:350].replace("\n", " "))
        out.write("\n\n")

print("Preview created:", output_path)
print("Chunks:", len(rows))
