import json
from pathlib import Path

INPUT_FILE = Path("data/master_dataset/StudentModules_ocr.jsonl")
OUTPUT_FILE = Path("data/master_dataset/StudentModules_chunks.jsonl")

CHUNK_SIZE = 1200


def chunk_text(text: str):
    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 1 <= CHUNK_SIZE:
            current += "\n" + paragraph
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    chunk_id = 0
    output = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            text = row.get("text", "").strip()
            if not text:
                continue

            page = row.get("page")
            source = row.get("source", "DSERT StudentModules")

            for chunk_index, chunk in enumerate(chunk_text(text)):
                output.append({
                    "chunk_id": f"studentmodules_{chunk_id:04d}",
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index,
                    "text": chunk,
                })
                chunk_id += 1

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Chunks created: {len(output)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
