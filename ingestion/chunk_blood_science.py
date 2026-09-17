import json
from pathlib import Path

INPUT_FILE = Path("data/master_dataset/science_kannada/science_part1_biology_ocr.jsonl")
OUTPUT_FILE = Path("data/master_dataset/science_kannada/blood_chunks.jsonl")

TARGET_PAGES = {98, 99, 100, 101}
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
    output = []
    chunk_id = 0

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row.get("page") not in TARGET_PAGES:
                continue

            text = row.get("text", "").strip()

            if not text:
                continue

            for chunk_index, chunk in enumerate(chunk_text(text)):
                output.append({
                    "chunk_id": f"science_blood_{chunk_id:04d}",
                    "source": "Karnataka 10th Standard Science Part 1 - Biology",
                    "page": row.get("page"),
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