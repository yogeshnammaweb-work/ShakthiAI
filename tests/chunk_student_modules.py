import json
import re

INPUT = r"data\master_dataset\StudentModules_ocr.jsonl"
OUTPUT = r"data\master_dataset\StudentModules_chunks.jsonl"


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, size=900, overlap=150):
    text = clean_text(text)

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)

            if len(paragraph) <= size:
                current = paragraph
            else:
                start = 0

                while start < len(paragraph):
                    end = min(start + size, len(paragraph))

                    if end < len(paragraph):
                        boundary = paragraph.rfind(" ", start, end)

                        if boundary > start + 400:
                            end = boundary

                    piece = paragraph[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    if end >= len(paragraph):
                        current = ""
                        break

                    start = max(end - overlap, start + 1)

    if current:
        chunks.append(current)

    return chunks


with open(INPUT, encoding="utf-8") as f:
    records = [json.loads(line) for line in f]


chunk_id = 0

with open(OUTPUT, "w", encoding="utf-8") as f:
    for record in records:
        page = record["page"]

        for index, chunk in enumerate(chunk_text(record["text"])):
            chunk_id += 1

            item = {
                "chunk_id": f"studentmodules_{chunk_id:04d}",
                "page": page,
                "chunk_index": index,
                "text": chunk,
            }

            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )


print("Created", chunk_id, "chunks")
print("Saved to:", OUTPUT)