from pathlib import Path
import json

INPUT_FILE = Path(
    "data/master_dataset/health_kannada/nhm_tb/tb_kannada_clean.jsonl"
)

OUTPUT_FILE = Path(
    "data/master_dataset/health_kannada/nhm_tb/tb_kannada_chunks.jsonl"
)

MAX_CHARS = 500


def split_text(text, max_chars=MAX_CHARS):
    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    sentences = [
        s.strip()
        for s in text.replace("\n", " ").split(".")
        if s.strip()
    ]

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence + "."

        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())

            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


records = []
chunk_count = 0

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for record_line in f:
        record = json.loads(record_line)

        chunks = split_text(record["text"])

        for index, chunk in enumerate(chunks):
            chunk_record = {
                "id": f"{record['id']}_chunk_{index:02d}",
                "source": record["source"],
                "language": record["language"],
                "topic": record["topic"],
                "chunk_index": index,
                "text": chunk,
            }

            records.append(chunk_record)
            chunk_count += 1


with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


print("TB SOURCE RECORDS:", len(records))
print("TB CHUNKS:", chunk_count)
print("OUTPUT:", OUTPUT_FILE)

for record in records:
    print(
        f"{record['id']} | "
        f"{record['topic']} | "
        f"{len(record['text'])} chars"
    )
