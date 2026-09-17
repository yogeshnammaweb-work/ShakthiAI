import json
from pathlib import Path


INPUT_PATH = Path(
    "data/master_dataset/health_kannada/"
    "kannada_anemia/pib_kannada_anemia_clean.jsonl"
)

OUTPUT_PATH = Path(
    "data/master_dataset/health_kannada/"
    "kannada_anemia/pib_kannada_anemia_chunks.jsonl"
)

CHUNK_SIZE = 900
OVERLAP = 150


def clean_text(text):
    text = text.replace("*****", "")
    return text.strip()


def chunk_text(text):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(
            start + CHUNK_SIZE,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - OVERLAP

    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    output_records = []

    for record in records:
        chunks = chunk_text(record["text"])

        for index, chunk in enumerate(chunks):
            output_records.append(
                {
                    "id": (
                        f"pib_kannada_anemia_"
                        f"{index:03d}"
                    ),
                    "source": record.get(
                        "source",
                        "PIB Kannada",
                    ),
                    "page": None,
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in output_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Chunking complete.")
    print("Input records:", len(records))
    print("Chunks:", len(output_records))
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap:", OVERLAP)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
