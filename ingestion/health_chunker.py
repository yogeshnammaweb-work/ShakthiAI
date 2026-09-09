import json
import re
from pathlib import Path


INPUT_FILE = Path(
    "data/master_dataset/health_kannada/"
    "fssai_yellowbook_kannada_fixed.jsonl"
)

OUTPUT_FILE = Path(
    "data/master_dataset/health_kannada/"
    "fssai_yellowbook_kannada_chunks.jsonl"
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str):
    paragraphs = re.split(r"\n\s*\n", text)
    return [
        p.strip()
        for p in paragraphs
        if len(p.strip()) >= 40
    ]


def create_chunks(text: str):
    paragraphs = split_into_paragraphs(text)

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        candidate = current + "\n" + paragraph

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            chunks.append(current.strip())

            overlap = current[-CHUNK_OVERLAP:]

            # Start the next chunk with a clean boundary
            overlap_start = overlap.find(" ")

            if overlap_start >= 0:
                overlap = overlap[overlap_start + 1:]

            current = overlap + "\n" + paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_pages = 0
    total_chunks = 0

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as source, OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)

            page = record["page"]
            text = normalize_text(record.get("text", ""))

            if not text:
                continue

            total_pages += 1

            chunks = create_chunks(text)

            for chunk_index, chunk in enumerate(chunks):
                chunk_id = (
                    f"fssai_yellowbook_page_"
                    f"{page}_chunk_{chunk_index}"
                )

                output_record = {
                    "chunk_id": chunk_id,
                    "source": record.get(
                        "source",
                        "fssai_yellowbook_kannada",
                    ),
                    "page": page,
                    "chunk_index": chunk_index,
                    "language": "kn",
                    "text": chunk,
                }

                output.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    ) + "\n"
                )

                total_chunks += 1

    print(f"Pages processed: {total_pages}")
    print(f"Health chunks created: {total_chunks}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()