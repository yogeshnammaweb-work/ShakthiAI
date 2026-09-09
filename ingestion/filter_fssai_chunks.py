import json
from pathlib import Path

INPUT_FILE = Path(
    "data/master_dataset/health_kannada/"
    "fssai_yellowbook_kannada_chunks.jsonl"
)

OUTPUT_FILE = Path(
    "data/master_dataset/health_kannada/"
    "fssai_yellowbook_kannada_chunks_filtered.jsonl"
)

# These pages are module covers, heading-only pages,
# assessment-tool pages, or a certificate page.
EXCLUDED_PAGES = {
    9,
    19,
    31,
    71,
    72,
    86,
    101,
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    kept_chunks = 0
    removed_chunks = 0

    with (
        INPUT_FILE.open("r", encoding="utf-8") as source,
        OUTPUT_FILE.open("w", encoding="utf-8") as output,
    ):
        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)
            total_chunks += 1

            page = record.get("page")

            if page in EXCLUDED_PAGES:
                removed_chunks += 1
                continue

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            kept_chunks += 1

    print(f"Total chunks: {total_chunks}")
    print(f"Kept chunks: {kept_chunks}")
    print(f"Removed chunks: {removed_chunks}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()