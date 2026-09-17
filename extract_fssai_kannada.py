import json
from pathlib import Path

import fitz


INPUT_PDF = Path(
    "data/master_dataset/health_kannada/fssai_yellowbook_kannada.pdf"
)

OUTPUT_JSONL = Path(
    "data/master_dataset/health_kannada/fssai_yellowbook_kannada_reextracted.jsonl"
)


def clean_block_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Ignore standalone page numbers.
        if line.isdigit():
            continue

        # Ignore standalone decorative question marks.
        if line == "?":
            continue

        lines.append(line)

    return "\n".join(lines)


def extract_page_text(page):
    blocks = page.get_text("blocks", sort=True)

    page_parts = []

    for block in blocks:
        text = block[4]

        cleaned = clean_block_text(text)

        if not cleaned:
            continue

        page_parts.append(cleaned)

    return "\n".join(page_parts)


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(INPUT_PDF)
    page_count = len(doc)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as output:
        for page_number, page in enumerate(doc, start=1):
            text = extract_page_text(page)

            record = {
                "page": page_number,
                "text": text,
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    doc.close()

    print(f"Extracted {page_count} pages")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()