import json
import re
import unicodedata
from pathlib import Path


INPUT_FILE = Path(
    "data/master_dataset/health_kannada/fssai_yellowbook_kannada.jsonl"
)

OUTPUT_FILE = Path(
    "data/master_dataset/health_kannada/fssai_yellowbook_kannada_clean.jsonl"
)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")

    # Remove common PDF extraction replacement characters.
    text = text.replace("�", "")

    # Fix the recurring extraction artifact where "l" appears
    # inside Kannada words before vowel signs / letters.
    replacements = {
        "lೇ": "ಲೇ",
        "lೆ": "ಲೆ",
        "lೋ": "ಲೋ",
        "lೊ": "ಲೊ",
        "lೌ": "ಲೌ",
        "lೈ": "ಲೈ",
        "lಾ": "ಲಾ",
        "lಿ": "ಲಿ",
        "lೀ": "ಲೀ",
        "lು": "ಲು",
        "lೂ": "ಲೂ",
        "lೃ": "ಲೃ",
        "lಂ": "ಲಂ",
        "lಃ": "ಲಃ",
        "l್": "ಲ್",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Common standalone artifact.
    text = re.sub(r"\bl\b", "", text)

    # Remove excessive spaces while preserving paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)

    # Clean spaces around line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def should_keep_page(page: int, text: str) -> bool:
    """
    Remove obvious front-matter pages.
    Keep the actual educational/health content.
    """

    # Pages 1-8 are cover/front matter/credits/contents.
    if page <= 8:
        return False

    if not text.strip():
        return False

    return True


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with INPUT_FILE.open("r", encoding="utf-8") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)

            page = int(record["page"])
            text = clean_text(record.get("text", ""))

            total += 1

            if not should_keep_page(page, text):
                continue

            cleaned_record = {
                "page": page,
                "text": text,
                "source": "fssai_yellowbook_kannada",
            }

            output.write(
                json.dumps(
                    cleaned_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            kept += 1

    print(f"Input pages: {total}")
    print(f"Pages kept: {kept}")
    print(f"Pages removed: {total - kept}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()