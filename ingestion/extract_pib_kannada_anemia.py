from pathlib import Path
from html.parser import HTMLParser
import json
import re


INPUT_PATH = Path(
    "data/master_dataset/health_kannada/"
    "kannada_anemia/pib_kannada_anemia.html"
)

OUTPUT_PATH = Path(
    "data/master_dataset/health_kannada/"
    "kannada_anemia/pib_kannada_anemia_raw.jsonl"
)


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self):
        return "\n".join(self.parts)


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    html = INPUT_PATH.read_text(encoding="utf-8", errors="replace")

    parser = TextExtractor()
    parser.feed(html)

    text = clean_text(parser.get_text())

    record = {
        "id": "pib_kannada_anemia_001",
        "source": "PIB Kannada",
        "text": text,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    kannada_chars = sum(
        1 for char in text
        if "\u0c80" <= char <= "\u0cff"
    )

    print("Extraction complete.")
    print("Characters:", len(text))
    print("Kannada characters:", kannada_chars)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()