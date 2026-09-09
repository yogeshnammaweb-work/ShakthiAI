from pathlib import Path
import json
import re

INPUT_FILE = Path(
    "data/master_dataset/health_kannada/nhm_tb/tb_posters_ocr.jsonl"
)

OUTPUT_FILE = Path(
    "data/master_dataset/health_kannada/nhm_tb/tb_posters_clean.jsonl"
)

# Poster 2/3 and Poster 4-5/6 contain essentially the same
# OCR-detected messages. Keep one representative from each group.
KEEP_FILES = {
    "Artwork Poster-1.jpg",
    "Artwork Poster-2.jpg",
    "Artwork Poster-4-5.jpg",
    "Artwork Poster-7-8.jpg",
}

# Conservative OCR corrections where the intended Kannada wording
# is clear from the extracted text.
REPLACEMENTS = {
    "ಕ್ಸಯ": "ಕ್ಷಯ",
    "ಡಾಟ್ಸ್‌": "ಡಾಟ್ಸ್",
    "ಕೋರ್ಸನ್ನು": "ಕೋರ್ಸನ್ನು",
    "ಅರ್ಧದಲ್ಲೇ": "ಅರ್ಧದಲ್ಲೇ",
    "ಅರ್ಧದಲ್ಲಿಯೇ": "ಅರ್ಧದಲ್ಲಿಯೇ",
    "ಕೆಮ್ಮುಇದ್ದಲ್ಲಿ": "ಕೆಮ್ಮು ಇದ್ದಲ್ಲಿ",
    "ರೋಗಿಗಳಿಗೆ ತೊರಿಸಿ": "ರೋಗಿಗಳಿಗೆ ತೋರಿಸಿ",
    "ಡಾಟ್ಟ್‌": "ಡಾಟ್ಸ್",
    "ಡಾಟ್ಸ್‌ ಚಿಕಿತ್ಸೆ": "ಡಾಟ್ಸ್ ಚಿಕಿತ್ಸೆ",
    "ಚಕಿತ್ಸ": "ಚಿಕಿತ್ಸೆ",
    "ಪ್ರಾರಂ": "ಪ್ರಾರಂಭ",
    "ತ್ತಲೇ": "ತ್ತಲೇ",
}

# Lines dominated by OCR noise: Latin/digit-heavy strings,
# isolated symbols, or extremely short fragments.
def is_noise(line):
    line = line.strip()

    if not line:
        return True

    kannada_chars = len(re.findall(r"[\u0C80-\u0CFF]", line))
    latin_chars = len(re.findall(r"[A-Za-z]", line))
    digit_chars = len(re.findall(r"[0-9೦-೯]", line))

    if kannada_chars == 0:
        return True

    if len(line) <= 3 and kannada_chars < 2:
        return True

    if latin_chars + digit_chars > kannada_chars * 2:
        return True

    return False


def clean_text(text):
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if is_noise(line):
            continue

        lines.append(line)

    # Remove consecutive duplicate lines produced by OCR.
    cleaned = []
    for line in lines:
        if not cleaned or line != cleaned[-1]:
            cleaned.append(line)

    return "\n".join(cleaned).strip()


records = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        if record["file"] not in KEEP_FILES:
            continue

        cleaned = clean_text(record["text"])

        if len(cleaned) < 50:
            print("SKIP SHORT:", record["file"])
            continue

        record["text"] = cleaned
        record["source"] = "Central TB Division Kannada Poster"
        record["language"] = "kn"

        records.append(record)


with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

print("INPUT RECORDS: 6")
print("KEPT RECORDS:", len(records))
print("OUTPUT:", OUTPUT_FILE)

for record in records:
    print(
        f"\n{record['file']}: "
        f"{len(record['text'])} chars"
    )
    print(record["text"])
