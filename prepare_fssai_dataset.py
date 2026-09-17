import json
import re
from pathlib import Path


INPUT_FILES = [
    Path("data/master_dataset/StudentModules_chunks.jsonl"),
    Path(
        "data/master_dataset/health_kannada/"
        "fssai_yellowbook_kannada_prepared.jsonl"
    ),
    Path(
        "data/master_dataset/health_kannada/nhm_tb/"
        "tb_kannada_chunks.jsonl"
    ),
]
OUTPUT = Path(
    "data/master_dataset/health_kannada/"
    "fssai_yellowbook_kannada_prepared.jsonl"
)


def clean_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r", "\n")

    # Remove excessive blank lines
    text = re.sub(r"\n[ \t]*\n+", "\n", text)

    # Remove repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Common OCR corrections observed in this PDF
    replacements = {
        "ಯಾವುದಲೇ": "ಯಾವುದೇ",
        "ಹಲೊಂದಿರುವ": "ಹೊಂದಿರುವ",
        "ಸಲೇರಿಕಲೊಂಡರೆ": "ಸೇರಿಕೊಂಡರೆ",
        "ಕಲೊಂಡ": "ಕೊಂಡ",
        "ಕಲೊಳ್ಳ": "ಕೊಳ್ಳ",
        "ಕಲೊಳ್ಳಿ": "ಕೊಳ್ಳಿ",
        "ತೊಳೆದುಕಲೊ": "ತೊಳೆದುಕೊಳ್ಳಿ",
        "ತೊಳೆದುಕಲೊಳ್ಳಿ": "ತೊಳೆದುಕೊಳ್ಳಿ",
        "ಸ್ವಚ್ಛವಾಗಿಟ್ಟುಕಲೊಳ್ಳ": "ಸ್ವಚ್ಛವಾಗಿಟ್ಟುಕೊಳ್ಳ",
        "ಉಜ್ಜಿಕಲೊಳ್ಳಿ": "ಉಜ್ಜಿಕೊಳ್ಳಿ",
        "ಒಣಗಿಸಿಕಲೊಳ್ಳಿ": "ಒಣಗಿಸಿಕೊಳ್ಳಿ",
        "ಹಲೊತ್ತು": "ಹೊತ್ತು",
        "ಬೇರಲೊಬ್ಬರ": "ಬೇರೊಬ್ಬರ",
        "ರಲೋಗಾಣು": "ರೋಗಾಣು",
        "ರಲೋಗಗಳು": "ರೋಗಗಳು",
        "ರಲೋಗವನ್ನು": "ರೋಗವನ್ನು",
        "ರಲೋಗಗಳಿಗೆ": "ರೋಗಗಳಿಗೆ",
        "ರಲೋಗಾಣುಗಳನ್ನು": "ರೋಗಾಣುಗಳನ್ನು",
        "ರlೋಗ": "ರೋಗ",
        "ಕlೈ": "ಕೈ",
        "ದlೇಹ": "ದೇಹ",
        "ವlೈಯಕ್ತಿಕ": "ವೈಯಕ್ತಿಕ",
        "ಆರlೋ�ಗ್ಯ": "ಆರೋಗ್ಯ",
        "ಪlೋಷಣ": "ಪೋಷಣ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove obvious replacement-character OCR artifacts
    text = text.replace("�", "")

    # Clean spaces around punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def is_useful_page(text: str) -> bool:
    """
    Keep pages containing meaningful Kannada educational/health content.
    """
    if not text:
        return False

    # Remove very short pages such as covers or empty pages
    if len(text.strip()) < 150:
        return False

    # Count Kannada Unicode characters
    kannada_count = len(
        re.findall(r"[\u0C80-\u0CFF]", text)
    )

    return kannada_count >= 30


def detect_topic(text: str) -> str:
    if any(
        x in text
        for x in ["ರೋಗಾಣು", "ಸೂಕ್ಷ್ಮಜೀವಿ", "ಸೋಂಕು"]
    ):
        return "ರೋಗಾಣುಗಳು ಮತ್ತು ಸೋಂಕು"

    if any(
        x in text
        for x in [
            "ಕೈಗಳನ್ನು",
            "ಕೈ ತೊಳೆಯ",
            "ಸ್ವಚ್ಛತೆ",
            "ನೈರ್ಮಲ್ಯ",
        ]
    ):
        return "ವೈಯಕ್ತಿಕ ಸ್ವಚ್ಛತೆ ಮತ್ತು ನೈರ್ಮಲ್ಯ"

    if any(
        x in text
        for x in ["ಪೌಷ್ಟಿಕ", "ಪೋಷಣ", "ಆಹಾರ"]
    ):
        return "ಆಹಾರ ಮತ್ತು ಪೌಷ್ಟಿಕತೆ"

    if any(
        x in text
        for x in ["ಆರೋಗ್ಯ", "ಆರೋಗ್ಯಕರ"]
    ):
        return "ಆರೋಗ್ಯ"

    if any(
        x in text
        for x in ["ನೀರು", "ಕುಡಿಯುವ ನೀರು"]
    ):
        return "ನೀರು ಮತ್ತು ಸುರಕ್ಷತೆ"

    return "ಆರೋಗ್ಯ ಮತ್ತು ಆಹಾರ ಸುರಕ್ಷತೆ"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            page = row.get("page")
            original_text = row.get("text", "")

            text = clean_text(original_text)

            if not is_useful_page(text):
                continue

            rows.append(
                {
                    "source": (
                        "FSSAI Yellow Book – "
                        "Safe and Nutritious Food"
                    ),
                    "language": "kn",
                    "page": page,
                    "topic": detect_topic(text),
                    "text": text,
                }
            )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as f:
        input_count = sum(
            1 for _ in f
        )

    print("Input pages:", input_count)
    print("Prepared pages:", len(rows))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()