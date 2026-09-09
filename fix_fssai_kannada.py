import json
import re
import unicodedata
from pathlib import Path


INPUT_FILE = Path(
    "data/master_dataset/health_kannada/fssai_yellowbook_kannada_clean.jsonl"
)

OUTPUT_FILE = Path(
    "data/master_dataset/health_kannada/fssai_yellowbook_kannada_fixed.jsonl"
)


# PDF text extraction repeatedly inserts an extra "ಲ" into
# Kannada words. These are conservative corrections based on
# patterns observed in this specific FSSAI document.
REPLACEMENTS = {
    "ರಲೋಗ": "ರೋಗ",
    "ರಲೋಗಾಣು": "ರೋಗಾಣು",
    "ಕಲೈ": "ಕೈ",
    "ಕಲೈಸ": "ಕೈಸ",
    "ಕಲೊಳ": "ಕೊಳ",
    "ಕಲೊಂಡ": "ಕೊಂಡ",
    "ಕಲೊಳ್ಳ": "ಕೊಳ್ಳ",
    "ಕಲೊಳ್ಳುವ": "ಕೊಳ್ಳುವ",
    "ಕಲೊಳ್ಳಿ": "ಕೊಳ್ಳಿ",
    "ಕಲೊಳ್ಳಬೇಕು": "ಕೊಳ್ಳಬೇಕು",
    "ತಲೊಳೆ": "ತೊಳೆ",
    "ತಲೊಳ": "ತೊಳ",
    "ತಲೊ": "ತೊ",
    "ನಲೋ": "ನೋ",
    "ಹಲೊಂದ": "ಹೊಂದ",
    "ಹಲೇಳ": "ಹೇಳ",
    "ಹಲೇಗೆ": "ಹೇಗೆ",
    "ಬಲೇಕ": "ಬೇಕ",
    "ಬಲೇರ": "ಬೇರ",
    "ವಲೈಯಕ್ತಿಕ": "ವೈಯಕ್ತಿಕ",
    "ದಲೇಹ": "ದೇಹ",
    "ಸಲೇರ": "ಸೇರ",
    "ಕಲೊಂಡ": "ಕೊಂಡ",
    "ಕಲೊಂಡು": "ಕೊಂಡು",
    "ಮಲಾಡ": "ಮಾಡ",
    "ಮಲಾಡುವ": "ಮಾಡುವ",
    "ಮಲಾಡಿ": "ಮಾಡಿ",
    "ಮಲಾಡಬೇಕು": "ಮಾಡಬೇಕು",
    "ಗಲೊಳ": "ಗೊಳ",
    "ಗಲೊಳ್ಳ": "ಗೊಳ್ಳ",
    "ಅಗಲೋಚರ": "ಅಗೋಚರ",
    "ಆರಲೋಗ್ಯ": "ಆರೋಗ್ಯ",
    "ಆರಲೋಗ": "ಆರೋಗ",
    "ಲೈಂಗಿಕ": "ಲೈಂಗಿಕ",
    "ಶಲೋಷ": "ಶೋಷ",
    "ಸಲಂಪರ್ಕ": "ಸಂಪರ್ಕ",
    "ಸಲಂದರ್ಭ": "ಸಂದರ್ಭ",
    "ಕಲೊನೆ": "ಕೊನೆ",
    "ಕಲೊನೆಯಲ್ಲಿ": "ಕೊನೆಯಲ್ಲಿ",
    "ಕಲೊಡ": "ಕೂಡ",
    "ಕಲೊಡಿ": "ಕೂಡಿ",
    "ಕಲೊಡಲೇ": "ಕೂಡಲೇ",
    "ಗಲೊತ್ತ": "ಗೊತ್ತ",
    "ಗಲೊತ್ತೆ": "ಗೊತ್ತೆ",
    "ತಲೋರ": "ತೋರ",
    "ತಲೋರಿಸ": "ತೋರಿಸ",
    "ವಿನಲೋದ": "ವಿನೋದ",
    "ಕಲಾರಣ": "ಕಾರಣ",
    "ಕಲಾಯಿಲೆ": "ಕಾಯಿಲೆ",
    "ಖಲಾಯಿಲೆ": "ಖಾಯಿಲೆ",
    "ಒಳಗಲೊಂಡ": "ಒಳಗೊಂಡ",
    "ಕಲುಶಿತಗಲೊಂಡ": "ಕಲುಶಿತಗೊಂಡ",
    "ನೀರಿನಲೊಂದಿಗೆ": "ನೀರಿನೊಂದಿಗೆ",
    "ಬೇರಲೊಬ್ಬ": "ಬೇರೊಬ್ಬ",
    "ಹಲೊಳಪಿನ": "ಹೊಳೆಪಿನ",
    "ತಲೈಲ": "ತೈಲ",
    "ತೊಳೆದುಕಲೊ": "ತೊಳೆದುಕೊ",
    "ತಲೊಳೆಯ": "ತೊಳೆಯ",
    "ತಲೊಳೆಯಬೇಕು": "ತೊಳೆಯಬೇಕು",
    "ಕಲೇವಲ": "ಕೇವಲ",
    "ಹಲೊತ್ತು": "ಹೊತ್ತು",
    "ಹಲೇಳಿರಿ": "ಹೇಳಿರಿ",
    "ನಲೊಂದಿಗೆ": "ನೊಂದಿಗೆ",
    "ಕಲೈಗಳನ್ನು": "ಕೈಗಳನ್ನು",
    "ಕಲೈಗಳಿಂದ": "ಕೈಗಳಿಂದ",
    "ಕಲೈಗಳಲ್ಲಿ": "ಕೈಗಳಲ್ಲಿ",
}


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")

    # Remove replacement characters left by PDF encoding.
    text = text.replace("�", "")

    # Apply longest replacements first.
    for old, new in sorted(
        REPLACEMENTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = text.replace(old, new)

    # Clean remaining extraction noise.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    changed = 0

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

            original_text = record.get("text", "")
            fixed_text = clean_text(original_text)

            total += 1

            if fixed_text != original_text:
                changed += 1

            output_record = {
                "page": record["page"],
                "text": fixed_text,
                "source": "fssai_yellowbook_kannada",
            }

            output.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Pages processed: {total}")
    print(f"Pages changed: {changed}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()