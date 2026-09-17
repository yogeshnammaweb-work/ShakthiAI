import json
from pathlib import Path

INPUT_FILE = Path("data/master_dataset/health_kannada/pib_wbc_chunks_fixed.jsonl")
OUTPUT_FILE = Path("data/master_dataset/health_kannada/pib_wbc_chunks_clean.jsonl")


def clean_text(text: str) -> str:
    text = text.replace("ಪ್ರತಿಕಾಯಗಳನ್ನುಉತ್ಪಾದಿಸುತ್ತವೆ", "ಪ್ರತಿಕಾಯಗಳನ್ನು ಉತ್ಪಾದಿಸುತ್ತವೆ")
    text = text.replace("ಪ್ರತಿಕಾಯಗಳನ್ನುಉತ್ಪಾದಿಸ", "ಪ್ರತಿಕಾಯಗಳನ್ನು ಉತ್ಪಾದಿಸ")
    return text


def main():
    output = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["text"] = clean_text(row.get("text", ""))
            output.append(row)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Clean records: {len(output)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()