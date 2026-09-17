import json
from pathlib import Path

INPUT_FILE = Path("data/master_dataset/health_kannada/pib_wbc_chunks.jsonl")
OUTPUT_FILE = Path("data/master_dataset/health_kannada/pib_wbc_chunks_fixed.jsonl")


def repair_text(value):
    if not isinstance(value, str):
        return value

    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def main():
    output = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            row["source"] = repair_text(row.get("source", ""))
            row["text"] = repair_text(row.get("text", ""))

            output.append(row)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Fixed records: {len(output)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()