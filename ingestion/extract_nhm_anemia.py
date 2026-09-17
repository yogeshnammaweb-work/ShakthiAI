import json
from pathlib import Path
import pymupdf


PDF_PATH = Path(
    "data/master_dataset/health_kannada/nhm_anemia/technical_handbook_on_anemia.pdf"
)

OUTPUT_PATH = Path(
    "data/master_dataset/health_kannada/nhm_anemia/nhm_anemia_raw.jsonl"
)


def main():
    doc = pymupdf.open(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        record = {
            "id": f"nhm_anemia_page_{page_number}",
            "source": "NHM Technical Handbook on Anaemia",
            "page": page_number,
            "text": text,
        }

        records.append(record)

    doc.close()

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    non_empty = sum(
        1 for record in records if record["text"].strip()
    )

    total_chars = sum(
        len(record["text"])
        for record in records
    )

    print("Extraction complete.")
    print("Pages:", len(records))
    print("Pages with text:", non_empty)
    print("Total characters:", total_chars)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()