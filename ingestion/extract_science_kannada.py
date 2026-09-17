import json
from pathlib import Path

from pypdf import PdfReader


INPUT_DIR = Path("data/master_dataset/science_kannada")
OUTPUT_DIR = INPUT_DIR


PDF_FILES = [
    (
        INPUT_DIR / "10th_kannada_science_part1.pdf",
        OUTPUT_DIR / "10th_kannada_science_part1.jsonl",
        "10th Kannada Science Part 1",
    ),
    (
        INPUT_DIR / "10th_kannada_science_part2.pdf",
        OUTPUT_DIR / "10th_kannada_science_part2.jsonl",
        "10th Kannada Science Part 2",
    ),
]


def extract_pdf(pdf_path: Path, output_path: Path, source_name: str):
    print(f"\nReading: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    print(f"Pages: {total_pages}")

    extracted = 0
    total_chars = 0

    with output_path.open("w", encoding="utf-8") as f:
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            record = {
                "id": f"science_{'part1' if 'Part 1' in source_name else 'part2'}_{page_number:03d}",
                "source": source_name,
                "page": page_number,
                "text": text,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            if text:
                extracted += 1
                total_chars += len(text)

            if page_number % 20 == 0 or page_number == total_pages:
                print(
                    f"Processed {page_number}/{total_pages} "
                    f"pages | chars={total_chars}"
                )

    print(f"\nSaved: {output_path}")
    print(f"Non-empty pages: {extracted}/{total_pages}")
    print(f"Total characters: {total_chars}")


def main():
    for pdf_path, output_path, source_name in PDF_FILES:
        extract_pdf(
            pdf_path=pdf_path,
            output_path=output_path,
            source_name=source_name,
        )

    print("\nScience extraction complete.")


if __name__ == "__main__":
    main()