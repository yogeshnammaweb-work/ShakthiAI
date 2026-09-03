import pymupdf
import pytesseract
import io
import json
from PIL import Image

PDF_PATH = r"data\master_dataset\StudentModules.pdf"
OUTPUT_PATH = r"data\master_dataset\StudentModules_ocr.jsonl"

pdf = pymupdf.open(PDF_PATH)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for page_num in range(len(pdf)):
        print(f"OCR page {page_num + 1}/{len(pdf)}...")

        page = pdf[page_num]

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            alpha=False
        )

        img = Image.open(
            io.BytesIO(pix.tobytes("png"))
        )

        text = pytesseract.image_to_string(
            img,
            lang="kan",
            config="--psm 3"
        )

        record = {
            "page": page_num + 1,
            "text": text
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print(f"\nSaved OCR to: {OUTPUT_PATH}")
