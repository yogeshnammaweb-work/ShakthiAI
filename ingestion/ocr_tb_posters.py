from pathlib import Path
import json
import pytesseract
from PIL import Image

INPUT_DIR = Path("data/master_dataset/health_kannada/nhm_tb")
OUTPUT_FILE = Path("data/master_dataset/health_kannada/nhm_tb/tb_posters_ocr.jsonl")

files = sorted(INPUT_DIR.rglob("*.jpg"))

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for i, image_path in enumerate(files, start=1):
        print(f"OCR {i}/{len(files)}: {image_path.name}")

        image = Image.open(image_path)

        text = pytesseract.image_to_string(
            image,
            lang="kan",
            config="--psm 6",
        ).strip()

        record = {
            "id": f"tb_kannada_poster_{i:02d}",
            "source": "Central TB Division Kannada Poster",
            "language": "kn",
            "file": image_path.name,
            "text": text,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nDONE: {len(files)} posters")
print(f"OUTPUT: {OUTPUT_FILE}")
