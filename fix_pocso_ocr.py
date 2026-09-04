from pathlib import Path

p = Path("data/master_dataset/StudentModules_ocr.jsonl")
s = p.read_text(encoding="utf-8")

replacements = {
    "7೦೦೨5೦": "POCSO",
    "೧೦೮೦5೦": "POCSO",
    "೯೦೦5೦": "POCSO",
    "7೦೦9೦": "POCSO",
    "೧೦೦೨5೦": "POCSO",
    "೧೦೦೦೦": "POCSO",
}

for old, new in replacements.items():
    count = s.count(old)
    print(f"{old} -> {new}: {count}")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("POCSO OCR cleanup complete.")
