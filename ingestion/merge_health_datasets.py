from pathlib import Path
import json

INPUT_FILES = [
    Path("data/master_dataset/StudentModules_chunks.jsonl"),
    Path(
        "data/master_dataset/health_kannada/"
        "fssai_yellowbook_kannada_chunks_filtered.jsonl"
    ),
    Path(
        "data/master_dataset/health_kannada/nhm_tb/"
        "tb_kannada_chunks.jsonl"
    ),
]

OUTPUT_FILE = Path(
    "data/master_dataset/health_master_chunks.jsonl"
)


def load_records(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


all_records = []

for input_file in INPUT_FILES:
    records = load_records(input_file)

    print(
        f"{input_file.name}: {len(records)} records"
    )

    all_records.extend(records)


# Generate stable IDs for legacy StudentModules records
# that do not already contain an ID.
for index, record in enumerate(all_records):
    if not record.get("id"):
        record["id"] = f"studentmodules_chunk_{index:04d}"


ids = [record["id"] for record in all_records]

if len(ids) != len(set(ids)):
    duplicates = [
        record_id
        for record_id in set(ids)
        if ids.count(record_id) > 1
    ]

    raise ValueError(
        f"Duplicate IDs found: {duplicates}"
    )


with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for record in all_records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("TOTAL RECORDS:", len(all_records))
print("UNIQUE IDS:", len(set(ids)))
print("OUTPUT:", OUTPUT_FILE)
