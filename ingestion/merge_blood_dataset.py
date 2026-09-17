import json
from pathlib import Path

MASTER_PATH = Path(
    "data/master_dataset/health_master_chunks.jsonl"
)

BLOOD_PATH = Path(
    "data/master_dataset/science_kannada/blood_components_kannada.jsonl"
)

BACKUP_PATH = Path(
    "data/master_dataset/health_master_chunks_before_blood.jsonl"
)


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

    return records


print("Loading existing master dataset...")
master_records = load_jsonl(MASTER_PATH)

print(f"EXISTING MASTER RECORDS: {len(master_records)}")

print("Loading blood dataset...")
blood_records = load_jsonl(BLOOD_PATH)

print(f"BLOOD RECORDS: {len(blood_records)}")


master_ids = {
    record.get("id")
    for record in master_records
}

blood_ids = {
    record.get("id")
    for record in blood_records
}

duplicates = master_ids & blood_ids

if duplicates:
    raise RuntimeError(
        f"Duplicate IDs already exist in master dataset: {sorted(duplicates)}"
    )


if len(blood_records) != 5:
    raise RuntimeError(
        f"Expected 5 blood records, found {len(blood_records)}"
    )


# Safety backup before modifying the master dataset.
MASTER_PATH.replace(BACKUP_PATH)

print(f"BACKUP CREATED: {BACKUP_PATH}")


merged_records = master_records + blood_records

with MASTER_PATH.open("w", encoding="utf-8") as f:
    for record in merged_records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("=" * 80)
print("MERGE COMPLETE")
print("=" * 80)
print(f"OLD RECORD COUNT: {len(master_records)}")
print(f"ADDED RECORD COUNT: {len(blood_records)}")
print(f"NEW RECORD COUNT: {len(merged_records)}")
print(f"UNIQUE ID COUNT: {len({r.get('id') for r in merged_records})}")
print(f"MASTER: {MASTER_PATH}")
print(f"BACKUP: {BACKUP_PATH}")

if len(merged_records) != 173:
    raise RuntimeError(
        f"Expected 173 records after merge, got {len(merged_records)}"
    )

if len({r.get("id") for r in merged_records}) != 173:
    raise RuntimeError("Duplicate IDs detected after merge.")

print()
print("MERGE VALIDATION PASSED")