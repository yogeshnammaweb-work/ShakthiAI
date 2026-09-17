import json
import shutil
from pathlib import Path

from merge_anemia_iron import ANEMIA_IRON_RECORDS

MASTER_FILE = Path(
    "data/master_dataset/health_master_chunks.jsonl"
)

OLD_MASTER_FILE = Path(
    "data/master_dataset/health_master_chunks_before_anemia_iron.jsonl"
)

BACKUP_FILE = Path(
    "data/master_dataset/health_master_chunks_before_restore.jsonl"
)


RECOVERY_IDS = {
    "blood_hemoglobin_001",
    "blood_plasma_001",
    "blood_platelets_001",
    "blood_rbc_001",
    "blood_wbc_001",
    "pib_kannada_anemia_000",
    "pib_kannada_anemia_001",
    "pib_kannada_anemia_002",
    "studentmodules_chunk_0153",
    "studentmodules_chunk_0154",
    "studentmodules_chunk_0155",
    "studentmodules_chunk_0156",
    "studentmodules_chunk_0157",
}


def load_jsonl(path: Path):
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
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {exc}"
                ) from exc

    return records


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    print("=" * 80)
    print("RESTORING MASTER KNOWLEDGE DATASET")
    print("=" * 80)

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Current master not found: {MASTER_FILE}"
        )

    if not OLD_MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Recovery source not found: {OLD_MASTER_FILE}"
        )

    # ------------------------------------------------------------------
    # Load current corrected master
    # ------------------------------------------------------------------
    current_records = load_jsonl(MASTER_FILE)

    print(
        f"Current master records: {len(current_records)}"
    )

    if len(current_records) != 160:
        raise RuntimeError(
            "Expected corrected master to contain exactly "
            f"160 records, found {len(current_records)}."
        )

    current_ids = {
        record.get("id")
        for record in current_records
    }

    if None in current_ids:
        raise RuntimeError(
            "Current master contains a record without an ID."
        )

    if len(current_ids) != len(current_records):
        raise RuntimeError(
            "Duplicate IDs detected in current master."
        )

    # ------------------------------------------------------------------
    # Load old 173-record master and recover exactly 13 records
    # ------------------------------------------------------------------
    old_records = load_jsonl(OLD_MASTER_FILE)

    print(
        f"Recovery source records: {len(old_records)}"
    )

    if len(old_records) != 173:
        raise RuntimeError(
            "Expected recovery source to contain exactly "
            f"173 records, found {len(old_records)}."
        )

    recovered_records = [
        record
        for record in old_records
        if record.get("id") in RECOVERY_IDS
    ]

    print(
        f"Recovery records found: {len(recovered_records)}"
    )

    if len(recovered_records) != 13:
        found_ids = {
            record.get("id")
            for record in recovered_records
        }

        missing_ids = sorted(
            RECOVERY_IDS - found_ids
        )

        raise RuntimeError(
            "Expected exactly 13 recovery records.\n"
            f"Missing recovery IDs: {missing_ids}"
        )

    recovered_ids = {
        record.get("id")
        for record in recovered_records
    }

    collisions = recovered_ids & current_ids

    if collisions:
        raise RuntimeError(
            "Recovery ID collision detected:\n"
            + "\n".join(sorted(collisions))
        )

    # ------------------------------------------------------------------
    # Load the exact 8 anemia/iron records from the existing script
    # ------------------------------------------------------------------
    anemia_records = list(ANEMIA_IRON_RECORDS)

    print(
        f"Anemia/iron records found: {len(anemia_records)}"
    )

    if len(anemia_records) != 8:
        raise RuntimeError(
            "Expected exactly 8 anemia/iron records, "
            f"found {len(anemia_records)}."
        )

    anemia_ids = {
        record.get("id")
        for record in anemia_records
    }

    if None in anemia_ids:
        raise RuntimeError(
            "Anemia/iron dataset contains a record without an ID."
        )

    if len(anemia_ids) != len(anemia_records):
        raise RuntimeError(
            "Duplicate IDs detected in anemia/iron records."
        )

    # ------------------------------------------------------------------
    # Check all new IDs against current master
    # ------------------------------------------------------------------
    all_new_ids = recovered_ids | anemia_ids

    collisions = all_new_ids & current_ids

    if collisions:
        raise RuntimeError(
            "New-record ID collision with current master:\n"
            + "\n".join(sorted(collisions))
        )

    # ------------------------------------------------------------------
    # Check recovery/anemia overlap
    # ------------------------------------------------------------------
    overlap = recovered_ids & anemia_ids

    if overlap:
        raise RuntimeError(
            "Recovery and anemia/iron datasets contain "
            "duplicate IDs:\n"
            + "\n".join(sorted(overlap))
        )

    # ------------------------------------------------------------------
    # Create backup BEFORE modifying the master
    # ------------------------------------------------------------------
    shutil.copy2(
        MASTER_FILE,
        BACKUP_FILE,
    )

    print(
        f"Backup created: {BACKUP_FILE}"
    )

    # ------------------------------------------------------------------
    # Build final master
    # ------------------------------------------------------------------
    final_records = (
        current_records
        + recovered_records
        + anemia_records
    )

    final_ids = {
        record.get("id")
        for record in final_records
    }

    print(
        f"Final records before write: {len(final_records)}"
    )

    if len(final_records) != 181:
        raise RuntimeError(
            "Expected final master to contain exactly "
            f"181 records, found {len(final_records)}."
        )

    if len(final_ids) != 181:
        raise RuntimeError(
            "Final master contains duplicate IDs."
        )

    # ------------------------------------------------------------------
    # Write final master
    # ------------------------------------------------------------------
    write_jsonl(
        MASTER_FILE,
        final_records,
    )

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------
    verified_records = load_jsonl(MASTER_FILE)

    if len(verified_records) != 181:
        raise RuntimeError(
            "Final verification failed: expected 181 records, "
            f"found {len(verified_records)}."
        )

    verified_ids = {
        record.get("id")
        for record in verified_records
    }

    if len(verified_ids) != 181:
        raise RuntimeError(
            "Final verification failed: duplicate IDs detected."
        )

    # Verify all 13 recovered records
    verified_recovery_ids = {
        record.get("id")
        for record in verified_records
        if record.get("id") in RECOVERY_IDS
    }

    if verified_recovery_ids != RECOVERY_IDS:
        raise RuntimeError(
            "Final verification failed: not all 13 recovery "
            "records are present."
        )

    # Verify all 8 anemia/iron records
    verified_anemia_ids = {
        record.get("id")
        for record in verified_records
        if record.get("id") in anemia_ids
    }

    if verified_anemia_ids != anemia_ids:
        raise RuntimeError(
            "Final verification failed: not all 8 anemia/iron "
            "records are present."
        )

    # ------------------------------------------------------------------
    # Verify corrected FSSAI page 51 remains intact
    # ------------------------------------------------------------------
    page_51_records = [
        record
        for record in verified_records
        if (
            record.get("page") == 51
            and "FSSAI" in record.get("source", "")
        )
    ]

    if len(page_51_records) != 1:
        raise RuntimeError(
            "Expected exactly one FSSAI page 51 record, "
            f"found {len(page_51_records)}."
        )

    page_51_text = page_51_records[0].get(
        "text",
        "",
    )

    iron_position = page_51_text.find(
        "à²•à²¬à³à²¬à²¿à²£à²¦ à²…à²‚à²¶à²¦ à²•à³Šà²°à²¤à³†"
    )

    iodine_position = page_51_text.find(
        "à²…à²¯à³‹à²¡à²¿à²¨à³ à²•à³Šà²°à²¤à³†"
    )

    if (
        iron_position == -1
        or iodine_position == -1
        or iron_position >= iodine_position
    ):
        raise RuntimeError(
            "FSSAI page 51 reading-order verification failed."
        )

    print()
    print("=" * 80)
    print("MASTER DATASET RESTORATION PASSED")
    print("=" * 80)
    print(
        f"Corrected records          : {len(current_records)}"
    )
    print(
        f"Recovered records          : {len(recovered_records)}"
    )
    print(
        f"Anemia/iron records        : {len(anemia_records)}"
    )
    print(
        f"Final records              : {len(verified_records)}"
    )
    print(
        f"Unique IDs                 : {len(verified_ids)}"
    )
    print(
        "FSSAI page 51 order        : VERIFIED"
    )
    print(
        f"Backup                     : {BACKUP_FILE}"
    )
    print(
        f"Master                     : {MASTER_FILE}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
