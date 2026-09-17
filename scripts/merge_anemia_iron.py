import json
import shutil
from pathlib import Path


MASTER_FILE = Path(
    "data/master_dataset/health_master_chunks.jsonl"
)

BACKUP_FILE = Path(
    "data/master_dataset/health_master_chunks_before_anemia_iron.jsonl"
)


ANEMIA_IRON_RECORDS = [
    {
        "id": "health_anemia_001",
        "text": "ರಕ್ತಹೀನತೆ ಎಂದರೆ ರಕ್ತದಲ್ಲಿನ ಕೆಂಪು ರಕ್ತಕಣಗಳ ಸಂಖ್ಯೆ ಅಥವಾ ಹಿಮೋಗ್ಲೋಬಿನ್ ಪ್ರಮಾಣ ಸಾಮಾನ್ಯಕ್ಕಿಂತ ಕಡಿಮೆಯಾಗಿರುವ ಸ್ಥಿತಿ. ಹಿಮೋಗ್ಲೋಬಿನ್ ದೇಹದ ಅಂಗಾಂಶಗಳಿಗೆ ಆಮ್ಲಜನಕವನ್ನು ಸಾಗಿಸಲು ಅಗತ್ಯವಾಗಿದೆ.",
        "source": "WHO Anaemia Fact Sheet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/anaemia",
    },
    {
        "id": "health_anemia_002",
        "text": "ರಕ್ತಹೀನತೆಯ ಸಾಮಾನ್ಯ ಲಕ್ಷಣಗಳಲ್ಲಿ ದಣಿವು, ದೌರ್ಬಲ್ಯ, ತಲೆ ಸುತ್ತುವುದು ಅಥವಾ ತಲೆ ಹಗುರವಾಗಿರುವ ಅನುಭವ ಮತ್ತು ಉಸಿರಾಟದ ತೊಂದರೆ ಸೇರಿವೆ. ಈ ಲಕ್ಷಣಗಳಿಂದ ಮಾತ್ರ ರಕ್ತಹೀನತೆ ಇದೆ ಎಂದು ಖಚಿತಪಡಿಸಲಾಗುವುದಿಲ್ಲ; ಆರೋಗ್ಯ ಕಾರ್ಯಕರ್ತರ ಮೌಲ್ಯಮಾಪನ ಅಗತ್ಯವಾಗಬಹುದು.",
        "source": "WHO Anaemia Fact Sheet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/anaemia",
    },
    {
        "id": "health_anemia_003",
        "text": "ರಕ್ತಹೀನತೆಗೆ ಹಲವು ಕಾರಣಗಳಿವೆ. ಪೌಷ್ಟಿಕಾಂಶಗಳ ಕೊರತೆ, ವಿಶೇಷವಾಗಿ ಕಬ್ಬಿಣದ ಕೊರತೆ, ಸೋಂಕುಗಳು, ಉರಿಯೂತ, ದೀರ್ಘಕಾಲದ ಕಾಯಿಲೆಗಳು ಮತ್ತು ಕೆಲವು ಆನುವಂಶಿಕ ರಕ್ತಕಣಗಳ ಕಾಯಿಲೆಗಳು ಕಾರಣವಾಗಬಹುದು.",
        "source": "WHO Anaemia Fact Sheet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/anaemia",
    },
    {
        "id": "health_iron_001",
        "text": "ಕಬ್ಬಿಣದ ಮಾತ್ರೆಗಳನ್ನು ಸೇವಿಸಿದಾಗ ಮಲ ಕಪ್ಪು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಬಹುದು. ಕಬ್ಬಿಣದ ಚಿಕಿತ್ಸೆಯ ಸಂದರ್ಭದಲ್ಲಿ ಮಲ ಕಪ್ಪಾಗುವುದು ಸಾಮಾನ್ಯ ಅಡ್ಡ ಪರಿಣಾಮವಾಗಿರಬಹುದು ಮತ್ತು ಇದರಿಂದ ಮಾತ್ರ ಹೆದರಬೇಕಾಗಿಲ್ಲ.",
        "source": "WHO Iron Deficiency Anaemia guideline",
        "source_url": "https://cdn.who.int/media/docs/default-source/2021-dha-docs/ida_assessment_prevention_control.pdf",
    },
    {
        "id": "health_iron_002",
        "text": "ಕಬ್ಬಿಣದ ಮಾತ್ರೆಗಳಿಂದ ಮಲ ಕಪ್ಪಾಗಬಹುದು ಮತ್ತು ಇದು ಸಾಮಾನ್ಯವಾಗಿ ಹಾನಿಕಾರಕವಲ್ಲ. ವೈದ್ಯರು ಅಥವಾ ಆರೋಗ್ಯ ಕಾರ್ಯಕರ್ತರು ಸೂಚಿಸಿರುವ ಕಬ್ಬಿಣದ ಚಿಕಿತ್ಸೆಯನ್ನು ಕೇವಲ ಮಲ ಕಪ್ಪಾಗಿದೆ ಎಂಬ ಕಾರಣಕ್ಕೆ ಸ್ವಯಂವಾಗಿ ನಿಲ್ಲಿಸಬಾರದು.",
        "source": "WHO Iron Deficiency Anaemia guideline",
        "source_url": "https://cdn.who.int/media/docs/default-source/2021-dha-docs/ida_assessment_prevention_control.pdf",
    },
    {
        "id": "health_iron_003",
        "text": "ಭಾರತದ ಅನೀಮಿಯಾ ಮುಕ್ತ ಭಾರತ ಕಾರ್ಯಕ್ರಮದಲ್ಲಿ ಮಕ್ಕಳ ಮತ್ತು ಕಿಶೋರರಂತಹ ಗುರಿ ಗುಂಪುಗಳಿಗೆ ಕಬ್ಬಿಣ ಮತ್ತು ಫೋಲಿಕ್ ಆಮ್ಲದ ಪೂರಕಗಳನ್ನು ನೀಡಲಾಗುತ್ತದೆ. ಕಬ್ಬಿಣದ ಪೂರಕಗಳ ಪ್ರಮಾಣ ಮತ್ತು ಅವಧಿಯನ್ನು ವಯಸ್ಸು ಹಾಗೂ ಆರೋಗ್ಯ ಸ್ಥಿತಿಗೆ ಅನುಗುಣವಾಗಿ ಆರೋಗ್ಯ ವ್ಯವಸ್ಥೆಯ ಮಾರ್ಗಸೂಚಿಗಳ ಪ್ರಕಾರ ನಿರ್ಧರಿಸಲಾಗುತ್ತದೆ.",
        "source": "National Health Mission - Anaemia Mukt Bharat",
        "source_url": "https://www.nhm.gov.in/digigov-portal/index1.php?lang=1&level=3&lid=797&sublinkid=1448",
    },
    {
        "id": "health_iron_004",
        "text": "ಕಬ್ಬಿಣದ ಪೂರಕಗಳನ್ನು ಸೇವಿಸುವಾಗ ವಾಕರಿಕೆ, ಹೊಟ್ಟೆಯ ಅಸ್ವಸ್ಥತೆ, ಅತಿಸಾರ ಅಥವಾ ಮಲಬದ್ಧತೆ ಮುಂತಾದ ಅಡ್ಡ ಪರಿಣಾಮಗಳು ಕೆಲವರಿಗೆ ಕಾಣಿಸಬಹುದು. ಅಡ್ಡ ಪರಿಣಾಮಗಳು ತೊಂದರೆ ನೀಡಿದರೆ ವೈದ್ಯರು ಅಥವಾ ಆರೋಗ್ಯ ಕಾರ್ಯಕರ್ತರನ್ನು ಸಂಪರ್ಕಿಸಬೇಕು.",
        "source": "WHO Iron Deficiency Anaemia guideline",
        "source_url": "https://cdn.who.int/media/docs/default-source/2021-dha-docs/ida_assessment_prevention_control.pdf",
    },
    {
        "id": "health_iron_005",
        "text": "ರಕ್ತಹೀನತೆಯ ಲಕ್ಷಣಗಳಿದ್ದರೆ ಕಾರಣವನ್ನು ಪತ್ತೆಹಚ್ಚಲು ಆರೋಗ್ಯ ಕಾರ್ಯಕರ್ತರನ್ನು ಸಂಪರ್ಕಿಸುವುದು ಮುಖ್ಯ. ಕೇವಲ ಲಕ್ಷಣಗಳ ಆಧಾರದ ಮೇಲೆ ಕಬ್ಬಿಣದ ಮಾತ್ರೆಗಳನ್ನು ಸ್ವಯಂ ಆರಂಭಿಸುವುದು ಅಥವಾ ನಿಲ್ಲಿಸುವುದು ಸರಿಯಲ್ಲ.",
        "source": "WHO Anaemia Fact Sheet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/anaemia",
    },
]


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    return records


def save_jsonl(path, records):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {MASTER_FILE}"
        )

    print("=" * 70)
    print("SHAKTHI AI - MERGE ANEMIA / IRON DATASET")
    print("=" * 70)

    existing_records = load_jsonl(
        MASTER_FILE
    )

    print(
        f"EXISTING MASTER RECORDS: "
        f"{len(existing_records)}"
    )

    existing_ids = {
        record.get("id")
        for record in existing_records
    }

    duplicate_ids = [
        record["id"]
        for record in ANEMIA_IRON_RECORDS
        if record["id"] in existing_ids
    ]

    if duplicate_ids:
        raise RuntimeError(
            "Duplicate record IDs detected: "
            f"{duplicate_ids}"
        )

    shutil.copy2(
        MASTER_FILE,
        BACKUP_FILE,
    )

    print(
        f"BACKUP CREATED: {BACKUP_FILE}"
    )

    merged_records = (
        existing_records
        + ANEMIA_IRON_RECORDS
    )

    save_jsonl(
        MASTER_FILE,
        merged_records,
    )

    final_records = load_jsonl(
        MASTER_FILE
    )

    final_ids = [
        record.get("id")
        for record in final_records
    ]

    if len(final_records) != (
        len(existing_records)
        + len(ANEMIA_IRON_RECORDS)
    ):
        raise RuntimeError(
            "Final record count validation failed."
        )

    if len(final_ids) != len(set(final_ids)):
        raise RuntimeError(
            "Duplicate IDs detected after merge."
        )

    print(
        f"ANEMIA / IRON RECORDS ADDED: "
        f"{len(ANEMIA_IRON_RECORDS)}"
    )

    print(
        f"FINAL MASTER RECORDS: "
        f"{len(final_records)}"
    )

    print(
        f"UNIQUE ID COUNT: "
        f"{len(set(final_ids))}"
    )

    print()
    print("MERGE VALIDATION PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    main()