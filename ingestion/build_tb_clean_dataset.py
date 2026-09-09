from pathlib import Path
import json

OUTPUT_FILE = Path(
    "data/master_dataset/health_kannada/nhm_tb/tb_kannada_clean.jsonl"
)

records = [
    {
        "id": "tb_clean_001",
        "source": "Central TB Division Kannada Poster",
        "language": "kn",
        "topic": "DOTS treatment completion",
        "text": (
            "ಯಾವುದೇ ಸ್ಥಿತಿಯಲ್ಲೂ ಡಾಟ್ಸ್ ಕೋರ್ಸನ್ನು ಅರ್ಧದಲ್ಲೇ ನಿಲ್ಲಿಸಬೇಡಿ. "
            "ಹಾಗೆ ಮಾಡಿದಲ್ಲಿ ಕ್ಷಯ ರೋಗಕ್ಕೆ ಚಿಕಿತ್ಸೆ ಪಡೆಯುವುದು ಕಷ್ಟವಾಗಿರುತ್ತದೆ."
        ),
    },
    {
        "id": "tb_clean_002",
        "source": "Central TB Division Kannada Poster",
        "language": "kn",
        "topic": "TB prevention",
        "text": (
            "ಕೆಮ್ಮುವಾಗ ಯಾವಾಗಲೂ ಮುಖವನ್ನು ಬಟ್ಟೆಯಿಂದ ಕರವಸ್ತ್ರದಿಂದ ಮುಚ್ಚಿಕೊಳ್ಳಿ. "
            "ಕೆಮ್ಮುವಾಗ ಮತ್ತು ಸೀನುವಾಗ ರೋಗಾಣುಗಳು ಹರಡುತ್ತವೆ."
        ),
    },
    {
        "id": "tb_clean_003",
        "source": "Central TB Division Kannada Poster",
        "language": "kn",
        "topic": "TB testing",
        "text": (
            "ಕುಟುಂಬದ ಯಾವುದೇ ವ್ಯಕ್ತಿಗೆ 2 ವಾರಕ್ಕಿಂತ ಹೆಚ್ಚು ಕಾಲ ಕೆಮ್ಮು ಇದ್ದಲ್ಲಿ, "
            "ಸಮೀಪದ ಡಾಟ್ಸ್ ಕೇಂದ್ರದಲ್ಲಿ ಅವರ ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ."
        ),
    },
    {
        "id": "tb_clean_004",
        "source": "Central TB Division Kannada Poster",
        "language": "kn",
        "topic": "Drug-resistant TB",
        "text": (
            "ಡಾಟ್ಸ್ ಕೋರ್ಸನ್ನು ಅರ್ಧದಲ್ಲಿಯೇ ಕೈ ಬಿಟ್ಟಲ್ಲಿ, "
            "ಡಾಟ್ಸ್ ರೋಗ ಹದಗೆಡಬಹುದು ಮತ್ತು ಎಮ್‌ಡಿಆರ್ ಆಗಬಹುದು."
        ),
    },
    {
        "id": "tb_clean_005",
        "source": "Central TB Division Kannada Poster",
        "language": "kn",
        "topic": "TB treatment duration",
        "text": (
            "ಇದಕ್ಕಾಗಿ 24 ರಿಂದ 27 ತಿಂಗಳುಗಳ ವರೆಗೆ ಚಿಕಿತ್ಸೆಯನ್ನು ಪಡೆಯಬಹುದಾಗಿದೆ."
        ),
    },
    {
        "id": "tb_clean_006",
        "source": "Central TB Division Kannada Poster",
        "language": "kn",
        "topic": "TB transmission",
        "text": (
            "ಡಾಟ್ಸ್ ಚಿಕಿತ್ಸೆ ಪ್ರಾರಂಭವಾಗುತ್ತಲೇ, "
            "ಕ್ಷಯ ರೋಗ ಹರಡುವ ಸಂಭವ ಕಡಿಮೆಯಾಗುತ್ತದೆ."
        ),
    },
    {
        "id": "tb_clean_007",
        "source": "Central TB Division Kannada Poster",
        "language": "kn",
        "topic": "TB treatment availability",
        "text": (
            "ಭಾರತ ಸರ್ಕಾರದ ಮೂಲಕ ಒದಗಿಸಲಾಗುವ ಕ್ಷಯ ರೋಗ ಚಿಕಿತ್ಸೆ ಮತ್ತು "
            "ಉನ್ನತ ಗುಣಮಟ್ಟದ ಔಷಧಿಗಳು ಎಲ್ಲಾ ಡಾಟ್ಸ್ ಕೇಂದ್ರಗಳಲ್ಲೂ "
            "ಉಚಿತವಾಗಿ ಲಭ್ಯವಿರುತ್ತದೆ."
        ),
    },
]

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

print("TB CLEAN RECORDS:", len(records))
print("OUTPUT:", OUTPUT_FILE)

for record in records:
    print(
        f"{record['id']} | "
        f"{record['topic']} | "
        f"{len(record['text'])} chars"
    )
