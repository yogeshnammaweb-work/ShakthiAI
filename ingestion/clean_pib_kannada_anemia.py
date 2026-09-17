import json
import re
from pathlib import Path


INPUT_PATH = Path(
    "data/master_dataset/health_kannada/"
    "kannada_anemia/pib_kannada_anemia_raw.jsonl"
)

OUTPUT_PATH = Path(
    "data/master_dataset/health_kannada/"
    "kannada_anemia/pib_kannada_anemia_clean.jsonl"
)


def normalize(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        record = json.load(f)

    raw = record["text"]

    # The first publication ID is the verified end
    # of the first complete Kannada article.
    end_marker = "(ಪ್ರಕಟಣೆ ಐ.ಡಿ.: 2063604)"

    end = raw.find(end_marker)

    if end == -1:
        raise RuntimeError(
            "First publication ID marker not found."
        )

    first_article = raw[:end]

    # Locate the Kannada article title.
    title_marker = "ಪ್ರಧಾನಮಂತ್ರಿ ಗರೀಬ್ ಕಲ್ಯಾಣ್"

    start = first_article.rfind(
        title_marker,
        0,
        end,
    )

    if start == -1:
        raise RuntimeError(
            "Kannada article title not found."
        )

    # Move to the beginning of the title line.
    line_start = first_article.rfind(
        "\n",
        0,
        start,
    )

    if line_start == -1:
        line_start = 0
    else:
        line_start += 1

    article = first_article[line_start:]

    # Remove obvious page/navigation noise.
    lines = []

    for line in article.splitlines():
        line = normalize(line)

        if not line:
            continue

        if line.startswith("ವಿಸಿಟರ್ ಕೌಂಟರ್"):
            continue

        if line.startswith("ಪ್ರಕಟಣೆಯನ್ನು ಇದರಲ್ಲಿ ಓದಿ"):
            continue

        lines.append(line)

    # Remove consecutive duplicate lines only.
    cleaned_lines = []

    for line in lines:
        if not cleaned_lines or line != cleaned_lines[-1]:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    output = {
        "id": "pib_kannada_anemia_001",
        "source": "PIB Kannada",
        "text": text,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                output,
                ensure_ascii=False,
            )
            + "\n"
        )

    kannada_chars = len(
        re.findall(
            r"[\u0C80-\u0CFF]",
            text,
        )
    )

    print("Cleaning complete.")
    print("Article boundary:", end)
    print("Characters:", len(text))
    print("Kannada characters:", kannada_chars)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
