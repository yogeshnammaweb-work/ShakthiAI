import pymupdf
import pytesseract
import io
from PIL import Image

pdf = pymupdf.open(r"data\master_dataset\StudentModules.pdf")
pages = [2, 9, 14, 24, 34]

for p in pages:
    pix = pdf[p].get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        alpha=False
    )
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    text = pytesseract.image_to_string(
        img,
        lang="kan",
        config="--psm 3"
    )

    print(f"\nPAGE {p + 1}: {len(text)} chars")
    print(text[:500])
    print("---")
