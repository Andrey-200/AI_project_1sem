from pathlib import Path
import io

from PIL import Image
import easyocr
from striprtf.striprtf import rtf_to_text

def run_llm_extraction(path: Path):
    if path.suffix.lower() == ".rtf":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = rtf_to_text(raw)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    words = text.split()
    spans = []
    cursor = 0
    for i, w in enumerate(words):
        start = text.find(w, cursor)
        end = start + len(w)
        spans.append([start, end, w, i])
        cursor = end

    return {"text": text, "spans": spans}

reader = easyocr.Reader(["ru", "en"], gpu=False)

def run_ocr(path: Path):
    results = reader.readtext(str(path))

    text_parts = []
    spans = []
    cursor = 0

    for bbox, word, conf in results:
        word = word.strip()
        if not word:
            continue

        start = cursor
        end = cursor + len(word)
        text_parts.append(word)

        spans.append([start, end, word, len(spans)])

        cursor = end + 1

    full_text = " ".join(text_parts)
    return {"text": full_text, "spans": spans}

def validate_image(path: Path) -> bool:
    if path.stat().st_size == 0:
        return False

    try:
        with Image.open(path) as im:
            im.verify()

        with Image.open(path) as im:
            im.load()
            if im.width <= 0 or im.height <= 0:
                return False
            if im.mode not in ("RGB", "RGBA", "L"):
                pass


        with Image.open(path) as im:
            buf = io.BytesIO()
            im.save(buf, format=im.format)

        return True
    except Exception:
        return False

def dispatch_document(file_path: str):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        if not validate_image(path):
            return {"status": "corrupted", "reason": "image integrity check failed"}

        return run_ocr(path)

    elif suffix in {".rtf", ".txt"}:
        return run_llm_extraction(path)

    else:
        return {"status": "unsupported", "suffix": suffix}


if __name__ == "__main__":
    result = dispatch_document("test.txt")
    print(result)