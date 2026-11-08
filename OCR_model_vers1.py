from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from pathlib import Path
from striprtf.striprtf import rtf_to_text


import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL = "ai-forever/sage-fredt5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL, eos_token="</s>")
model = T5ForConditionalGeneration.from_pretrained(MODEL)
model.eval()  # CPU по умолчанию, как у тебя device=-1 """

file = "test_file_2_for_OCR_model.rtf"


def read_rtf_safely(file):

    for enc in ["utf-8","cp1251","latin-1"]:
        try:
            with open(file,"r",encoding=enc) as f:
                
                raw_bytes = f.read()
            return rtf_to_text(raw_bytes)
            
        except UnicodeDecodeError:
            continue

        return None

def read_file(file):
    file_path = Path(file)
    suffix = file_path.suffix

    if suffix == ".png":


        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

        image = Image.open("scan.png").convert("RGB")

        pixel_values = processor(images = image, return_tensors = "pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids,skip_special_tokens = True)[0]

        returned_data = generated_ids

    elif suffix == ".rtf":



        


        res = read_rtf_safely(file)

        if res != None:
            return res

        else:

            return False


r = read_file(file)


def split_long_text(text,chunk_size):
    chunks = []

    while len(text) > chunk_size:

        cut_pos = text.rfind(" ", 0, chunk_size)

        if cut_pos == -1:
            cut_pos = chunk_size

        chunks.append(text[:cut_pos].strip())
        text = text[cut_pos:].strip()

    if text:
        chunks.append(text)

    return chunks

def correct_piece(piece):
    encodings = tokenizer(piece, return_tensors = "pt")
    with torch.no_grad():
        out = model.generate(
            **encodings,
            num_beams = 4,
            do_sample = False,
            max_new_tokens = 256,)

        return tokenizer.batch_decode(out,skip_special_tokens = True)[0].strip()


def check_file(result):
    if result != False:

        paragraphs = [p.strip() for p in result.split("\n") if p.strip()]
        total = sum(1 for p in paragraphs for _ in (split_long_text(p, 400) if len(p) > 512 else [p]))
        done = 0

        fixed_pieces = []
        for p in paragraphs:

            pieces = split_long_text(p,chunk_size=400) if len(p) > 512 else [p]
            for piece in pieces:
                done += 1
                print(f"[{done}/{total}] обработка: {piece[:60]!r}...", flush=True)
                fixed_pieces.append(correct_piece(piece))

        return "\n".join(fixed_pieces)
    else:
        return False
  

print(check_file(r))
