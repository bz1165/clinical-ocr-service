import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pdf2image import convert_from_bytes
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel

# bring in your mapping logic
from scripts.convert_data import map_to_templates, extract_subject_id

# 1) pick device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2) load your local InternVL3-8B checkpoint
MODEL_DIR = "pretrained/InternVL3-8B"
tokenizer_ivl = AutoTokenizer.from_pretrained(
    MODEL_DIR, trust_remote_code=True, use_fast=False
)
model_ivl = AutoModel.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

# generation config
GEN_CFG = dict(max_new_tokens=512, do_sample=False)

# 3) image preprocessing helpers
def build_transform(input_size=448):
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

def dynamic_preprocess(img: Image.Image, image_size: int = 448, max_tiles: int = 12, use_thumbnail: bool = True):
    w, h = img.size
    ar = w / h
    # build all (rows,cols) pairs whose product ≤ max_tiles
    pairs = []
    for n in range(1, max_tiles + 1):
        for rows in range(1, n + 1):
            if n % rows == 0:
                cols = n // rows
                pairs.append((rows, cols))
    # choose best aspect‐ratio match
    rows, cols = min(pairs, key=lambda rc: abs((rc[1]/rc[0]) - ar))
    img_resized = img.resize((cols * image_size, rows * image_size))
    tiles = [
        img_resized.crop((c*image_size, r*image_size, (c+1)*image_size, (r+1)*image_size))
        for r in range(rows) for c in range(cols)
    ]
    if use_thumbnail and len(tiles) > 1:
        tiles.append(img_resized.resize((image_size, image_size)))
    return tiles

def pil_to_pixel_values(img: Image.Image, input_size: int = 448, max_tiles: int = 12):
    tfm   = build_transform(input_size)
    tiles = dynamic_preprocess(img, image_size=input_size, max_tiles=max_tiles)
    batch = torch.stack([tfm(t) for t in tiles], dim=0)
    return batch.to(device).to(torch.bfloat16)

# 4) FastAPI setup
app = FastAPI(title="Clinical OCR Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ping")
async def ping():
    return {"msg": "pong"}

@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    """
    Accepts a PDF or image, runs InternVL3 OCR + mapping,
    and returns raw text + mapped fields per Excel template.
    """
    data = await file.read()
    ext  = os.path.splitext(file.filename)[1].lower()

    # convert to PIL pages
    if ext == ".pdf":
        pages = convert_from_bytes(data, dpi=300)
    else:
        pages = [Image.open(io.BytesIO(data))]

    # OCR each page
    chunks = []
    for i, img in enumerate(pages, start=1):
        pv = pil_to_pixel_values(img)
        prompt = "<image>\n请提取表单字段及数值："
        txt = model_ivl.chat(
            tokenizer_ivl,
            pv,
            prompt,
            generation_config=GEN_CFG,
            return_history=False
        ).strip()
        chunks.append(f"--- Page {i} ---\n{txt}")

    text_blob = "\n\n".join(chunks)

    # semantic mapping into your Excel templates
    filled_templates = map_to_templates(text_blob)
    subject_id      = extract_subject_id(text_blob)

    # convert DataFrames to JSON-friendly dicts
    mapped = {
        name: df.to_dict(orient="records")[0]
        for name, df in filled_templates.items()
    }

    return JSONResponse({
        "subject_id": subject_id,
        "raw_text":   text_blob,
        "mapped":     mapped
    })
