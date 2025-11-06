import io, os, re
from typing import Dict
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import torch
import cv2  # for manual box drawing fallback

# ---- Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/model1.pt")
OUT_DIR = Path(os.getenv("OUT_DIR", "out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 0 if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Tarweej Model-1 (Counts + Boxes-only Preview)", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(OUT_DIR), html=False), name="static")

# ---- Load model once ----
model = YOLO(MODEL_PATH)
model.to(DEVICE)
CLASS_NAMES = model.model.names  # id -> class name

class CountResponse(BaseModel):
    counts: Dict[str, int]
    annotated_image_url: str  # boxes-only (no text)

@app.get("/")
def root():
    return {
        "message": "POST /predict with an image. Returns counts + boxes-only annotated_image_url.",
        "device": "cuda" if DEVICE == 0 else "cpu",
        "classes": CLASS_NAMES
    }

def _safe_name(upload_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(upload_name or "upload").stem)
    return f"{stem}_annotated_boxes.png"

def _save_png(rgb: np.ndarray, out_path: Path):
    Image.fromarray(rgb).save(out_path, format="PNG")

@app.post("/predict", response_model=CountResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.9),
    iou: float = Query(0.45, ge=0.1, le=0.9),
    imgsz: int = Query(1280, ge=320, le=2048),
):
    # Read image
    data = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")

    # Inference
    res = model.predict(pil_img, conf=conf, iou=iou, imgsz=imgsz, device=DEVICE, verbose=False)
    r = res[0]

    # --- counts (brand -> count), no labels on image ---
    counts = Counter()
    if r.boxes is not None and len(r.boxes) > 0:
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for cid in cls_ids:
            cname = CLASS_NAMES.get(int(cid), str(int(cid)))
            counts[cname] += 1

    # --- build a boxes-only preview ---
    # Try Ultralytics plot with labels disabled (supported in recent versions).
    preview_bgr = None
    try:
        preview_bgr = r.plot(labels=False, boxes=True, conf=False)  # boxes only
    except TypeError:
        # Fallback: draw rectangles manually without text
        # Start from original image
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            for (x1, y1, x2, y2) in xyxy:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)  # no text
        preview_bgr = img_bgr

    # Save PNG (convert BGR->RGB)
    fname = _safe_name(file.filename)
    out_path = OUT_DIR / fname
    _save_png(preview_bgr[:, :, ::-1], out_path)

    annotated_url = str(request.base_url).rstrip("/") + f"/static/{fname}"
    return CountResponse(counts=dict(counts), annotated_image_url=annotated_url)
