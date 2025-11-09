# app.py
import io, os, re
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import torch
import cv2

# ---------------------- Config ----------------------
MODEL1_PATH = os.getenv("MODEL1_PATH", "models/model1.pt")  # bottle detector
MODEL2_PATH = os.getenv("MODEL2_PATH", "models/model2.pt")  # bands (class must include 'Band' in its name)
OUT_DIR = Path(os.getenv("OUT_DIR", "out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 0 if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Tarweej Models API (Counts + Bands)", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(OUT_DIR), html=False), name="static")

# ---------------------- Load models once ----------------------
try:
    model1 = YOLO(MODEL1_PATH).to(DEVICE)  # bottles
    CLASS_NAMES_1: Dict[int, str] = model1.model.names
except Exception as e:
    raise RuntimeError(f"Failed to load Model-1 at {MODEL1_PATH}: {e}")

try:
    model2 = YOLO(MODEL2_PATH).to(DEVICE)  # bands (+ optional EyeMarker)
    CLASS_NAMES_2: Dict[int, str] = model2.model.names
except Exception as e:
    raise RuntimeError(f"Failed to load Model-2 at {MODEL2_PATH}: {e}")

# ---------------------- Schemas ----------------------
class CountResponse(BaseModel):
    counts: Dict[str, int]
    annotated_image_url: str

class MinimalBandsResponse(BaseModel):
    total_bottles: int
    brand_counts: Dict[str, int]
    category_counts: Dict[str, int]          # {"honey": x, "olive": y, "unknown": z}
    top_bottles: int
    middle_bottles: int
    bottom_bottles: int
    annotated_image_url: str

# ---------------------- Helpers ----------------------
def _safe_name(upload_name: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(upload_name or "upload").stem)
    return f"{stem}_{suffix}.png"

def _save_png_bgr(img_bgr: np.ndarray, out_path: Path):
    Image.fromarray(img_bgr[:, :, ::-1]).save(out_path, format="PNG")

def _point_in_poly(pt: Tuple[float, float], poly: np.ndarray) -> bool:
    """Ray casting; poly is Nx2 array (float)."""
    if poly.shape[0] < 3:
        return False
    x, y = pt
    inside = False
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def _polygon_from_ultra_mask(mask_xy: np.ndarray) -> np.ndarray:
    """Ensure polygon Nx2 float array from Ultralytics masks.xy entry."""
    poly = np.asarray(mask_xy, dtype=np.float32)
    if poly.ndim == 1:
        poly = poly.reshape(-1, 2)
    return poly

def _bands_from_model2_result(r2) -> Tuple[List[dict], List[np.ndarray]]:
    """
    Parse bands & optional eye markers from a YOLO result.
    Returns:
      bands:   [{'poly': Nx2 float32, 'cy': float}, ...]
      markers: [Nx2 float32, ...]  (not used in 3-zone mode)
    A class is considered a band if its name contains 'band' (case-insensitive).
    """
    bands: List[dict] = []
    markers: List[np.ndarray] = []
    name_map = CLASS_NAMES_2

    if getattr(r2, "boxes", None) is not None and len(r2.boxes) > 0:
        cls_ids = r2.boxes.cls.cpu().numpy().astype(int)
        xyxy = r2.boxes.xyxy.cpu().numpy()
        masks_xy = []
        if getattr(r2, "masks", None) is not None and r2.masks is not None and getattr(r2.masks, "xy", None):
            masks_xy = r2.masks.xy  # list of polygons
        for i, cid in enumerate(cls_ids):
            cname = str(name_map.get(int(cid), str(int(cid)))).lower()
            if "band" in cname:
                if masks_xy:
                    poly = _polygon_from_ultra_mask(masks_xy[i])
                else:
                    x1, y1, x2, y2 = xyxy[i]
                    poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                cy = float(np.mean(poly[:, 1]))
                bands.append({"poly": poly, "cy": cy})
            elif "eye" in cname:
                # not used in 3-zone endpoint, but we collect anyway
                if masks_xy:
                    markers.append(_polygon_from_ultra_mask(masks_xy[i]))
                else:
                    x1, y1, x2, y2 = xyxy[i]
                    markers.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
    return bands, markers

def _split_three_zones(N: int) -> dict:
    """
    Split section ids 1..N into contiguous Top/Middle/Bottom groups.
    Remainder goes to Top then Middle. Examples:
      N=5  -> Top=[1,2], Middle=[3],   Bottom=[4,5]
      N=6  -> Top=[1,2], Middle=[3,4], Bottom=[5,6]
      N=8  -> Top=[1,2,3], Middle=[4,5,6], Bottom=[7,8]
    """
    if N <= 0:
        return {"Top": [], "Middle": [], "Bottom": []}
    q, r = divmod(N, 3)
    top_n = q + (1 if r > 0 else 0)
    middle_n = q + (1 if r > 1 else 0)
    bottom_n = q
    ids = list(range(1, N + 1))
    top_ids = ids[:top_n]
    middle_ids = ids[top_n: top_n + middle_n]
    bottom_ids = ids[top_n + middle_n:]
    return {"Top": top_ids, "Middle": middle_ids, "Bottom": bottom_ids}

def _infer_category(class_name: str) -> str:
    n = class_name.lower()
    if "_honey" in n:
        return "honey"
    if "_olive" in n:
        return "olive"
    return "unknown"

def _draw_preview_boxes_and_bands_labeled(orig_rgb: np.ndarray,
                                          bottle_xyxy: np.ndarray,
                                          bands_sorted: list,
                                          zone_map: dict) -> np.ndarray:
    """Boxes-only + translucent bands + 'Top/Middle/Bottom' labels."""
    img = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = img.shape[:2]
    overlay = img.copy()

    colors = {
        "Top": (0, 255, 0),
        "Middle": (0, 200, 255),
        "Bottom": (255, 200, 0)
    }

    # Fill each band's polygon with its zone color
    for zone_name, ids in zone_map.items():
        for sid in ids:
            b = next((bb for bb in bands_sorted if bb["section_id"] == sid), None)
            if b is None:
                continue
            poly = np.round(b["poly"]).astype(np.int32)
            if poly.shape[0] >= 3:
                cv2.fillPoly(overlay, [poly], colors[zone_name])

    # Blend lightly
    img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

    # Put zone labels (left side, vertically centered on first band of each zone)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for zone_name, ids in zone_map.items():
        if not ids:
            continue
        b = next((bb for bb in bands_sorted if bb["section_id"] == ids[0]), None)
        if b is None:
            continue
        cy = int(np.mean(b["poly"][:, 1]))
        tx = int(W * 0.02)
        ty = max(30, min(H - 30, cy))
        cv2.putText(img, zone_name.upper(), (tx, ty), font, 1.0, colors[zone_name], 2, cv2.LINE_AA)

    # Draw boxes (no text)
    if bottle_xyxy is not None and len(bottle_xyxy) > 0:
        for (x1, y1, x2, y2) in bottle_xyxy.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return img

# ---------------------- Endpoints ----------------------
@app.get("/")
def root():
    return {
        "message": "POST /predict (counts) or /predict_with_bands (Top/Middle/Bottom).",
        "device": "cuda" if DEVICE == 0 else "cpu",
        "model1_classes": CLASS_NAMES_1,
        "model2_classes": CLASS_NAMES_2
    }

# ---------- 1) Simple counts + boxes-only preview ----------
@app.post("/predict", response_model=CountResponse)
async def predict_simple(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.9),
    iou: float = Query(0.45, ge=0.1, le=0.9),
    imgsz: int = Query(1280, ge=320, le=2048),
):
    data = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    res = model1.predict(pil_img, conf=conf, iou=iou, imgsz=imgsz, device=DEVICE, verbose=False)[0]

    counts = Counter()
    xyxy = None
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        for cid in cls_ids:
            counts[CLASS_NAMES_1.get(int(cid), str(int(cid)))] += 1

    # boxes-only preview
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if xyxy is not None:
        for (x1, y1, x2, y2) in xyxy.astype(int):
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

    fname = _safe_name(file.filename, "annotated_boxes")
    out_path = OUT_DIR / fname
    _save_png_bgr(img_bgr, out_path)
    annotated_url = str(request.base_url).rstrip("/") + f"/static/{fname}"

    return CountResponse(counts=dict(counts), annotated_image_url=annotated_url)

# ---------- 2) Bands + Top/Middle/Bottom ----------
@app.post("/predict_with_bands", response_model=MinimalBandsResponse)
async def predict_with_bands(
    request: Request,
    file: UploadFile = File(...),
    conf1: float = Query(0.25, ge=0.05, le=0.9, description="Model-1 conf"),
    conf2: float = Query(0.25, ge=0.05, le=0.9, description="Model-2 conf"),
    iou1: float = Query(0.45, ge=0.1, le=0.9, description="Model-1 NMS"),
    iou2: float = Query(0.45, ge=0.1, le=0.9, description="Model-2 NMS"),
    imgsz1: int = Query(1280, ge=320, le=2048),
    imgsz2: int = Query(1024, ge=320, le=2048),
):
    # 1) Read image
    data = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil_img)

    # 2) Model-1: bottles
    r1 = model1.predict(pil_img, conf=conf1, iou=iou1, imgsz=imgsz1, device=DEVICE, verbose=False)[0]
    bottle_xyxy = r1.boxes.xyxy.cpu().numpy() if (r1.boxes is not None and len(r1.boxes) > 0) else np.zeros((0, 4))
    bottle_cls = r1.boxes.cls.cpu().numpy().astype(int) if (r1.boxes is not None and len(r1.boxes) > 0) else np.zeros((0,), dtype=int)

    # 3) Model-2: bands
    r2 = model2.predict(pil_img, conf=conf2, iou=iou2, imgsz=imgsz2, device=DEVICE, verbose=False)[0]
    bands, _unused_markers = _bands_from_model2_result(r2)
    if not bands:
        # Fail-safe: single full-image band so we still respond
        H, W = rgb.shape[:2]
        poly = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        bands = [{"poly": poly, "cy": float(H / 2)}]

    # 4) Sort bands and assign section ids
    bands.sort(key=lambda b: b["cy"])
    for i, b in enumerate(bands, start=1):
        b["section_id"] = i

    # 5) 3-zone map (Top/Middle/Bottom) for any N (3..12 supported, but any N works)
    N = len(bands)
    zone_map = _split_three_zones(N)  # {'Top':[...], 'Middle':[...], 'Bottom':[...]}

    # Index section -> zone
    section_to_zone: Dict[int, str] = {}
    for z, ids in zone_map.items():
        for sid in ids:
            section_to_zone[sid] = z

    # 6) Assign bottles, compute brand/category + zone totals
    brand_counts = Counter()
    category_counts = Counter()
    zone_totals = Counter({"Top": 0, "Middle": 0, "Bottom": 0})

    for i in range(len(bottle_cls)):
        x1, y1, x2, y2 = bottle_xyxy[i]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # section assignment (point-in-polygon)
        sid = None
        for b in bands:
            if _point_in_poly((cx, cy), b["poly"]):
                sid = b["section_id"]
                break

        cname = CLASS_NAMES_1.get(int(bottle_cls[i]), str(int(bottle_cls[i])))
        brand_counts[cname] += 1
        category_counts[_infer_category(cname)] += 1

        if sid is not None:
            z = section_to_zone.get(sid, None)
            if z is not None:
                zone_totals[z] += 1

    total_bottles = int(sum(brand_counts.values()))

    # 7) Render preview with bands labeled TOP/MIDDLE/BOTTOM + boxes-only for bottles
    preview_bgr = _draw_preview_boxes_and_bands_labeled(rgb, bottle_xyxy, bands, zone_map)
    fname = _safe_name(file.filename, "bands_top_middle_bottom")
    out_path = OUT_DIR / fname
    _save_png_bgr(preview_bgr, out_path)
    annotated_url = str(request.base_url).rstrip("/") + f"/static/{fname}"

    # 8) Build minimal response
    return MinimalBandsResponse(
        total_bottles=total_bottles,
        brand_counts=dict(brand_counts),
        category_counts={
            "honey": int(category_counts.get("honey", 0)),
            "olive": int(category_counts.get("olive", 0)),
            "unknown": int(category_counts.get("unknown", 0)),
        },
        top_bottles=int(zone_totals["Top"]),
        middle_bottles=int(zone_totals["Middle"]),
        bottom_bottles=int(zone_totals["Bottom"]),
        annotated_image_url=annotated_url,
    )
