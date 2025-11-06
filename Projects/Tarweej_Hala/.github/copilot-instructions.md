This repository is a small ML-backed FastAPI service for counting bottles and computing shelf "bands"/zones using Ultralytics YOLO models.

Key points for an AI coding agent working on this codebase:

- Big picture
  - The code exposes a FastAPI app that runs one or two YOLO models and serves results and a boxes/bands preview via `/static`.
  - Main entry points: `app.py` (full multi-model implementation), `app1.py` and `app2.py` (lighter/variant versions). Prefer `app.py` as the canonical implementation.
  - Models live in `models/` (e.g. `models/model1.pt`, `models/model2.pt`). `model1` detects bottles (brands); `model2` detects shelf "bands" and optional EyeMarker polygons.
  - Endpoints to know:
    - GET `/` — returns device and model class names (useful to verify model loading).
    - POST `/predict` — returns counts + boxes-only annotated image URL. (Present in all variants.)
    - POST `/predict_with_bands` or `/predict_with_bands` variants — returns per-section and per-zone breakdowns and a bands preview (app.py implements detailed BandsResponse).

- Important implementation patterns and conventions
  - Models are loaded once at module import: `YOLO(MODEL_PATH).to(DEVICE)`. If model loading fails the process raises and exits — check `MODEL1_PATH`/`MODEL2_PATH` env vars.
  - Device selection is automatic: DEVICE = 0 if torch.cuda.is_available() else "cpu". Passing 0 to Ultralytics means CUDA device 0. For multi-GPU or explicit control, set `CUDA_VISIBLE_DEVICES` or change env vars.
  - Preview images are intentionally "boxes-only" (no text) and are saved to `OUT_DIR` (default `out/`). The service mounts `out/` at `/static` so annotated URLs are returned as e.g. `http://host:port/static/<file>`.
  - Band detection logic expects model-2 class names to include the substring `band` (case-insensitive) or `eye` for EyeMarker. The code uses masks when available, otherwise falls back to bounding boxes.
  - Fail-safes: if no bands are detected, the code creates a single full-image band so downstream logic still operates. EyeMarker detection (if present) can override which section(s) are treated as the Eye zone.
  - Filename safety: uploaded file stems are sanitized via `_safe_name(...)` to avoid unsafe characters.

- Parameters and request patterns
  - Predict endpoints accept query params: `conf`, `iou`, and `imgsz` (defaults in code: conf=0.25, iou=0.45, imgsz ~1024-1280). Tests and reproductions should reuse these defaults unless tuning is required.
  - Uploads use multipart `file` field. Example (curl):
    - curl -s -X POST -F "file=@image.jpg" "http://localhost:8000/predict?conf=0.25" - the response JSON includes `counts` and `annotated_image_url`.

- Dev / run workflow (discoverable in repo)
  - Dependencies are pinned in `requirements.tx` (note the filename). Use `pip install -r requirements.tx` to reproduce the environment locally.
  - Run the API locally with uvicorn: `uvicorn app:app --host 0.0.0.0 --port 8000 --reload` (use `app.py` as the module).
  - To verify model classes and device: GET `/` after server start — it returns `model1_classes` and `model2_classes` and device string.

- Files and examples to reference in edits
  - `app.py` — canonical, full-featured implementation (2 models, Zones/Bands logic, robust responses).
  - `app1.py` / `app2.py` — smaller or alternative versions; changes here may indicate experiments.
  - `models/` — binary model files (`.pt`) used by YOLO. Changing model artifacts requires updating `MODEL*_PATH` env vars or file names.
  - `OUT_DIR` usage — see `OUT_DIR = Path(os.getenv("OUT_DIR", "out"))` and `app.mount("/static", StaticFiles(...))` for static serving behavior.

- Troubleshooting hints (observed in code)
  - If the server fails at import with a model load error, inspect `MODEL1_PATH`/`MODEL2_PATH` and ensure the `.pt` files exist and are compatible with installed `ultralytics`/PyTorch versions.
  - When masks are missing, the code falls back to box-to-rectangle polygons — be careful when changing mask handling.
  - If GPU isn't used, verify `torch.cuda.is_available()` and the CUDA-compatible `torch` wheel in `requirements.tx` (pinned here). On multi-GPU systems, 0 means first GPU.

If anything in these notes is unclear or you want the agent to include additional sections (for example, test-running commands, sample images or a small runnable smoke test), tell me what to add and I'll iterate. 
