"""
InferEdge - Production ML Inference Server
FastAPI + ONNX Runtime with dynamic batching, health checks, and structured logging.

Audit fixes (audit/production-remediation):
- P1 FIX: postprocess() assumed YOLOv5 output format [N, 85] with objectness at [4].
  Added format auto-detection: handles both YOLOv5 ([N, 85]) and
  YOLOv8 ([4+C, N] transposed to [N, 4+C], no objectness).
- P1 FIX: Race condition on global request_count / total_latency in async context.
  Replaced with asyncio.Lock-protected counter.
- P1 FIX: No file size limit on /detect endpoint — DoS vector.
  Added MAX_UPLOAD_MB limit enforced before reading body.
- P1 FIX: Letterboxing added to preprocess() — plain resize distorts
  aspect ratio and causes inaccurate bounding box coordinates.
- P2 FIX: INPUT_NAME now read from actual ONNX graph (was hardcoded to "images").
- P2 FIX: Added readiness check to /health (model loaded vs stub mode).
- P2 FIX: Graceful startup failure logged clearly.
- P2 FIX: Added /metrics Prometheus text format endpoint.
- P2 FIX: class_id bounds-checked against COCO_NAMES length.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("inferedge")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH    = Path(os.environ.get("MODEL_PATH",  "models/model.onnx"))
INPUT_SHAPE   = (640, 640)           # (H, W)
CONF_THRESH   = float(os.environ.get("CONF_THRESH", "0.25"))
IOU_THRESH    = float(os.environ.get("IOU_THRESH",  "0.45"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
_DEVICE_PREF  = os.environ.get("ORT_PROVIDER", "CUDAExecutionProvider")

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

# ---------------------------------------------------------------------------
# ONNX Model
# ---------------------------------------------------------------------------

class ONNXModel:
    def __init__(self, model_path: Path):
        providers = [_DEVICE_PREF, "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session    = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name    # e.g. "images" or "input"
        self.out_name   = self.session.get_outputs()[0].name
        # Detect output format from shape metadata
        out_shape = self.session.get_outputs()[0].shape        # [1, 85, 8400] or [1, 8400, 85]
        try:
            dim1, dim2 = int(out_shape[1]), int(out_shape[2])
            # YOLOv8: [1, 4+C, N] where 4+C < N  (e.g. 84 < 8400)
            # YOLOv5: [1, N, 5+C] where N > 5+C  (e.g. 25200 > 85)
            self.yolov8 = dim1 < dim2
        except Exception:
            self.yolov8 = False
        log.info(
            "Model loaded: %s | Provider: %s | Input: %s | Format: %s",
            model_path, self.session.get_providers()[0], self.input_name,
            "YOLOv8" if self.yolov8 else "YOLOv5",
        )

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run([self.out_name], {self.input_name: blob})[0]


# ---------------------------------------------------------------------------
# Preprocessing (letterbox + normalise)
# ---------------------------------------------------------------------------

def letterbox_frame(
    img: np.ndarray,
    new_shape: tuple[int, int] = INPUT_SHAPE,
    color: tuple = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w   = img.shape[:2]
    nh, nw = new_shape
    scale  = min(nw / w, nh / h)
    new_w  = int(round(w * scale))
    new_h  = int(round(h * scale))
    dw     = (nw - new_w) / 2
    dh     = (nh - new_h) / 2
    img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, scale, (dw, dh)


def preprocess(image_bytes: bytes) -> tuple[np.ndarray, int, int, float, tuple[float, float]]:
    nparr  = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    orig_h, orig_w = img.shape[:2]
    padded, scale, pad = letterbox_frame(img, INPUT_SHAPE)
    rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]   # HWC → NCHW
    return blob, orig_w, orig_h, scale, pad


# ---------------------------------------------------------------------------
# Postprocessing (format-aware)
# ---------------------------------------------------------------------------

def postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    conf_thresh: float,
    yolov8: bool,
) -> List[dict]:
    """
    Parse detection output and remap to original image coordinates.

    YOLOv8 output: [1, 4+C, N] → each column = [cx,cy,w,h,cls0..clsN]
    YOLOv5 output: [1, N, 5+C] → each row    = [cx,cy,w,h,obj,cls0..clsN]
    """
    dw, dh = pad
    raw = output[0]    # strip batch dim

    if yolov8:
        # [4+C, N] → transpose → [N, 4+C]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        def get_conf_class(row):
            class_scores = row[4:]
            class_id     = int(np.argmax(class_scores))
            return float(class_scores[class_id]), class_id
    else:
        # YOLOv5: [N, 5+C], objectness at index 4
        def get_conf_class(row):
            obj          = float(row[4])
            class_scores = row[5:]
            class_id     = int(np.argmax(class_scores))
            score        = float(class_scores[class_id]) * obj
            return score, class_id

    boxes, scores, class_ids = [], [], []
    for row in raw:
        confidence, class_id = get_conf_class(row)
        if confidence < conf_thresh:
            continue
        if class_id >= len(COCO_NAMES):
            continue
        cx, cy, w, h = row[:4]
        x1 = int((cx - w / 2 - dw) / scale)
        y1 = int((cy - h / 2 - dh) / scale)
        x2 = int((cx + w / 2 - dw) / scale)
        y2 = int((cy + h / 2 - dh) / scale)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices    = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, IOU_THRESH)
    detections = []
    for i in indices:
        x, y, bw, bh = boxes[i]
        detections.append({
            "class_id":   class_ids[i],
            "class_name": COCO_NAMES[class_ids[i]],
            "confidence": round(scores[i], 4),
            "bbox":       [x, y, x + bw, y + bh],
        })
    return detections


# ---------------------------------------------------------------------------
# App state (thread-safe counters)
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.lock          = asyncio.Lock()
        self.request_count = 0
        self.total_latency = 0.0

    async def record(self, ms: float) -> None:
        async with self.lock:
            self.request_count += 1
            self.total_latency += ms

    async def snapshot(self) -> dict:
        async with self.lock:
            n   = self.request_count
            avg = (self.total_latency / n) if n else 0.0
        return {"requests_total": n, "avg_latency_ms": round(avg, 2)}


stats: _Stats = _Stats()
model: ONNXModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        try:
            model = ONNXModel(MODEL_PATH)
        except Exception as exc:
            log.error("Model load failed: %s — running in stub mode", exc)
    else:
        log.warning("Model not found at %s — running in stub mode", MODEL_PATH)
    yield
    log.info("InferEdge shutting down.")


app = FastAPI(
    title="InferEdge",
    description="Production ML inference server — YOLOv5/YOLOv8 ONNX + FastAPI",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class DetectionResponse(BaseModel):
    detections:  List[dict]
    latency_ms:  float
    image_size:  List[int]
    model_format: str


@app.get("/health")
async def health():
    return {
        "status":       "ok" if model else "degraded",
        "model_loaded": model is not None,
        "model_path":   str(MODEL_PATH),
        "model_format": ("YOLOv8" if (model and model.yolov8) else "YOLOv5") if model else "n/a",
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    snap = await stats.snapshot()
    lines = [
        "# HELP inferedge_requests_total Total inference requests",
        "# TYPE inferedge_requests_total counter",
        f"inferedge_requests_total {snap['requests_total']}",
        "# HELP inferedge_avg_latency_ms Average inference latency",
        "# TYPE inferedge_avg_latency_ms gauge",
        f"inferedge_avg_latency_ms {snap['avg_latency_ms']}",
        "# HELP inferedge_model_loaded Whether model is loaded",
        "# TYPE inferedge_model_loaded gauge",
        f"inferedge_model_loaded {1 if model else 0}",
    ]
    return "\n".join(lines) + "\n"


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    conf: float = CONF_THRESH,
):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Size guard before reading
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB} MB")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, f"Expected image, got: {file.content_type}")

    image_bytes = await file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB} MB")

    t0 = time.perf_counter()
    try:
        blob, orig_w, orig_h, scale, pad = preprocess(image_bytes)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    output     = await asyncio.get_event_loop().run_in_executor(None, model.infer, blob)
    detections = postprocess(output, scale, pad, conf, model.yolov8)
    latency_ms = (time.perf_counter() - t0) * 1000

    await stats.record(latency_ms)
    log.info("Detected %d objects in %.1f ms", len(detections), latency_ms)

    return DetectionResponse(
        detections   = detections,
        latency_ms   = round(latency_ms, 2),
        image_size   = [orig_w, orig_h],
        model_format = "YOLOv8" if model.yolov8 else "YOLOv5",
    )


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False, workers=1)
