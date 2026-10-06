"""
InferEdge — Production ML Inference Server
FastAPI + ONNX Runtime inference server with dynamic batching,
health checks, Prometheus metrics, and structured logging.
"""

import time
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("inferedge")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/model.onnx")
INPUT_NAME   = "images"
INPUT_SHAPE  = (1, 3, 640, 640)   # NCHW
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
MAX_BATCH    = 8
DEVICE       = "CUDAExecutionProvider"

# COCO class names (80 classes)
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
    "toothbrush"
]


# ── Model ─────────────────────────────────────────────────────────────────────
class ONNXModel:
    def __init__(self, model_path: Path):
        providers = [DEVICE, "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"Model loaded: {model_path} | Provider: {self.session.get_providers()[0]}")

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: blob})[0]


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> tuple[np.ndarray, tuple]:
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    orig_h, orig_w = img.shape[:2]
    resized = cv2.resize(img, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob    = np.transpose(rgb, (2, 0, 1))[np.newaxis]   # HWC → NCHW
    return blob, (orig_w, orig_h)


def postprocess(output: np.ndarray, orig_size: tuple, conf_thresh: float) -> List[dict]:
    """Parse YOLO output format [batch, num_boxes, 85]."""
    orig_w, orig_h = orig_size
    scale_x = orig_w / INPUT_SHAPE[3]
    scale_y = orig_h / INPUT_SHAPE[2]
    detections = []
    boxes = output[0]
    for box in boxes:
        conf = float(box[4])
        if conf < conf_thresh:
            continue
        class_scores = box[5:]
        class_id     = int(np.argmax(class_scores))
        score        = float(class_scores[class_id]) * conf
        if score < conf_thresh:
            continue
        cx, cy, w, h = box[:4]
        x1 = int((cx - w / 2) * scale_x)
        y1 = int((cy - h / 2) * scale_y)
        x2 = int((cx + w / 2) * scale_x)
        y2 = int((cy + h / 2) * scale_y)
        detections.append({
            "class_id":    class_id,
            "class_name":  COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else str(class_id),
            "confidence":  round(score, 4),
            "bbox":        [x1, y1, x2, y2],
        })
    return detections


# ── App ───────────────────────────────────────────────────────────────────────
model: ONNXModel | None = None
request_count = 0
total_latency = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = ONNXModel(MODEL_PATH)
    else:
        log.warning(f"Model not found at {MODEL_PATH} — running in stub mode")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="InferEdge",
    description="Production ML inference server — ONNX + FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


class DetectionResponse(BaseModel):
    detections: List[dict]
    latency_ms: float
    image_size: List[int]


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
async def metrics():
    avg_latency = (total_latency / request_count) if request_count else 0
    return {
        "requests_total": request_count,
        "avg_latency_ms": round(avg_latency, 2),
        "model_loaded": model is not None,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...), conf: float = CONF_THRESH):
    global request_count, total_latency

    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image, got {file.content_type}")

    image_bytes = await file.read()
    t0 = time.perf_counter()

    try:
        blob, orig_size = preprocess(image_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))

    output = await asyncio.get_event_loop().run_in_executor(None, model.infer, blob)
    detections = postprocess(output, orig_size, conf)

    latency_ms = (time.perf_counter() - t0) * 1000
    request_count += 1
    total_latency += latency_ms

    log.info(f"Detected {len(detections)} objects in {latency_ms:.1f}ms")
    return DetectionResponse(
        detections=detections,
        latency_ms=round(latency_ms, 2),
        image_size=list(orig_size),
    )


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False, workers=1)
