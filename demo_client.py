"""
InferEdge — Quick client demo
Tests the inference server with a sample image (no GPU required).
Usage:
  1. Start server: python inference.py
  2. In another terminal: python demo_client.py
"""

import requests
import time
import json
import io
import sys
import urllib.request
from pathlib import Path

SERVER = "http://localhost:8000"

SAMPLE_URL = (
    "https://ultralytics.com/images/zidane.jpg"
)


def check_health():
    try:
        r = requests.get(f"{SERVER}/health", timeout=5)
        data = r.json()
        print(f"✓ Server healthy | model={data.get('model')} | device={data.get('device')}")
        return True
    except Exception as e:
        print(f"✗ Server not reachable at {SERVER}: {e}")
        print("  Start it with: python inference.py")
        return False


def download_sample(path="sample.jpg"):
    if not Path(path).exists():
        print(f"Downloading sample image → {path}")
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return path


def run_inference(img_path: str, n_runs: int = 10):
    print(f"\nRunning inference on {img_path} ({n_runs} requests)...")
    latencies = []
    results = None
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    for i in range(n_runs):
        t0 = time.perf_counter()
        r = requests.post(
            f"{SERVER}/predict",
            files={"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            timeout=30,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        if r.status_code == 200:
            results = r.json()
        else:
            print(f"  Request {i+1} failed: {r.status_code}")

    latencies.sort()
    print(f"\n── Latency stats ({n_runs} runs) ──────────────────")
    print(f"  P50:  {latencies[len(latencies)//2]:.1f} ms")
    print(f"  P95:  {latencies[int(len(latencies)*0.95)]:.1f} ms")
    print(f"  P99:  {latencies[int(len(latencies)*0.99)]:.1f} ms")
    print(f"  Min:  {min(latencies):.1f} ms")
    print(f"  Max:  {max(latencies):.1f} ms")

    if results:
        dets = results.get("detections", [])
        print(f"\n── Detections ({len(dets)} objects) ──────────────────")
        for d in dets[:5]:
            print(f"  {d['class_name']:<20} conf={d['confidence']:.3f}  bbox={d['bbox']}")


def main():
    print("InferEdge Demo Client")
    print("=" * 40)
    if not check_health():
        sys.exit(1)
    img = download_sample()
    run_inference(img, n_runs=20)
    print("\nDone. Edit SERVER variable to point to a remote endpoint.")


if __name__ == "__main__":
    main()
