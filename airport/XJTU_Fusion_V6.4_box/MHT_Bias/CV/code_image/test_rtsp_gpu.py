from pathlib import Path
import socket
import sys
import time

import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parents[1] / "Src"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO
from core.app_config import CV_MODEL_WEIGHTS_FILE, OPTICAL_IP

MODEL_PATH = Path(CV_MODEL_WEIGHTS_FILE)
RTSP_URL = f"rtsp://{OPTICAL_IP}:554/channel=0,stream=0"


def check_rtsp_reachability(host: str, port: int = 554, timeout_sec: float = 2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True, f"tcp_ok({host}:{port})"
    except Exception as exc:
        return False, f"tcp_failed({host}:{port}): {exc}"


def main():
    print(f"model: {MODEL_PATH}")
    print(f"rtsp: {RTSP_URL}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"predict_device: {device}")
    reachable, reachability_reason = check_rtsp_reachability(OPTICAL_IP, 554)
    print(f"rtsp_check: {reachability_reason}")

    model = YOLO(str(MODEL_PATH))
    print("opening rtsp...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    print(f"opened: {cap.isOpened()}")
    if not cap.isOpened():
        raise SystemExit(1)

    ok, frame = cap.read()
    cap.release()
    print(f"first_frame_ok: {ok}")
    if not ok or frame is None:
        raise SystemExit(2)

    h, w = frame.shape[:2]
    print(f"frame_shape: {w}x{h}")

    start = time.time()
    result_count = 0
    for _ in model.predict(
        source=frame,
        imgsz=640,
        conf=0.6,
        device=device,
        verbose=False,
        stream=True,
    ):
        result_count += 1
    elapsed = time.time() - start
    print(f"result_count: {result_count}")
    print(f"infer_time_sec: {elapsed:.3f}")


if __name__ == "__main__":
    main()
