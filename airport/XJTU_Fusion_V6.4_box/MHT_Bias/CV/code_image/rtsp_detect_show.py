import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent
MHT_BIAS_ROOT = BASE_DIR.parents[1]
CV_ROOT = BASE_DIR.parent
SRC_DIR = MHT_BIAS_ROOT / "Src"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO
from core.app_config import CV_DETECTION_RESULTS_FILE, CV_MODEL_WEIGHTS_FILE, OPTICAL_IP

# ???????????????? RTSP ??????? mp4?
RTSP_URL = f"rtsp://{OPTICAL_IP}:554/channel=0,stream=0"
MODEL_PATH = Path(CV_MODEL_WEIGHTS_FILE)
OPTICAL_STATUS_FILE = SRC_DIR / "data" / "optical_status.json"
CV_RESULT_FILE = Path(CV_DETECTION_RESULTS_FILE)

CONF_THRES = 0.65
MAX_QUEUE = 3
WINDOW_NAME = "Optical RTSP Detect"
SHOW_WINDOW = os.environ.get("AIRR_CV_SHOW_WINDOW", "1").strip().lower() not in ("0", "false", "no", "off")
REQUIRE_RECORDING_ACTIVE = os.environ.get("AIRR_CV_DETECT_REQUIRE_RECORDING", "1").strip().lower() not in ("0", "false", "no", "off")

# ???????????????????
# ?????????????????? False?
DETECT_ONLY_WHEN_TRACKING = True
TRACKING_STALE_SEC = 1.0

# ??? ROI?????????????????
DETECT_TRACKING_ROI_ONLY = True
ROI_EXPAND_RATIO = 0.6
ROI_MIN_SIZE = 96

# ???????????????? LABEL_FONT_SCALE / LABEL_THICKNESS?
BOX_THICKNESS = 4
LABEL_FONT_SCALE = 1.2
LABEL_THICKNESS = 3
STATUS_FONT_SCALE = 1.0
STATUS_THICKNESS = 2
DISPLAY_SCALE = 0.7
CV_RESULT_WRITE_INTERVAL_SEC = 0.2

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def check_rtsp_reachability(host: str, port: int = 554, timeout_sec: float = 2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True, f"tcp_ok({host}:{port})"
    except Exception as exc:
        return False, f"tcp_failed({host}:{port}): {exc}"


class RtspReader:
    def __init__(self, url: str, max_queue: int = 3):
        self.url = url
        self.cap = None
        self.running = False
        self.q = queue.Queue(maxsize=max_queue)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _open_capture(self):
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print("opening rtsp:", self.url)
                self.cap = self._open_capture()
                print("opened:", self.cap.isOpened())
                time.sleep(1.0)
                continue

            ok, frame = self.cap.read()
            if not ok:
                print("rtsp read failed, retrying...")
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(0.05)
                continue

            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


def read_optical_status():
    try:
        data = json.loads(OPTICAL_STATUS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"current_status": None, "reason": "status_file_missing"}
    except Exception as exc:
        return {"current_status": None, "reason": f"status_read_failed:{exc}"}

    try:
        age = time.time() - float(data.get("timestamp", 0.0))
    except (TypeError, ValueError):
        age = None
    data["status_age_sec"] = age
    return data


_last_cv_result_write_time = 0.0
_last_cv_result_error_time = 0.0


def write_cv_result(payload, force=False):
    global _last_cv_result_write_time, _last_cv_result_error_time
    now = time.time()
    if not force and now - _last_cv_result_write_time < CV_RESULT_WRITE_INTERVAL_SEC:
        return

    payload = dict(payload or {})
    payload["timestamp"] = now
    try:
        CV_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = CV_RESULT_FILE.with_suffix(f"{CV_RESULT_FILE.suffix}.{os.getpid()}.tmp")
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_file, CV_RESULT_FILE)
        _last_cv_result_write_time = now
    except Exception as exc:
        if now - _last_cv_result_error_time >= 2.0:
            print("write cv result skipped:", exc)
            _last_cv_result_error_time = now


def optical_tracking_enabled(status):
    if not DETECT_ONLY_WHEN_TRACKING:
        return True, "always_detect"

    current_status = status.get("current_status")
    age = status.get("status_age_sec")
    recording_active = bool(status.get("true_position_recording_active"))
    if current_status != 2:
        return False, f"optical_not_tracking(status={current_status})"
    if age is None or age > TRACKING_STALE_SEC:
        return False, f"optical_status_stale(age={age})"
    if REQUIRE_RECORDING_ACTIVE and not recording_active:
        return False, "waiting_for_j_record_command"
    return True, "optical_tracking_and_recording"


def draw_status(frame, text, color=(0, 0, 255)):
    cv2.rectangle(frame, (8, 8), (850, 54), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        STATUS_FONT_SCALE,
        color,
        STATUS_THICKNESS,
        cv2.LINE_AA,
    )


def draw_label_box(frame, x1, y1, x2, y2, label):
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
    )
    label_y1 = max(0, y1 - text_h - baseline - 10)
    label_y2 = label_y1 + text_h + baseline + 10
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 12, label_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 6, label_y2 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        (0, 0, 0),
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def select_tracking_target(status):
    targets = status.get("latest_targets") or []
    valid_targets = []
    for target in targets:
        try:
            pos_x = float(target.get("pos_x", 0))
            pos_y = float(target.get("pos_y", 0))
            width = float(target.get("width", 0))
            height = float(target.get("height", 0))
        except (TypeError, ValueError):
            continue
        if pos_x <= 0 or pos_y <= 0 or width <= 0 or height <= 0:
            continue
        valid_targets.append(target)

    if not valid_targets:
        return None

    # 跟踪状态下一般只有一个锁定目标；多目标时优先相似度高的目标。
    return max(valid_targets, key=lambda t: float(t.get("similarity") or 0.0))


def make_roi(frame, target):
    frame_h, frame_w = frame.shape[:2]
    cx = float(target["pos_x"])
    cy = float(target["pos_y"])
    width = max(float(target["width"]), ROI_MIN_SIZE)
    height = max(float(target["height"]), ROI_MIN_SIZE)

    width *= 1.0 + ROI_EXPAND_RATIO
    height *= 1.0 + ROI_EXPAND_RATIO

    x1 = max(0, int(round(cx - width / 2.0)))
    y1 = max(0, int(round(cy - height / 2.0)))
    x2 = min(frame_w, int(round(cx + width / 2.0)))
    y2 = min(frame_h, int(round(cy + height / 2.0)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_tracking_roi(frame, roi):
    if roi is None:
        return
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), BOX_THICKNESS)
    cv2.putText(
        frame,
        "tracking ROI",
        (x1, min(frame.shape[0] - 10, y2 + 32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        STATUS_FONT_SCALE,
        (255, 180, 0),
        STATUS_THICKNESS,
        cv2.LINE_AA,
    )


def draw_results(frame, results, model, offset=(0, 0), print_if_detected=False):
    current_dets = []
    detection_items = []
    offset_x, offset_y = offset

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            if isinstance(model.names, dict):
                cls_name = model.names.get(cls_id, str(cls_id))
            else:
                cls_name = model.names[cls_id] if cls_id < len(model.names) else str(cls_id)

            label = f"{cls_name} {score:.2f}"
            current_dets.append(f"{label} [{x1},{y1},{x2},{y2}]")
            detection_items.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": score,
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )
            draw_label_box(frame, x1, y1, x2, y2, label)

    if print_if_detected and current_dets:
        print("detections:", " | ".join(current_dets))

    return frame, len(current_dets) > 0, detection_items


def run_tracking_roi_detection(model, frame, status, device):
    target = select_tracking_target(status)
    roi = make_roi(frame, target) if target else None
    if DETECT_TRACKING_ROI_ONLY and roi is None:
        return frame, False, "no_tracking_box", []

    if DETECT_TRACKING_ROI_ONLY:
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        results = list(model.predict(roi_frame, conf=CONF_THRES, device=device, verbose=False, stream=True))
        frame, has_det, detections = draw_results(frame, results, model, offset=(x1, y1), print_if_detected=True)
        draw_tracking_roi(frame, roi)
        return frame, has_det, "roi", detections

    results = list(model.predict(frame, conf=CONF_THRES, device=device, verbose=False, stream=True))
    frame, has_det, detections = draw_results(frame, results, model, print_if_detected=True)
    return frame, has_det, "full_frame", detections


def main():
    reachable, reachability_reason = check_rtsp_reachability(OPTICAL_IP, 554)
    print(f"rtsp_check: {reachability_reason}")
    print("loading model:", MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    print(f"predict_device: {device}")
    if not reachable:
        print("rtsp unreachable, script will still start and keep retrying.")
    model = YOLO(str(MODEL_PATH))
    print(f"model loaded, device={device}")
    print("optical status file:", OPTICAL_STATUS_FILE)

    reader = RtspReader(RTSP_URL, max_queue=MAX_QUEUE)
    reader.start()

    prev_time = time.time()

    while True:
        frame = reader.read()
        if frame is None:
            continue

        status = read_optical_status()
        should_detect, reason = optical_tracking_enabled(status)

        if should_detect:
            frame, has_det, detect_scope, detections = run_tracking_roi_detection(model, frame, status, device)
            write_cv_result(
                {
                    "active": True,
                    "reason": "optical_tracking",
                    "scope": detect_scope,
                    "has_detection": bool(has_det),
                    "detections": detections,
                    "best_detection": max(detections, key=lambda d: d["confidence"]) if detections else None,
                    "optical_status": {
                        "current_status": status.get("current_status"),
                        "latest_azimuth": status.get("latest_azimuth"),
                        "latest_pitch": status.get("latest_pitch"),
                        "status_age_sec": status.get("status_age_sec"),
                    },
                }
            )
            status_text = (
                f"OPTICAL TRACKING | scope={detect_scope} | detect={'yes' if has_det else 'none'} | "
                f"az={status.get('latest_azimuth')} pitch={status.get('latest_pitch')}"
            )
            draw_status(frame, status_text, color=(0, 255, 0))
        else:
            write_cv_result(
                {
                    "active": False,
                    "reason": reason,
                    "scope": None,
                    "has_detection": False,
                    "detections": [],
                    "best_detection": None,
                    "optical_status": {
                        "current_status": status.get("current_status"),
                        "status_age_sec": status.get("status_age_sec"),
                    },
                }
            )
            draw_status(frame, f"Detection paused: {reason}", color=(0, 200, 255))

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            STATUS_FONT_SCALE,
            (0, 0, 255),
            STATUS_THICKNESS,
            cv2.LINE_AA,
        )

        if SHOW_WINDOW:
            if DISPLAY_SCALE != 1.0:
                frame_show = cv2.resize(
                    frame,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                frame_show = frame

            cv2.imshow(WINDOW_NAME, frame_show)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        else:
            time.sleep(0.001)

    reader.stop()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
