from pathlib import Path
import sys

import torch

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parents[1] / "Src"
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO
from core.app_config import CV_MODEL_WEIGHTS_FILE

MODEL_FILE = Path(CV_MODEL_WEIGHTS_FILE)
DEFAULT_VIDEO_CANDIDATES = [
    Path(r"D:\video_new\stream0_status2_20260518_171148_20260518_171148.mp4"),
    BASE_DIR / "0418_01.mp4",
    BASE_DIR / "0419.mp4",
]


def pick_video_file() -> Path:
    for path in DEFAULT_VIDEO_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "未找到可用测试视频。已检查: "
        + ", ".join(str(path) for path in DEFAULT_VIDEO_CANDIDATES)
    )


def main():
    video_file = pick_video_file()
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"model: {MODEL_FILE}")
    print(f"video: {video_file}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    print(f"predict_device: {device}")

    model = YOLO(str(MODEL_FILE))
    result_count = 0
    for _ in model.predict(
        source=str(video_file),
        imgsz=640,
        conf=0.6,
        device=device,
        project=str(BASE_DIR / "runs"),
        name="testmp4_verify",
        save=True,
        verbose=False,
        stream=True,
    ):
        result_count += 1
    print(f"results: {result_count} frames")
    save_dir = getattr(model.predictor, "save_dir", None)
    if save_dir is not None:
        print(f"save_dir: {save_dir}")


if __name__ == "__main__":
    main()
