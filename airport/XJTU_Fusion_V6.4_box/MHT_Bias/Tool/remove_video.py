import os
from pathlib import Path
import cv2

# =========================
# 配置区
# =========================
VIDEO_DIR = r"D:\\video_new"
MIN_DURATION_SECONDS = 2

# True = 只打印将删除的文件，不真正删除
# False = 真正删除
DRY_RUN = False

# 要扫描的视频扩展名
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"
}


def get_video_duration(video_path: str):
    """
    使用 OpenCV 获取视频时长（秒）
    获取失败时返回 None
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[跳过] 无法打开视频: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps is None or fps <= 0:
            print(f"[跳过] FPS 无效: {video_path}")
            return None

        if frame_count is None or frame_count < 0:
            print(f"[跳过] 帧数无效: {video_path}")
            return None

        duration = frame_count / fps
        return float(duration)

    except Exception as e:
        print(f"[错误] 获取时长失败: {video_path} | {e}")
        return None
    finally:
        if cap is not None:
            cap.release()


def is_video_file(path: Path):
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def delete_short_videos(root_dir: str, min_duration: float, dry_run: bool = True):
    root = Path(root_dir)

    if not root.exists():
        print(f"[错误] 目录不存在: {root_dir}")
        return

    total_count = 0
    delete_count = 0
    skip_count = 0

    print("=" * 60)
    print(f"扫描目录: {root_dir}")
    print(f"删除阈值: 小于 {min_duration:.2f} 秒")
    print(f"模式: {'预演模式（不真正删除）' if dry_run else '正式删除模式'}")
    print("=" * 60)

    for file_path in root.rglob("*"):
        if not is_video_file(file_path):
            continue

        total_count += 1
        duration = get_video_duration(str(file_path))

        if duration is None:
            skip_count += 1
            continue

        print(f"[检测] {file_path} | 时长: {duration:.3f} 秒")

        if duration < min_duration:
            delete_count += 1
            if dry_run:
                print(f"  -> [将删除] {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"  -> [已删除] {file_path}")
                except Exception as e:
                    print(f"  -> [删除失败] {file_path} | {e}")

    print("\n" + "=" * 60)
    print("处理完成")
    print(f"视频总数: {total_count}")
    print(f"符合删除条件数量: {delete_count}")
    print(f"跳过数量: {skip_count}")
    print("=" * 60)


if __name__ == "__main__":
    delete_short_videos(
        root_dir=VIDEO_DIR,
        min_duration=MIN_DURATION_SECONDS,
        dry_run=DRY_RUN
    )
