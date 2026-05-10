import os
import cv2
import time
import threading
from datetime import datetime

# =========================
# 配置区
# =========================
RTSP_URLS = [
    {
        "name": "stream0",
        "url": "rtsp://10.129.41.98:554/channel=0,stream=0"
    },
    {
        "name": "stream1",
        "url": "rtsp://10.129.41.98:554/channel=1,stream=0"
    }
]

SAVE_DIR = r"D:\video"

# 是否显示预览窗口
SHOW_WINDOW = False

# RTSP 优先 TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 退出标志
stop_event = threading.Event()


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def open_rtsp(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap


def create_writer(save_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    return writer


def record_rtsp(stream_name: str, rtsp_url: str):
    print(f"[{stream_name}] 启动录制线程")
    print(f"[{stream_name}] RTSP 地址: {rtsp_url}")

    file_name = f"{stream_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    save_path = os.path.join(SAVE_DIR, file_name)

    cap = None
    writer = None
    reconnect_count = 0
    width, height, fps = None, None, None

    while not stop_event.is_set():
        # 1. 打开 RTSP
        if cap is None or not cap.isOpened():
            print(f"[{stream_name}] 正在连接 RTSP...")
            cap = open_rtsp(rtsp_url)

            if not cap.isOpened():
                print(f"[{stream_name}] RTSP 打开失败，1秒后重试")
                time.sleep(1)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[{stream_name}] 首帧读取失败，重新连接")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            height, width = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 1 or fps > 120:
                fps = 25.0

            print(f"[{stream_name}] 分辨率: {width}x{height}, FPS: {fps}")
            print(f"[{stream_name}] 保存路径: {save_path}")

            writer = create_writer(save_path, fps, width, height)
            if not writer.isOpened():
                print(f"[{stream_name}] VideoWriter 打开失败")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            writer.write(frame)

            if SHOW_WINDOW:
                cv2.imshow(stream_name, frame)

        # 2. 正常读帧
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{stream_name}] 读帧失败，准备重连")
            reconnect_count += 1

            if cap is not None:
                cap.release()
                cap = None

            time.sleep(1)
            continue

        # 3. 如果分辨率变化，重建 writer
        current_h, current_w = frame.shape[:2]
        if current_w != width or current_h != height:
            print(f"[{stream_name}] 检测到分辨率变化: {width}x{height} -> {current_w}x{current_h}")

            width, height = current_w, current_h

            if writer is not None:
                writer.release()

            new_file_name = f"{stream_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            save_path = os.path.join(SAVE_DIR, new_file_name)

            writer = create_writer(save_path, fps, width, height)
            if not writer.isOpened():
                print(f"[{stream_name}] 分辨率变化后 VideoWriter 重建失败")
                time.sleep(1)
                continue

            print(f"[{stream_name}] 已切换保存文件: {save_path}")

        # 4. 写入视频
        writer.write(frame)

        # 5. 预览
        if SHOW_WINDOW:
            cv2.imshow(stream_name, frame)

    # 退出清理
    print(f"[{stream_name}] 正在退出，重连次数: {reconnect_count}")

    if cap is not None:
        cap.release()

    if writer is not None:
        writer.release()

    if SHOW_WINDOW:
        try:
            cv2.destroyWindow(stream_name)
        except Exception:
            pass


def main():
    ensure_dir(SAVE_DIR)
    print("保存目录:", SAVE_DIR)

    threads = []

    for item in RTSP_URLS:
        t = threading.Thread(
            target=record_rtsp,
            args=(item["name"], item["url"]),
            daemon=True
        )
        t.start()
        threads.append(t)

    print("所有 RTSP 录制线程已启动")
    print("按 q 键退出程序")

    try:
        while True:
            if SHOW_WINDOW:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("检测到 q，准备退出")
                    stop_event.set()
                    break
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("检测到 Ctrl+C，准备退出")
        stop_event.set()

    for t in threads:
        t.join()

    cv2.destroyAllWindows()
    print("程序已退出")


if __name__ == "__main__":
    main()
