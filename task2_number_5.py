"""第五题：使用 CSRT 依次追踪两段视频中的小球并保存结果。"""

from pathlib import Path
import math

import cv2


VIDEO_FOLDER = Path(r"D:\comsen tasks\task all\task2\number_5")

VIDEO_NAMES = ["example3.mp4", "example4.mp4"]

OUTPUT_FOLDER = Path(__file__).resolve().parent / "results"

SHOW_VIDEO = True  # 是否显示追踪窗口。
MANUAL_ROI = False  # True 使用鼠标选框；False 使用预设 ROI。
MAX_WIDTH = 960  # 较大视频按比例缩至此宽度。

INITIAL_ROIS = {  # 原分辨率下的 (x, y, w, h)；更换视频需重新选框。
    "example3.mp4": (299, 8, 114, 116),
    "example4.mp4": (1425, 807, 330, 273),  # 白球在第一帧底部有局部裁切。
}


def create_tracker():
    """兼容不同 OpenCV 版本中 CSRT 的位置"""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError("当前 OpenCV 没有 CSRT。请查看学习说明中的环境安装部分。")


def choose_roi(frame, original_roi, scale, manual_roi):
    """返回处理画面中的 (x, y, w, h)，并检查选框是否有效。"""
    if manual_roi:
        print("请用鼠标拖框圈住小球，按 Enter 或空格确认；按 C 取消当前选框。")
        roi = cv2.selectROI("Select the ball", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select the ball")

    else:
        roi = tuple(round(value * scale) for value in original_roi)  # 将原视频坐标换算到缩放后的画面。

    x, y, w, h = map(int, roi)
    height, width = frame.shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError("没有选中小球，请重新运行并拖出一个矩形。")

    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("初始框超出画面，请检查坐标、视频尺寸或使用手动选框。")

    return x, y, w, h


def track_video(video_path, output_path, original_roi, show_video=True, manual_roi=False):
    """读取、初始化、逐帧追踪、画框、保存；最后返回本次处理的统计信息。"""
    video_path = Path(video_path)
    output_path = Path(output_path)

    if video_path.resolve() == output_path.resolve():
        raise ValueError("输入和输出不能是同一个文件，否则可能覆盖原视频。")

    cap = cv2.VideoCapture(str(video_path))
    writer = None
    written_frames = 0
    successful_updates = 0
    failed_updates = 0
    stopped_early = False

    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")

        ret, original_frame = cap.read()
        if not ret:
            raise RuntimeError("视频已打开，但是读不到第一帧。")

        original_height, original_width = original_frame.shape[:2]
        scale = min(1.0, MAX_WIDTH / original_width)
        width = round(original_width * scale)
        height = round(original_height * scale)
        frame_size = (width, height)
        frame = cv2.resize(original_frame, frame_size)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0  # 原视频帧率无效时使用回退值。
            print("无法读取有效帧率，按 30 FPS 保存。")
        delay_ms = max(1, round(1000 / fps))

        bbox = choose_roi(frame, original_roi, scale, manual_roi)
        tracker = create_tracker()
        init_result = tracker.init(frame, bbox)
        if init_result is False:  # 部分 OpenCV 版本成功时返回 None。
            raise RuntimeError("追踪器初始化失败，请重新选择小球。")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频：{output_path}")

        while True:
            if written_frames == 0:  # 第一帧只初始化，从第二帧开始 update。
                found = True
                status = "INITIALIZED"
            else:
                found, bbox = tracker.update(frame)
                if found:
                    successful_updates += 1
                    status = "TRACKING"
                else:
                    failed_updates += 1
                    status = "LOST"

            display_frame = frame.copy()  # 只在副本上标注，避免影响追踪器输入。
            if found:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center = (x + w // 2, y + h // 2)
                cv2.circle(display_frame, center, 3, (0, 255, 0), -1)

            color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(display_frame, f"CSRT: {status}", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display_frame, f"Frame {written_frames + 1} | Source FPS {fps:.2f}",  # Source FPS 表示原视频帧率，不是处理速度。
                        (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            writer.write(display_frame)  # 失败帧也保存，保持输出时间轴连续。
            written_frames += 1

            if written_frames % 100 == 0:
                print(f"{video_path.name}：已保存 {written_frames} 帧", flush=True)

            if show_video:
                cv2.imshow("CSRT tracking - press Q to stop", display_frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    stopped_early = True  # 按 Q 结束当前视频，main 仍会处理下一段。
                    break

            ret, original_frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(original_frame, frame_size)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_video or manual_roi:
            cv2.destroyAllWindows()

    print(f"已保存：{output_path}")
    print(f"总帧数 {written_frames}；update 报告成功 {successful_updates} 帧，失败 {failed_updates} 帧。")  # update 成功次数不等于真实定位准确率。
    if stopped_early:
        print("本次按 Q 提前停止，输出视频不是完整时长。")

    return {"frames": written_frames, "successful_updates": successful_updates,
            "failed_updates": failed_updates, "fps": fps, "size": frame_size,
            "stopped_early": stopped_early}


def main():
    for video_name in VIDEO_NAMES:
        input_path = VIDEO_FOLDER / video_name
        output_path = OUTPUT_FOLDER / f"{Path(video_name).stem}_tracked.mp4"
        print(f"开始处理：{input_path}")
        track_video(input_path, output_path, INITIAL_ROIS[video_name],
                    show_video=SHOW_VIDEO, manual_roi=MANUAL_ROI)


if __name__ == "__main__":
    main()
