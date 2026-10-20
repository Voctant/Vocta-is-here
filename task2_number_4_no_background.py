"""第四小题的不适用背景减除
运行方法和 MOG2那个一样
"""

from pathlib import Path
import math
import sys

import cv2
import numpy as np


MIN_AREA = 150           # 最小轮廓面积
DRAW_START_FRAME = 30
SHOW_VIDEO = True      #按 q可提前结束


def main():
    folder = Path(__file__).resolve().parent
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "example2.mp4"
    output_path = folder / "number_4_no_background.mp4"
    if video_path.resolve() == output_path.resolve():
        raise ValueError("输入视频和输出视频不能是同一个文件。")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频，请检查路径：{video_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("视频已打开，但读不到第一帧。")
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
        print("无法取得有效帧率，暂按 30 FPS 保存。")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        out.release()
        raise RuntimeError("无法创建输出视频，请检查输出目录及 MP4 编码支持。")

    #这里开始改动，我直接把那个背景减除器删了
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    written_frames = 0

    try:
        while ret:
            #GPT老师教我把学习过程删掉，直接生成二值图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            #开运算，闭运算
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            mask = cv2.dilate(mask, kernel_open, iterations=1)

            #不改动画框的方式
            # 没有背景减除时，这些轮廓也可能属于静止的路面或车道线。------GPT说的
            box_count = 0
            if written_frames >= DRAW_START_FRAME:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < MIN_AREA or area > width * height * 0.25:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < 10 or h < 10:
                        continue
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    box_count += 1

            if written_frames < DRAW_START_FRAME:
                text = "No background subtraction: boxes start at frame 30"
            else:
                text = f"No background subtraction: {box_count} regions"
            cv2.putText(frame, text, (20, height - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out.write(frame)
            written_frames += 1
            if written_frames % 300 == 0:
                print(f"已写入 {written_frames} 帧", flush=True)
            if SHOW_VIDEO:
                cv2.imshow("No background subtraction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            ret, frame = cap.read()
    finally:

        cap.release()
        out.release()
        if SHOW_VIDEO:
            cv2.destroyAllWindows()

    print(f"输出位置：{output_path}")


if __name__ == "__main__":
    main()
