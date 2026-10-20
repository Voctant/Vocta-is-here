"""第四小题的MOG2方法
运行：把 example2.mp4 放在本文件旁边，然后运行本文件。
"""

from pathlib import Path
import math
import sys

import cv2
import numpy as np

MIN_AREA = 150           # 小于这个面积的轮廓当作噪点
WARMUP_FRAMES = 30       # 前 30 帧用于建立背景，预热帧
SHOW_VIDEO = True      # 按q可提前结束

def main():
    folder = Path(__file__).resolve().parent
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "example2.mp4"
    output_path = folder / "number_4_MOG2.mp4"
    if video_path.resolve() == output_path.resolve():
        raise ValueError("输入视频和输出视频不能是同一个文件。")

    cap = cv2.VideoCapture(str(video_path))     #创建视频捕获对象
    if not cap.isOpened():
        raise RuntimeError(f"无法打开，请检查路径：{video_path}")   #检测视频是否成功打开

    # 读取第一帧，确保尺寸
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("视频已打开，但读不到第一帧。")

    height, width = frame.shape[:2]  #获取高，宽数据
    fps = cap.get(cv2.CAP_PROP_FPS)  #获取帧率
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
        print("无法取得有效帧率，暂按 30 FPS 保存。")   #避免帧率过快问题，    QAQ在我的第一次代码中一直生成0秒视频啊QAQ帧率太快了

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")      #指定MP4编码
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))   #视频写入器！
    if not out.isOpened():
        cap.release()
        out.release()
        raise RuntimeError("无法创建输出视频，请检查输出目录及 MP4 编码支持。")

    # history历史长度；varThreshold：前景判断阈值。
    # 去掉阴影，生成二值图！
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36, detectShadows=True)
    kernel_open = np.ones((3, 3), np.uint8)   #去除白噪点
    kernel_close = np.ones((7, 7), np.uint8)     #完整轮廓
    written_frames = 0

    try:
        while ret:   #循环抽取帧
            # 背景是0，疑似阴影是127，运动前景是255
            learning_rate = 0.05 if written_frames < WARMUP_FRAMES else 0.005
            fg_mask = bg_sub.apply(frame, learningRate=learning_rate)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            # 开运算去除孤立小点，闭运算连接物体内部的小缺口
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            fg_mask = cv2.morphologyEx( fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            fg_mask = cv2.dilate(fg_mask, kernel_open, iterations=1)

            # 预热阶段不画框，以免第一帧整幅画面被框住。
            box_count = 0   #数框框
            if written_frames >= WARMUP_FRAMES:    #帧数带到预热之后，开始画框框！
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #二值图中找轮廓
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < MIN_AREA or area > width * height * 0.25:    #计算轮廓面积，过大过小都抛弃！
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)   #画矩形
                    if w < 10 or h < 10:     #忽略过小的目标
                        continue
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    box_count += 1

            text = "MOG2: learning background" if written_frames < WARMUP_FRAMES else f"MOG2: {box_count} regions"
            cv2.putText(frame, text, (20, height - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 将每一帧都写入，防止跳帧   QAQ第一份代码全给我跳了！！！
            out.write(frame)
            written_frames += 1
            if written_frames % 300 == 0:
                print(f"已写入 {written_frames} 帧", flush=True)
            if SHOW_VIDEO:
                cv2.imshow("MOG2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            ret, frame = cap.read()   #读取下一帧，循环
    finally:     #强制收尾，防止报错，同时释放缓存

        # 循环结束后释放缓存
        cap.release()
        out.release()
        if SHOW_VIDEO:
            cv2.destroyAllWindows()

    print(f"输出位置：{output_path}")


if __name__ == "__main__":
    main()
