#植物识别推理脚本：加载训练好的模型.pt，处理 setput/ 目录的图片，输出标注图到 output/。

import os
import sys
import glob
import cv2
from ultralytics import YOLO

# 默认输入/输出目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "setput")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def find_best():
    base = r"E:\Users\21081\Documents\GitHub\Pigeon_test\yolo训练\Train_Script\runs"
    bests = sorted(
        glob.glob(os.path.join(base, "**", "weights", "best.pt"), recursive=True),
        key=os.path.getmtime
    )
    return bests[-1] if bests else None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR   # 默认 setput
    dst = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR  # 默认 output

    if not os.path.isdir(src):
        print(f"输入目录不存在或不是文件夹: {src}")
        print(f"请把图片放进: {INPUT_DIR}")
        return

    best = find_best()
    if not best:
        print("未找到 best.pt，请先完成训练")
        return
    model = YOLO(best)
    print(f"加载模型: {best}")
    print(f"类别: {model.names}")

    files = [os.path.join(src, f) for f in sorted(os.listdir(src))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not files:
        print(f"输入目录 {src} 中没有图片")
        return

    os.makedirs(dst, exist_ok=True)
    for f in files:
        try:
            res = model(f, conf=0.25)          #置信度阈值
            annotated = res[0].plot()
            name = os.path.splitext(os.path.basename(f))[0] + "_pred.jpg"
            out = os.path.join(dst, name)
            cv2.imwrite(out, annotated)
            print(f"已保存: {out}")
        except Exception as e:
            print(f"处理失败 {f}: {e}")

    print(f"完成，共处理 {len(files)} 张，输出目录: {dst}")


if __name__ == "__main__":
    main()
