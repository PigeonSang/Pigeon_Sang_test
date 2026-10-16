# -*- coding: utf-8 -*-
"""从数据集 labels 找出每张图含 >=2 种植物类别的多品种图，拷贝到 setput。"""
import os
import glob
import shutil

base = r"E:\Users\21081\Documents\GitHub\Pigeon_test\yolo训练\Train_Set\custom_dataset"
setput = r"E:\Users\21081\Documents\GitHub\Pigeon_test\yolo训练\Set\setput"
os.makedirs(setput, exist_ok=True)

found = []
for sp in ["train", "val"]:
    lbl_dir = os.path.join(base, "labels", sp)
    img_dir = os.path.join(base, "images", sp)
    for tf in glob.glob(os.path.join(lbl_dir, "*.txt")):
        stem = os.path.basename(tf)[:-4]
        classes = set()
        with open(tf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        classes.add(parts[0])
        if len(classes) >= 2:  # 一张图至少两种植物
            jpg = os.path.join(img_dir, stem + ".jpg")
            if os.path.exists(jpg):
                found.append((stem, sorted(classes), sp, jpg))

print(f"多品种图共 {len(found)} 张:")
# 拷贝到 setput
import collections
name2label = {0:"白桦",1:"红松",2:"狗尾草",3:"芦苇",4:"蒲公英"}
for i, (stem, cls, sp, jpg) in enumerate(found, 1):
    labels = [name2label.get(int(c), c) for c in cls]
    newname = f"multi_{i:02d}_{'_'.join(labels)}.jpg"
    dst = os.path.join(setput, newname)
    shutil.copy2(jpg, dst)
    print(f"  {newname}  (原 {stem}.jpg, {sp}, 类别: {'/'.join(labels)})")
