import os
import glob
import cv2
from ultralytics import YOLO


def main():
    # 工作目录：本次训练的输入输出都放在这里
    work = r"E:\Users\21081\Desktop\test"
    os.makedirs(work, exist_ok=True)
    os.chdir(work)

    # 1. 加载预训练模型（yolov8n.pt，基于COCO数据集预训练，会自动下载到 work 目录）
    model = YOLO('yolov8n.pt')

    # 2. 执行训练（参数详细说明，新手可直接套用）
    results = model.train(
        data='coco128.yaml',                # 官方COCO128数据集配置文件（训练时自动下载到 work/datasets）
        epochs=50,                          # 训练轮数，默认100，新手练手50足够
        batch=16,                           # 批量大小，根据GPU显存调整（8GB显存建议8，若报显存不足请改成8）
        imgsz=640,                          # 输入图像尺寸，默认640x640
        lr0=0.01,                           # 初始学习率，默认0.01
        device=0,                           # 训练设备：0表示第1块GPU，-1表示CPU
        patience=50,                        # 早停耐心值：50轮内验证集性能无提升则停止训练
        save=True,                          # 保存训练过程中的最佳模型
        project=os.path.join(work, 'coco128_train'),  # 项目保存路径（落在 work 下）
        name='yolov8n_coco128',             # 实验名称
        pretrained=True,                    # 使用预训练权重（迁移学习，提升训练效果）
        cache=True,                         # 缓存数据，加速训练
        exist_ok=True                       # 目录已存在时复用，避免自动加 -N 后缀
    )

    # 3. 训练完成后，在验证集上评估模型性能
    val_results = model.val()
    # 打印核心评估指标（mAP@0.5：目标检测核心指标，越高越好）
    print(f"训练完成！mAP@0.5：{val_results.box.map:.4f}")

    # 4. 用训练好的模型预测（测试图优先用 work/test.jpg，不存在则自动从数据集取一张）
    #    动态查找最新生成的 best.pt，避免目录名带 -N 后缀导致找不到
    best_list = sorted(glob.glob(os.path.join(work, 'coco128_train', '**', 'weights', 'best.pt'), recursive=True),
                       key=os.path.getmtime)
    best_pt = best_list[-1] if best_list else None
    if not best_pt:
        raise FileNotFoundError("未找到训练产出的 best.pt，请先完成训练")
    print(f"加载训练权重：{best_pt}")
    model = YOLO(best_pt)
    test_img = os.path.join(work, 'test.jpg')
    if not os.path.exists(test_img):
        data_dir = os.path.join(work, 'datasets', 'coco128', 'images', 'train2017')
        if os.path.isdir(data_dir):
            files = sorted(os.listdir(data_dir))
            test_img = os.path.join(data_dir, files[0]) if files else None
        else:
            test_img = None

    if test_img and os.path.exists(test_img):
        results = model(test_img, conf=0.25)
        annotated_img = results[0].plot()
        out = os.path.join(work, 'coco128_result.jpg')
        cv2.imwrite(out, annotated_img)
        print(f"预测结果已保存为 {out}")
    else:
        print("未找到测试图片（test.jpg 或数据集图片），已跳过预测步骤")


if __name__ == '__main__':
    main()
