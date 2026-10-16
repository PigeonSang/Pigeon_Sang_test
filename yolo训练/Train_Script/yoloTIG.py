import os
import glob
import cv2
from ultralytics import YOLO


def main():
    # 项目根目录
    project_root = r"E:\Users\21081\Documents\GitHub\Pigeon_test\yolo训练"
    # 自定义数据集配置
    data_yaml = os.path.join(project_root, "Train_Set", "custom_dataset", "data.yaml")
    # 训练输入输出工作目录
    work = os.path.join(project_root, "Train_Script", "runs")
    # 验证集图片目录
    val_img_dir = os.path.join(project_root, "Train_Set", "custom_dataset", "images", "val")
    # =============================

    os.makedirs(work, exist_ok=True)
    os.chdir(work)

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=data_yaml,             # 自定义数据集 data.yaml（5 类，路径已指向当前项目）
        epochs=100,                 # 训练轮数（小数据集建议 100-300，可调）
        batch=16,                   # 批量大小（RTX 4070 8GB 建议 16；显存不足改 8）
        imgsz=640,                  # 输入图像尺寸 640x640
        lr0=0.01,                   # 初始学习率
        device=0,                   # 第 0 块 GPU（-1 用 CPU）
        patience=50,                # 早停：50 轮验证集无提升则停止
        save=True,                  # 保存最佳模型
        project=os.path.join(work, 'plant_train'),   # 输出目录
        name='yolov8n_plant',       # 实验名
        pretrained=True,            # 使用预训练权重（迁移学习）
        cache=True,                 # 缓存数据加速
        exist_ok=True               # 目录已存在复用，避免 -N 后缀
    )

    # 训练后在验证集评估
    val_results = model.val()
    print(f"训练完成！mAP@0.5：{val_results.box.map:.4f}")

    # 用训练好的模型预测（取验证集一张图）
    best_list = sorted(
        glob.glob(os.path.join(work, 'plant_train', '**', 'weights', 'best.pt'), recursive=True),
        key=os.path.getmtime
    )
    best_pt = best_list[-1] if best_list else None
    if not best_pt:
        raise FileNotFoundError("未找到训练产出的 best.pt，请先完成训练")
    print(f"加载训练权重：{best_pt}")
    model = YOLO(best_pt)

    test_img = os.path.join(work, 'plant_test.jpg')
    if not os.path.exists(test_img) and os.path.isdir(val_img_dir):
        files = sorted(os.listdir(val_img_dir))
        test_img = os.path.join(val_img_dir, files[0]) if files else None

    if test_img and os.path.exists(test_img):
        preds = model(test_img, conf=0.25)
        annotated = preds[0].plot()
        out = os.path.join(work, 'plant_result.jpg')
        cv2.imwrite(out, annotated)
        print(f"预测结果已保存：{out}")
    else:
        print("未找到测试图片，已跳过预测步骤")


if __name__ == '__main__':
    main()
