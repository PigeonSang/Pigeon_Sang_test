from ultralytics import YOLO
import cv2  # 用于图像可视化

# 1. 加载预训练模型（yolov8n.pt：轻量化模型，速度快，适合入门）
model = YOLO('yolov8n.pt')

# 2. 定义测试图像路径（替换为你的图像路径，支持jpg、png等格式）
img_path = 'E:/Users/21081/Desktop/Work/Thesis/test/1.jpg'  # 如Windows：'C:/Users/xxx/Desktop/test.jpg'；Linux：'~/Desktop/test.jpg'

# 3. 执行预测（conf：置信度阈值，默认0.25，低于此值的检测框不显示）
results = model(img_path, conf=0.25)

# 4. 可视化预测结果
# 提取第一个图像的预测结果
result = results[0]
# 绘制检测框、类别、置信度（plot()方法自动绘制）
annotated_img = result.plot()

# 显示结果
cv2.imshow('YOLOv8 Image Detection', annotated_img)
# 等待按键关闭窗口（0表示任意按键关闭）
cv2.waitKey(0)
# 释放窗口资源
cv2.destroyAllWindows()

# 5. 保存预测结果
cv2.imwrite('result.jpg', annotated_img)
print("预测结果已保存为result.jpg")

# 6. 打印详细预测信息（可选，了解检测框坐标、类别、置信度）
for box in result.boxes:
    # 检测框坐标（x1, y1, x2, y2：左上角和右下角坐标）
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    # 置信度
    conf = box.conf[0].item()
    # 类别（0：person，1：bicycle...，可查看ultralytics/data/datasets/coco128.yaml了解所有类别）
    cls = box.cls[0].item()
    cls_name = model.names[cls]
    print(f"类别：{cls_name}，置信度：{conf:.2f}，坐标：({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
