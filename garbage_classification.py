from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os

class GarbageDetector:
    def __init__(self):
        self.model = None
        self.class_names = [
            'Paper', 'Cardboard', 'Class', 'Plastic', 'Metal', 'Zip-topcan', 'Glassbottles', 'Batteries', 'Fluorescentlighttubes', 'Paintsandsolvents', 'Chemicals', 'Medications', 'Foodscraps', 'Vegetablepeels', 'Fruitpeels', 'Coffeegrounds', 'Tealeaves', 'Planttrimmings', 'SoiledPlastic', 'TornTextiles', 'Ceramics', 'Cigarettebutts', 'Wood', 'Drywall', 'Bricks', 'Concrete', 'Furniture', 'Appliance', 'Mattresses', 'Computer', 'Mobilephones', 'Televisions', 'Blood-soakedgauze', 'Needles', 'MedicalWaste', 'OldCothes', 'Bedding', 'Oldbook', 'Foodbox', 'Plasticbag'
        ]
    
    def train(self, data_yaml_path):
        """
        训练模型
        :param data_yaml_path: 数据配置文件路径
        """
        # 创建YOLO模型
        self.model = YOLO('yolov8s.pt')  # 使用yolov8n架构创建新模型
        
        # 开始训练
        results = self.model.train(
            data=data_yaml_path,
            epochs=50,            # 训练轮数
            imgsz=320,            # 图片尺寸
            batch=16,             # 批次大小
            workers=8,            # 数据加载器的工作进程数
            patience=20,         # 早停策略
            device='0' if torch.cuda.is_available() else 'cpu',  # 使用GPU或CPU
            pretrained=True,        # 使用预训练权重
            optimizer='AdamW',      # 使用AdamW优化器
            lr0=0.003,             # 初始学习率
            weight_decay=0.0003     # 权重衰减
        )
        
    def predict(self, image_paths):
        """
        对多张图片进行预测并在一个画布上用matplotlib展示
        :param image_paths: 图片路径或路径列表
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        # 支持单张图片字符串输入
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        num_images = len(image_paths)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, image_path in enumerate(image_paths):
            img = mpimg.imread(image_path)
            ax = axes[idx]
            ax.imshow(img)
            ax.set_title(os.path.basename(image_path))
            ax.axis('off')

            results = self.model(image_path)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    label = f'{self.class_names[cls]} {conf:.2f}'
                    ax.text(x1, y1 - 5, label, color='g', fontsize=8, backgroundcolor='w')

        # 隐藏多余的子图
        for j in range(idx + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    detector = GarbageDetector()
    # 如果有预训练模型，可以加载
    if os.path.exists('best.pt'):
      detector.model = YOLO('best.pt')
    else:
      # 训练模型
      detector.train("data.yaml")

    # 预测单张或多张图片
    detector.predict([
        "datasets/val/images/TornTextiles48.jpg",
        "datasets/val/images/Batteries18.jpeg",
        "datasets/val/images/Beddingt_9.jpg",
        "datasets/val/images/Blood-soakedgauze21.jpg",
        "datasets/val/images/Bricks13.jpg",
        "datasets/val/images/Ceramics45.jpg",
        "datasets/val/images/Cigarettebutts10.jpg",
        "datasets/val/images/Computer19.jpg",
        "datasets/val/images/Cardboard3.jpg",
        "datasets/val/images/Vegetablepeels19.jpg",
        "datasets/val/images/Tealeaves11.jpg",
        "datasets/val/images/Fluorescentlighttubes6.jpeg",
    ])