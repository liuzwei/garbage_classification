from ultralytics import YOLO
import torch
import cv2
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
            epochs=30,            # 训练轮数
            imgsz=320,            # 图片尺寸
            batch=16,             # 批次大小
            workers=8,            # 数据加载器的工作进程数
            patience=50,         # 早停策略
            device='0' if torch.cuda.is_available() else 'cpu',  # 使用GPU或CPU
            pretrained=True,        # 使用预训练权重
            optimizer='AdamW',      # 使用AdamW优化器
            lr0=0.001,             # 初始学习率
            weight_decay=0.0005     # 权重衰减
        )
        
    def predict(self, image_path):
        """
        对单张图片进行预测
        :param image_path: 图片路径
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
            
        # 进行预测
        results = self.model(image_path)
        
        # 读取原始图片
        img = cv2.imread(image_path)
        
        # 在图片上绘制预测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取坐标
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 获取分类和置信度
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签文本
                label = f'{self.class_names[cls]} {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img

if __name__ == "__main__":
    detector = GarbageDetector()
    
    # 训练模型
    detector.train("data.yaml")
    
    # 预测单张图片
    result_img = detector.predict("datasets\val\images\Appliance3.jpg")
    
    # 显示结果
    cv2.imshow("Prediction", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()