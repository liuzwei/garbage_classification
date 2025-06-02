import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt


# 加载数据集
DATA_DIR = "D:\\datasets\\Garbage-Classification\\garbage-dataset"
BATCH_SIZE = 32

# 创建训练和验证数据集
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="training",
  seed=123,
  batch_size=BATCH_SIZE)

# print(train_ds)

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  batch_size=BATCH_SIZE)

# 打印数据集信息
# print(val_ds)

# 标准化数据处理
normalization_layer = tf.keras.layers.Rescaling(1./255)
# 应用标准化层到训练和验证数据集
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 构建模型
def create_model():
    model = tf.keras.Sequential([
        # 构建一个垃圾分类的卷积神经网络
        tf.keras.layers.Resizing(128, 128),  # 输入图片尺寸统一
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)  # 10个类别
    ])
    return model

model = create_model()
# 编译模型
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
