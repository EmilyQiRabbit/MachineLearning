# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
# tf.keras 是一种在 TensorFlow 中构建和训练模型的高阶 API。
# 运行方法：python basic-classification.py

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) # 1.13.0-dev20181118

# 可以从 TensorFlow 直接访问 Fashion MNIST，只需导入和加载数据即可

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 加载数据集会返回 4 个 NumPy 数组：
# train_images 和 train_labels 数组是训练集，即模型用于学习的数据。
# test_images 和 test_labels 数组是测试集，用于测试模型。

# 数据预处理：将这些值缩小到 0 到 1 之间，然后将其馈送到神经网络模型。
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

# 构建模型

# 神经网络的基本构造块是层。层从馈送到其中的数据中提取表示结果。希望这些表示结果有助于解决手头问题。

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 第二层该层会返回一个具有 10 个概率得分的数组
# 这些得分的总和为 1。每个节点包含一个得分，表示当前图像属于 10 个类别中某一个的概率。

# 编译模型

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型：调用 model.fit 方法，使模型与训练数据“拟合”。

model.fit(train_images, train_labels, epochs=5)

# 评估准确率

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# 如果机器学习模型在新数据上的表现不如在训练数据上的表现，就表示出现过拟合。

# 做出预测

predictions = model.predict(test_images)

# 预测结果是一个具有 10 个数字的数组。这些数字说明模型对于图像对应于 10 种不同服饰中每一个服饰的“置信度”。


