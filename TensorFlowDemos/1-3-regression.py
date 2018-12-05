# -*- coding: utf-8 -*-

# 回归问题：预测的是连续输出值。本篇将会预测房价。
# 而分类问题则是预测离散标签。

# 依旧使用 tf.keras API。他是一种在 TensorFlow 中构建和训练模型的高阶 API。

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# 数据集

# 波士顿房价数据集
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# 这个数据集包含 13 个不同的特征...

# 每个输入数据特征都有不同的范围。
# 一些特征用介于 0 到 1 之间的比例表示，另外一些特征的范围在 1 到 12 之间，还有一些特征的范围在 0 到 100 之间，等等。
# 真实的数据往往都是这样，了解如何探索和清理此类数据是一项需要加以培养的重要技能。

# 可以使用 Pandas 库在格式规范的表格中显示数据集的前几行：

# Pandas: Python Data Analysis Library
# sudo pip install --ignore-installed pandas
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

# 标准化特征：建议标准化使用不同比例和范围的特征。对于每个特征，用原值减去特征的均值，再除以标准偏差即可。

# Test data is *not* used when calculating the mean and std

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized

# 创建模型

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)
  # 均方误差 (MSE) 是用于回归问题的常见损失函数（与分类问题不同）
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# 训练模型

# 对该模型训练 500 个周期，并将训练和验证准确率记录到 history 对象中。

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

# 使用存储在 history 对象中的统计数据可视化模型的训练进度。
# 我们希望根据这些数据判断：对模型训练多长时间之后它会停止优化。

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)

# 在大约 200 个周期之后，模型几乎不再出现任何改进。
# 我们更新一下 model.fit 方法，以便在验证分数不再提高时自动停止训练。
# 我们将使用一个回调来测试每个周期的训练状况。
# 如果模型在一定数量的周期之后没有出现任何改进，则自动停止训练。

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

# 预测

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")

# 其他注意点：

# 如果输入数据特征的值具有不同的范围，则应分别缩放每个特征。
# 如果训练数据不多，则选择隐藏层较少的小型网络，以避免出现过拟合。
# 早停法是防止出现过拟合的实用技术。
# python shell 的退出方式为 exit() || ctrl + z