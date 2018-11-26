# -*- coding: utf-8 -*-

# 依旧是采用 keras 构建和训练模型

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# 下载数据集

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 探索数据

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# Training entries: 25000, labels: 25000

print(train_data[0])

# [1, 14, 22, 16, 43, 530, 973, 1622, 1385...]
# 单词已经被转化为数字

# 准备数据

# 影评（整数数组）必须转换为张量（这里，理解为二维数组即可），然后才能馈送到神经网络中。
# 我们可以通过以下这种方法实现这种转换：

# 填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量。
# 我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层。

word_index = imdb.get_word_index()
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# 经过这样的处理，所有的数组长度都会变为 256

len(train_data[0]), len(train_data[1]) # (256, 256)

# 构建模型

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# 隐藏单元：如果模型具有更多隐藏单元（更高维度的表示空间）和/或更多层，则说明网络可以学习更复杂的表示法。
# 不过，这会使网络耗费更多计算资源，并且可能导致学习不必要的模式（可以优化在训练数据上的表现，但不会优化在测试数据上的表现）。

# 损失函数和优化器

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建验证集

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 评估模型

# 模型会返回两个值：损失（表示误差的数字，越低越好）和准确率。

results = model.evaluate(test_data, test_labels)

print(results)

# 创建准确率和损失随时间变化的图

# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包括训练期间发生的所有情况：

history_dict = history.history
history_dict.keys()

# dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()