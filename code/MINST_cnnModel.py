'''
声明：本程序基于MNIST_CNN的程序，使用其保存的CNN模型，并从本地导入一张图片
实现预测手写数字识别，更改图片需要改图片路径和test_label的值，不过不计算准
确度的话，也不用改label,唯一的问题是需要反色一下，黑底白字
'''

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, datasets

'''
# 这些用不上
#%%
datapath  = r'I:\Pycharm2019\project\project_TF\.idea\data\mnist.npz'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(datapath)
x_test1 = x_test
x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)   #归一化
print(x_test.shape)
print(y_test.shape)

'''

#%%
image_size = 28
x_image = tf.io.read_file('./my_models/number.png')
decode_img = tf.image.decode_png(x_image,channels=1)    # 读取完成后需解码
print(decode_img.shape)     # (28,28,1)
decode_img = tf.image.convert_image_dtype(decode_img, tf.float32)   # 转换为float32格式
test_img = tf.image.resize(decode_img,[image_size,image_size])
test_img = tf.reshape(test_img,[-1,image_size,image_size,1])
test_img = tf.keras.utils.normalize(test_img, axis=1)   # 归一化
test_img = test_img.numpy()     # 需要转换为numpy类型
test_label = np.array([3])      # 需要转换为数组类型
print(decode_img.shape)

# 查看图片
plt.imshow(np.squeeze(decode_img),cmap=plt.cm.binary)   # imshow只能打印出二维图像或第三维度是3和4的三维图像
plt.show()

#%%
# 重载模型
Saved_model = tf.keras.models.load_model('./my_models/MNIST_CNN_model.h5')
#%%
# 预测的权值
predictions = Saved_model.predict(test_img)
print(predictions)
print("预测结果为：{}".format(np.argmax(predictions[0])))


'''
i = 0
plt.imshow(x_test1[i],cmap=plt.cm.binary)
plt.show()

loss, acc = Saved_model.evaluate(test_img,test_label)
print("loss: {:.6f}".format(loss))
print("acc: {:.6f}".format(acc))
'''