import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gzip
import os
import datetime

starttime = datetime.datetime.now()

#long running

'''
def load_data(data_folder):

  files = [
      'emnist-balanced-train-labels-idx1-ubyte.gz', 'emnist-balanced-train-images-idx3-ubyte.gz',
      'emnist-balanced-test-labels-idx1-ubyte.gz', 'emnist-balanced-test-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(data_folder,fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)

(train_images, train_labels), (test_images, test_labels) = load_data('./emnist-gzip/gzip/balanced/')


train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images,-1)

test_images = np.expand_dims(test_images,-1)
'''

# 获取GPU列表

Model = tf.keras.models.load_model('./model/Emnist_CNN_2.h5')
#loss, acc = Model.evaluate(test_images, test_labels)
#print("Model, accuracy:{:5.2f}%".format(100 * acc))
#进行预测
src = cv2.imread('./test_image/image_test.jpg')
#测试图片进行预处理
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
picture = np.array(src)
#print(picture.shape)
picture = 255 - picture
picture = picture/255.0
picture = tf.transpose(picture)
picture = np.expand_dims(picture,-1)
#print(picture.shape)
picture = np.expand_dims(picture,0)
print(picture.shape)
test_pred = Model.predict(picture)
#测试1
symbol = np.argmax(test_pred)
if symbol == 0:
    print("预测值为：0")
if symbol == 1:
    print("预测值为：1")
if symbol == 2:
    print("预测值为：2")
if symbol == 3:
    print("预测值为：3")
if symbol == 4:
    print("预测值为：4")
if symbol == 5:
    print("预测值为：5")
if symbol == 6:
    print("预测值为：6")
if symbol == 7:
    print("预测值为：7")
if symbol == 8:
    print("预测值为：8")
if symbol == 9:
    print("预测值为：9")
if symbol == 10:
    print("预测值为：A")
if symbol == 11:
    print("预测值为：B")
if symbol == 12:
    print("预测值为：C")
if symbol == 13:
    print("预测值为：D")
if symbol == 14:
    print("预测值为：E")
if symbol == 15:
    print("预测值为：F")
if symbol == 16:
    print("预测值为：G")
if symbol == 17:
    print("预测值为：H")
if symbol == 18:
    print("预测值为：I")
if symbol == 19:
    print("预测值为：J")
if symbol == 20:
    print("预测值为：K")
if symbol == 21:
    print("预测值为：L")
if symbol == 22:
    print("预测值为：M")
if symbol == 23:
    print("预测值为：N")
if symbol == 24:
    print("预测值为：O")
if symbol == 25:
    print("预测值为：P")
if symbol == 26:
    print("预测值为：Q")
if symbol == 27:
    print("预测值为：R")
if symbol == 28:
    print("预测值为：S")
if symbol == 29:
    print("预测值为：T")
if symbol == 30:
    print("预测值为：U")
if symbol == 31:
    print("预测值为：V")
if symbol == 32:
    print("预测值为：W")
if symbol == 33:
    print("预测值为：X")
if symbol == 34:
    print("预测值为：Y")
if symbol == 35:
    print("预测值为：Z")
if symbol == 36:
    print("预测值为：a")
if symbol == 37:
    print("预测值为：b")
if symbol == 38:
    print("预测值为：d")
if symbol == 39:
    print("预测值为：e")
if symbol == 40:
    print("预测值为：f")
if symbol == 41:
    print("预测值为：g")
if symbol == 42:
    print("预测值为：h")
if symbol == 43:
    print("预测值为：n")
if symbol == 44:
    print("预测值为：q")
if symbol == 45:
    print("预测值为：r")
if symbol == 46:
    print("预测值为：t")

print("该值编号为：",symbol)
endtime = datetime.datetime.now()

print("RunTime: {}".format(endtime-starttime))
# plt.imshow(picture[0])
# plt.show()
