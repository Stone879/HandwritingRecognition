import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gzip
import os

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
print("train_images.shape:",train_images.shape)
print("test_images.shape:",test_images.shape)
train_images = np.expand_dims(train_images,-1)
print("train_images.shape",train_images.shape)
test_images = np.expand_dims(test_images,-1)
print("test_images.shape",test_images.shape)

# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',padding='same'))
model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(47,activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(train_images,train_labels,epochs=30,validation_data=(test_images,test_labels))
model.save('./model/Emnist_CNN_2.h5')
print("model have been saved!")
history.history.keys()
plt.plot(history.epoch,history.history.get('loss'),label='loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.legend()
plt.show()
plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()
plt.show()
model.evaluate(test_images,test_labels)
predict = model.predict(test_images)
print(predict[0])
print(np.argmax(predict[0]))
plt.imshow(np.argmax(predict[0]))
plt.show()
print(test_labels[0])

src = cv2.imread('./test_image/imag_0.jpg')
picture = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
picture = picture.reshape(1,28,28)
picture = picture/255.0
test_pred = model.predict(picture)

#预测值
print("预测值：",np.argmax(test_pred))