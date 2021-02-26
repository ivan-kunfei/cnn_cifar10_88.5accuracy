
import tensorflow as tf
import keras
from keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np


def make_new_image(image_orign):
	# 随机翻转
	image_out = tf.image.random_flip_left_right(image_orign)
	# 随机光照
	image_out = tf.image.random_brightness(image_out, max_delta=0.2)
	# 随机对比度
	image_out = tf.image.random_contrast(image_out, lower=0.8, upper=1.2)
	# 随机饱和度
	image_out = tf.image.random_saturation(image_out, lower=0.8, upper=1.2)
	image_out = image_out.numpy()
	return image_out


# 训练集数据，训练集标签，测试集数据，测试集标签  50000 train  10000 test
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 图像增强
print("=================开始进行图像增强====================")
length = train_images.shape[0]

new_images = []
new_labels = []
for i in range(length):
	img = train_images[i]
	label = train_labels[i]
	new_img = make_new_image(img)
	new_images.append(new_img)
	new_labels.append(label)
	if i % 100 == 0:
		print('图像增强： {}'.format(i))

# 新旧图像合并
new_images = np.array(new_images)
new_labels = np.array(new_labels)
train_images = np.concatenate((train_images, new_images))
train_labels = np.concatenate((train_labels, new_labels))

# 图像shuffle
p = np.random.permutation(length * 2)
train_images = train_images[p]
train_labels = train_labels[p]
print('=================图像增强结束=======================')

# 展示训练数据
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.gray)
	plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255, test_images / 255

# 创建模型
model = models.Sequential()
# 卷积   32通道的 3*3卷积核     param = 3*3卷积核 * 输入3通道 * 输出32通道  + 32偏置= 896
# 卷积核初始化
model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_initializer='LecunNormal'))
# 批标准化
model.add(layers.BatchNormalization())
# Relu 激活函数
model.add(layers.Activation('relu'))

# 同上
# param = 32通道 * 3*3  *32通道 + 32 = 9248
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
# 池化
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128, (1, 1), padding='same', kernel_initializer='LecunNormal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
# 池化
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
# 展开
model.add(layers.Flatten())
# dropout避免过拟合
model.add(layers.Dropout(0.5))
# softmax激活函数 使输出值归一化
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

file_name = 'models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5'
# period=5: 每2个epoch保存一次模型; verbose=1： print info
checkpoint = callbacks.ModelCheckpoint(filepath=file_name, monitor='val_acc', verbose=1, save_best_only=False,
									   mode='auto', period=5)

callbacks_lst = [checkpoint]
model.compile(optimizer=keras.optimizers.Adam(),
			  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, batch_size=50,
					validation_data=(test_images, test_labels), shuffle=True, callbacks=callbacks_lst)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# verbose=2 ： 每个epoch输出一行记录
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
