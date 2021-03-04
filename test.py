
import keras
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt


def load_model(weight_path):
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

	model.compile(optimizer=keras.optimizers.Adam(),
				  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
				  metrics=['accuracy'])

	# 加载参数
	model.load_weights(weight_path)
	print(model.summary())
	return model


def get_test_data(test_num=10000):
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	# 选取要测试的数据
	test_images = test_images[0:test_num]
	# 将像素的值标准化至0到1的区间内。
	# 数据归一化
	test_images = test_images / 255
	test_labels = test_labels[0:test_num].reshape([1, -1])[0]
	return test_images, test_labels

# 获取测试数据
test_images, test_labels = get_test_data()
# 模型加载权重
weights_path = 'trained_models/weights.94-0.89.hdf5'
model = load_model(weight_path=weights_path)
# 预测
output = model.predict(test_images)
output = np.array(output)
predict_labels = np.argmax(output, axis=-1)
print(predict_labels)
print(test_labels)
result = (predict_labels == test_labels)
acc = np.sum(result) / len(result)
print("================  test accuracy: {}  ================".format(acc))


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
			   'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.gray)
	img = test_images[i]
	img = img[np.newaxis,:]
	out_put = model.predict(img)
	predict_label = class_names[np.argmax(out_put[0])]
	true_label = class_names[test_labels[i]]
	if predict_label == true_label:
		img_name = "(True) This is {}".format(predict_label)
	else:
		img_name = "(False) This is {}".format(predict_label)

	plt.xlabel(img_name)
plt.show()
