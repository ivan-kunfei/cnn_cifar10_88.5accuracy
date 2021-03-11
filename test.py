
import keras
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt


def load_model(weight_path):
	# Create model
	model = models.Sequential()
	# Convolution   32 channels  (3*3)convolution kernels
	# Number of parameters = 3*3kernels * 3 input channels * 32 output channels + 32bias= 896
	# Initialization of kernel
	model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_initializer='LecunNormal'))
	# Batch normalization
	model.add(layers.BatchNormalization())
	# Relu activation
	model.add(layers.Activation('relu'))

	# Similarly to above
	# Number of parameters = 32 * 3*3  *32 + 32 = 9248
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
	# Max pooling
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
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Dropout(0.5))
	# Flatten
	model.add(layers.Flatten())
	# Dropout to avoid over fitting

	model.add(layers.Dropout(0.5))
	# Soft-max activation to normalize out values
	model.add(layers.Dense(10, activation='softmax'))

	model.compile(optimizer=keras.optimizers.Adam(),
				  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
				  metrics=['accuracy'])

	# Load model weights
	model.load_weights(weight_path)
	print(model.summary())
	return model


def get_test_data(test_num=10000):
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	# Select the data to be tested
	test_images = test_images[0:test_num]
	# Normalization of the input pixels
	test_images = test_images / 255
	test_labels = test_labels[0:test_num].reshape([1, -1])[0]
	return test_images, test_labels

# Get test data
test_images, test_labels = get_test_data()
# Load the model weights
weights_path = 'trained_models/weights.94-0.89.hdf5'
model = load_model(weight_path=weights_path)
# Prediction
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
