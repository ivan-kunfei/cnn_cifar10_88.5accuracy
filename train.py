import tensorflow as tf
import keras
from keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K


def make_new_image(image_origin):
	# Random flip
	image_out = tf.image.random_flip_left_right(image_origin)
	# Random brightness
	image_out = tf.image.random_brightness(image_out, max_delta=0.1)
	# Random contrast
	image_out = tf.image.random_contrast(image_out, lower=0.7, upper=1.3)
	# Random saturation
	image_out = tf.image.random_saturation(image_out, lower=0.7, upper=1.3)
	# Random crop
	image_out = tf.image.random_crop(value=image_out, size=[28, 28, 3])
	image_out = tf.image.resize(image_out, (32, 32))
	image_out = image_out.numpy()
	return image_out


# 50000 train  10000 test
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Image Enhancement
print("=================Image Enhancement Start====================")
length = train_images.shape[0]

new_images = []
new_labels = []
for i in range(length):
	img = train_images[i]
	label = train_labels[i]
	new_img = make_new_image(img)
	new_images.append(new_img)
	new_labels.append(label)
	if i % 1000 == 0:
		print('Image Enhancement： {}'.format(i))

# Concatenate new data and old
new_images = np.array(new_images)
new_labels = np.array(new_labels)
train_images = np.concatenate((train_images, new_images))
train_labels = np.concatenate((train_labels, new_labels))

# Shuffle data
p = np.random.permutation(length * 2)
train_images = train_images[p]
train_labels = train_labels[p]
print('=================图像增强结束=======================')

# Min-Max normalization
train_images, test_images = train_images / 255, test_images / 255

# Plot training data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i])
	plt.xlabel(class_names[train_labels[i][0]])
plt.show()

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

print(model.summary())

file_name = 'models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5'
# Callback function_1: Save model; period=1: Save model for every 1 epoch; verbose=1： print info
checkpoint = callbacks.ModelCheckpoint(filepath=file_name, monitor='val_acc', verbose=1, save_best_only=False,
									   mode='auto', period=1)
# Callback function_2: Lower the learning rate by multiplying 0.2 while val_loss stop decreasing for 5 epochs
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5,
										verbose=1, mode="auto", min_lr=0.0001)


# Callback function_3: Print info
class LrCallBack(callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		lr = self.model.optimizer.lr
		decay = self.model.optimizer.decay
		iterations = self.model.optimizer.iterations
		lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
		print("epoch: {}  当前学习率： {}".format(epoch+1, K.eval(lr_with_decay)))


lr_callback = LrCallBack()

# Callback function list
callbacks_lst = [checkpoint, reduce_lr, lr_callback]


optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer,
			  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
			  metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, batch_size=50,
					validation_data=(test_images, test_labels), shuffle=True, callbacks=callbacks_lst)

print(history.history)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

print(test_acc)
