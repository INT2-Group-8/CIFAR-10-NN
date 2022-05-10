# Import TensorFlow, Keras, and matplotlib
import sys
from tabnanny import verbose
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import ssl
from tensorflow import keras
from keras.regularizers import l2
 
# Download the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalise pixel values
#train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_mean = np.mean(train_images, axis=0)
train_std = np.std(train_images, axis=0)
train_images = (train_images - train_mean) / train_std
test_images = (test_images - train_mean) / train_std
  

# Verify the data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i])
  plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Create the convolutional base
# Dropout is used to prevent overfitting
# Batch normalisation normalises the contributions to a layer and decreases the number of epochs required
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, (3, 3),activation='selu', input_shape = (32,32,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, (3, 3), activation='selu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Conv2D(128, (3, 3), activation='selu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256, (3, 3), activation='selu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(256, (3, 3), activation='selu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(512, (3, 3), activation='selu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(512, (3, 3), activation='selu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(1024, (3, 3), activation='selu', kernel_initializer='he_uniform', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.5))

# Flatten the last output tensor into a Dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='selu', kernel_initializer='he_uniform'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(10, activation="softmax"))

# Display a summary of the model
model.summary()

# Setting custom learning rate scheduler
initial_learning_rate = 0.05
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=3124,
    decay_rate=0.988,
    staircase=True)

# Compile and train the model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum = 0.85, nesterov=True), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model
# Aim for val_loss decreasing and val_acc increasing
plt.plot(history.history['val_accuracy'], label='val_accuracy1')
plt.plot(history.history['val_loss'], label='val_loss1')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Print the test accuracy and test error
print("Test Accuracy:", str(round(test_acc * 100)))
print("Test Error:", str(100-round(test_acc * 100)))

# Create augmented data and a generator with batch size 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.4, height_shift_range=0.4, horizontal_flip=True, zoom_range=0.33)
train_generator = data_generator.flow(train_images, train_labels)

# Fit the model to the augmented data
steps = train_images.shape[0] // 32
history = model.fit(train_generator, validation_data=(test_images, test_labels), steps_per_epoch=steps, epochs=600)

# Evaluate the model
# Aim for val_loss decreasing and val_acc increasing
plt.plot(history.history['val_accuracy'], label='val_accuracy2')
plt.plot(history.history['val_loss'], label='val_loss2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Print the test accuracy and test error
print("Test Accuracy:", str(round(test_acc * 100)))
print("Test Error:", str(100-round(test_acc * 100)))