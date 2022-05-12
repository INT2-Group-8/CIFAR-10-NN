# Import TensorFlow, Keras, and matplotlib
import sys
import time
from datetime import timedelta
from tabnanny import verbose
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import ssl
from tensorflow import keras
from keras.regularizers import l2

# Time the train time
start_time = time.monotonic()

# Download the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalise pixel values
# train_images = train_images.astype('float32')
# test_images = test_images.astype('float32')
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
model.add(keras.layers.Dropout(0.4))

# Flatten the last output tensor into a Dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='selu', kernel_initializer='he_uniform'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))

# Display a summary of the model
model.summary()

# Setting custom learning rate scheduler
initial_learning_rate = 0.05
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=3125,
    decay_rate=0.988,
    staircase=True)

# Compile and train the model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum = 0.85, nesterov=True), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=150, batch_size=64, validation_data=(test_images, test_labels))

# Save accuracy for comparison 
pre_aug = history.history['val_accuracy']

# Evaluate the model
# Aim for val_loss decreasing and val_acc increasing
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Print the test accuracy and test error
print("Test Accuracy:", str(round(test_acc * 100)))
print("Test Error:", str(100-round(test_acc * 100)))

# Create augmented data and a generator
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.33, height_shift_range=0.33, horizontal_flip=True, zoom_range=0.33)
train_generator = data_generator.flow(train_images, train_labels)
train_generator.batch_size = 16
# Fit the model to the augmented data
steps = len(train_images) // 16
history = model.fit(train_generator, validation_data=(test_images, test_labels), steps_per_epoch=steps, validation_steps=len(test_images)//16,epochs=600)

# Evaluate the model post augmentation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Print the test accuracy and test error
print("Test Accuracy:", str(round(test_acc * 100)))
print("Test Error:", str(100-round(test_acc * 100)))

# End timer for training
end_time = time.monotonic()
print("Training time: ", timedelta(seconds=end_time - start_time))

# create subplots 

plot1 = plt.figure(1)
plt.title('Test Accuracy pre and post augmentation')
plt.plot(history.history['val_accuracy'], color = 'orange', label='Post Augmentation')
plt.plot(pre_aug, color ='blue', label = 'Pre Augmentation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')

plot2 = plt.figure(2)
plt.title('Test/Train Accuracies')
plt.plot(history.history['accuracy'], color ='blue', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')

plot3 = plt.figure(3)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='Train Loss')
plt.plot(history.history['val_loss'], color='orange', label='test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')

plot1.show()
plot2.show()
plot3.show()

# Keep the plots alive
input()
