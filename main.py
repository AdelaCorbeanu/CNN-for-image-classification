import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (64, 64))  # just to be sure
    image = image / 255.0  # normalize (bring to [0,1])
    return image

train_data = pd.read_csv('train.csv')
train_names = train_data['Image']
train_labels = train_data['Class']
val_data = pd.read_csv('val.csv')
val_image_names = val_data['Image']
val_labels = val_data['Class']

# convert labels to numbers
converter = LabelEncoder()
train_labels = converter.fit_transform(train_labels)
val_labels = converter.transform(val_labels)

# specify layers for model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(96, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# data augmentation (increased variations)
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=30
)

# datasets
train_images = np.array([preprocess_image('train_images/' + x) for x in train_names])
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(len(train_names)).batch(64)
val_images = np.array([preprocess_image('val_images/' + x) for x in val_image_names])
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(64)

# trying to save the best model so far when we reach 15/22/30 epochs
checkpoint_callback_1 = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_1cnn.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch',
    period=15,
    verbose=1
)

checkpoint_callback_2 = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_2cnn.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch',
    period=22,
    verbose=1
)

checkpoint_callback_3 = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_3cnn.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch',
    period=30,
    verbose=1
)

callbacks = [checkpoint_callback_1, checkpoint_callback_2, checkpoint_callback_3]

# the magic
model.fit(datagen.flow(train_images, train_labels), epochs=30, validation_data=val_dataset, callbacks=callbacks)
