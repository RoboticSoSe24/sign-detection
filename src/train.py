import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras


data_dir = '../data'
model_dir = '../models'

img_size = (96, 80, 3)
batch_size = 32

dataset = keras.utils.image_dataset_from_directory(
    directory=data_dir,
    label_mode='categorical',
    batch_size=batch_size,
    image_size=img_size[:2],
    seed=42,
    validation_split=0.2,
    subset='both')

model = models.Sequential([
    layers.Input(shape=img_size),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(12, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(6, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(
    x=dataset[0],
    validation_data=dataset[1],
    epochs=15)

model.save(model_dir + '/model_0.keras')
