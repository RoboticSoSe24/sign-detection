import tensorflow as tf
from tensorflow.keras import layers, models



data_dir = '../data'
model_dir = '../models'

img_size = (50, 42, 1)

batch_size = 32


# load categorized images from folders and convert to grayscale
dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=img_size[:2],
    seed=42,
    validation_split=0.2,
    subset='both')


# rescale pixel values to lie in range [0,1]
rescale = tf.keras.layers.Rescaling(scale=1.0/255)
dataset[0] = dataset[0].map(lambda image,label:(rescale(image),label))
dataset[1] = dataset[1].map(lambda image,label:(rescale(image),label))


# create CNN model
model = models.Sequential([
    layers.Input(shape=img_size),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(12, (3, 3), activation='relu'),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(6, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
])


# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# fit model to dataset
model.fit(
    x=dataset[0],
    validation_data=dataset[1],
    epochs=5)


# save for further evaluation
model.save(model_dir + '/model_0.keras')
