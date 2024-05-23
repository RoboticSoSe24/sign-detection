import tensorflow as tf
import numpy as np

import cv2



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
    seed=42)


# rescale pixel values to lie in range [0,1]
rescale = tf.keras.layers.Rescaling(scale=1.0/255)
dataset = dataset.map(lambda image,label:(rescale(image),label))


# load trained model
model = tf.keras.models.load_model(model_dir + '/model_0.keras')


# evaluate on entire dataset
eval = model.evaluate(dataset)
print(eval)


# manually walk through images
for element in dataset:
    for i in range(len(element[0])):
        label = np.argmax(element[1][i])
        img = np.array(element[0][i])

        prediction = model.predict(np.expand_dims(img,axis=0))
        predicted_class = np.argmax(prediction, axis=1)

        text = str(label) + '/' + str(predicted_class[0])
        cv2.putText(img, text, (2,12), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 3)
        cv2.putText(img, text, (2,12), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1)

        cv2.imshow('image', img)
        cv2.waitKey(0)
