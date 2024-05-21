import tensorflow as tf
import numpy as np
from PIL import Image


data_dir = '../data'
model_dir = '../models'

model = tf.keras.models.load_model(model_dir + '/model_0.keras')

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((150, 150))
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # normalize pixel values
    image = np.expand_dims(image, axis=0)  # add Batch-Dim.
    return image


input_img = preprocess_image(data_dir + '/right/1715362113949410008.jpg')

prediction = model.predict(input_img)
print(prediction)

predicted_class = np.argmax(prediction, axis=1)
print(predicted_class)
