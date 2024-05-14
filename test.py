import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model.h5')

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((150, 150))
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # Normalisieren Pixelwerte
    image = np.expand_dims(image, axis=0)  # Hinzufügen Batch-Dim.
    return image


input_img = preprocess_image("/home/ben/code/schilderkennung/data/right/1715362113949410008.jpg")

prediction = model.predict(input_img)
print(prediction)

predicted_class = np.argmax(prediction, axis=1)
print(predicted_class)
