import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from CNN_tutorial import build_model


num_filters = 8
filter_size = 3
pool_size = 2




# Build the model.
model = build_model()

# Load the model's saved weights.
model.load_weights('mnist.weights.h5')


num_images = 4

test_images = np.empty((num_images, 28, 28, 1))

for i in range(num_images):
    # Read the image
    image = tf.io.read_file('./testing-imgs/num' + str(i+1) + '.jpg')
    image = tf.image.decode_jpeg(image, channels=1)

    # Resize the image
    image_resized = tf.image.resize(image, [28, 28])

    # Convert to numpy array
    image_array = (image_resized.numpy() / 255.0) - 0.5 


    test_images[i] = image_array












# Predict on the first 5 test images.
predictions = model.predict(test_images)

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) 

# Check our predictions against the ground truths.
