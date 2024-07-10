import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Using this tutorial https://victorzhou.com/blog/keras-cnn-tutorial/




def build_model():
    num_filters = 8
    filter_size = 3
    pool_size = 2
    model = models.Sequential([
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=pool_size),
        # Conv2D(num_filters, filter_size, input_shape=(14, 14, 1)),
        # MaxPooling2D(pool_size=pool_size),
        Flatten(),
        # Dense(20, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    return model




if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    assert train_images.shape == (60000, 28, 28) #this is the image data, train contains 60,000 images of 28x28 pixel images. assert checks that the data is actually the dimentionality it's supposed to be
    assert test_images.shape == (10000, 28, 28) #10,000 images in test
    assert train_labels.shape == (60000,) #y is the classification for the images
    assert test_labels.shape == (10000,)

    #TODO: make some sort of visualizer for this




    # Normalize pixel values to be between -0.5 and 0.5. Victor Zhou says using smaller, centered values helps the model learn better- yet to be confirmed elsewhere
    train_images, test_images = (train_images / 255.0) - 0.5, (test_images / 255.0) - 0.5


    #visualize the first 25 images in the dataset
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(train_labels[i])
    plt.show()

    #reshape the array
    train_images, test_images = np.expand_dims(train_images, axis=3), np.expand_dims(test_images, axis=3)


    '''Every Keras model is either built using the Sequential class, which represents a linear stack of layers, 
    or the functional Model class, which is more customizeable. 
    We’ll be using the simpler Sequential model, since our CNN will be a linear stack of layers.'''


    '''The Sequential constructor takes an array of Keras Layers. We’ll use 3 types of layers for our CNN: Convolutional, Max Pooling, and Softmax.'''

    #Defining the architecture of the model


    model = build_model()


    #Configuring the model for training
    model.compile(
        'adam', #Adam gradient based optimizer
        loss='categorical_crossentropy', #look into losses
        metrics=['accuracy'],
    )


    model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=3,
        validation_data=(test_images, to_categorical(test_labels)),
    )

    # Save the model to disk
    model.save_weights('mnist.weights.h5')
    model.export('saved_model')