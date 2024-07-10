import tensorflow as tf


# Convert the SavedModel to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('mnist_model.tflite', 'wb') as f:
    f.write(tflite_model)