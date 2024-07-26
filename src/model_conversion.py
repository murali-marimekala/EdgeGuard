import tensorflow as tf

# Load the trained model
autoencoder = tf.keras.models.load_model('models/autoencoder_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

# Save the model
with open('models/autoencoder_model.tflite', 'wb') as f:
        f.write(tflite_model)
