# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

# Save the model
with open('autoencoder_model.tflite', 'wb') as f:
    f.write(tflite_model)
