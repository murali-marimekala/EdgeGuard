import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="autoencoder_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data (use normalized data for inference)
input_data = np.array(X_normalized, dtype=np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(input_data - output_data), axis=1)

# Set a threshold for anomaly detection
threshold = 0.5
anomalies = reconstruction_error > threshold
print("Anomalies detected:", np.sum(anomalies))
