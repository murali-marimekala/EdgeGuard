import numpy as np
import pandas as pd
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="models/autoencoder_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input tensor shape
print("Input tensor shape:", input_details[0]['shape'])

# Load the normalized data
X_normalized = pd.read_csv('data/normalized_network_traffic_data.csv').values

# Ensure the input data matches the expected shape
input_data = np.array(X_normalized, dtype=np.float32)

# Process each sample individually
for sample in input_data:
    sample = np.expand_dims(sample, axis=0)  # Add batch dimension

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], sample)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Calculate reconstruction error
    reconstruction_error = np.mean(np.square(sample - output_data), axis=1)

    # Set a threshold for anomaly detection
    threshold = 0.5
    anomalies = reconstruction_error > threshold
    print("Anomalies detected:", np.sum(anomalies))

