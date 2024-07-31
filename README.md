# EdgeGuard
EdgeGuard is a project designed to detect anomalies in network traffic using machine learning.

---

# EdgeGuard: Real-Time Network Anomaly Detection

EdgeGuard is a project designed to detect anomalies in network traffic using machine learning. This project demonstrates the workflow of gathering data, training a model, and deploying it on a Linux machine for real-time inference.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
EdgeGuard aims to provide a robust solution for real-time network anomaly detection using an autoencoder model. The project includes steps for data collection, preprocessing, model training, and deployment on a Linux machine.

## Features
- **Data Collection**: Simulate or capture network traffic data.
- **Data Preprocessing**: Normalize and prepare data for training.
- **Model Training**: Train an autoencoder model for anomaly detection.
- **Model Conversion**: Convert the trained model to TensorFlow Lite format.
- **Deployment**: Deploy the model on a Linux machine for real-time inference.
- **Anomaly Detection**: Detect anomalies in network traffic in real-time.

## Installation
### Prerequisites
- Python 3.6 or higher
- TensorFlow
- TensorFlow Lite
- Scikit-learn
- Pandas
- NumPy

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/edgeguard.git
   cd edgeguard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Data Collection
Generate synthetic network traffic data:
```python
import numpy as np
import pandas as pd

# Generate synthetic network traffic data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 10))
anomalous_data = np.random.normal(loc=5, scale=1, size=(50, 10))

# Combine and create a DataFrame
data = np.vstack([normal_data, anomalous_data])
labels = np.hstack([np.zeros(1000), np.ones(50)])
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(10)])
df['label'] = labels

# Save to CSV
df.to_csv('network_traffic_data.csv', index=False)
```

### Data Preprocessing
Load and preprocess the data:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('network_traffic_data.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

### Model Training
Train an autoencoder for anomaly detection:
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the autoencoder model
input_dim = X_normalized.shape[1]
encoding_dim = 5

input_layer = layers.Input(shape=(input_dim,))
encoder = layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = layers.Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_normalized, X_normalized, epochs=50, batch_size=32, shuffle=True)
```

### Model Conversion
Convert the trained model to TensorFlow Lite format:
```python
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

# Save the model
with open('autoencoder_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Deployment
Deploy the TensorFlow Lite model on a Linux machine for real-time inference:
```python
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
```
## Execution steps
Follow steps under
https://github.com/murali-marimekala/EdgeGuard/blob/main/QuickStart.md

## Project Structure
```
edgeguard/
├── data/
│   └── network_traffic_data.csv
├── models/
│   └── autoencoder_model.tflite
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_conversion.py
│   └── deployment.py
├── README.md
└── requirements.txt
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README to better fit your project. If you need any more help, just let me know!
