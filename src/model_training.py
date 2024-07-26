import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the normalized data
X_normalized = pd.read_csv('data/normalized_network_traffic_data.csv').values

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

# Ensure the models directory exists
import os
os.makedirs('models', exist_ok=True)

# Save the trained model
autoencoder.save('models/autoencoder_model.h5')

