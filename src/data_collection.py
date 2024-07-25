import numpy as np
import pandas as pd
import os

# Generate synthetic network traffic data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 10))
anomalous_data = np.random.normal(loc=5, scale=1, size=(50, 10))

# Combine and create a DataFrame
data = np.vstack([normal_data, anomalous_data])
labels = np.hstack([np.zeros(1000), np.ones(50)])
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(10)])
df['label'] = labels

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV in the data directory
df.to_csv('data/network_traffic_data.csv', index=False)
