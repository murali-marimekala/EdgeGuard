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
