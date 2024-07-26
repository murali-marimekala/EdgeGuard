import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data from the data directory
df = pd.read_csv('data/network_traffic_data.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Save the normalized data for further use
pd.DataFrame(X_normalized, columns=X.columns).to_csv('data/normalized_network_traffic_data.csv', index=False)

