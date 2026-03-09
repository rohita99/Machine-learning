# Expectation Maximization Algorithm using Gaussian Mixture Model

import numpy as np
from sklearn.mixture import GaussianMixture

# Sample dataset
X = np.array([[1], [2], [3], [10], [11], [12]])

# Create Gaussian Mixture Model with 2 clusters
gmm = GaussianMixture(n_components=2)

# Fit the model
gmm.fit(X)

# Predict cluster for each data point
labels = gmm.predict(X)

# Print results
print("Data Points:")
print(X.flatten())

print("\nCluster Labels:")
print(labels)

print("\nMeans of Clusters:")
print(gmm.means_)
