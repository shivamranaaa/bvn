import numpy as np

# Step 1: Generate synthetic binary classification data
np.random.seed(42)
n_samples = 100

# Class 0: centered at (2, 2)
X0 = np.random.randn(n_samples, 2) + np.array([2, 2])
y0 = np.zeros(n_samples)

# Class 1: centered at (6, 6)
X1 = np.random.randn(n_samples, 2) + np.array([6, 6])
y1 = np.ones(n_samples)

# Combine the data
X = np.vstack((X0, X1))
y = np.concatenate((y0, y1))
