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

from sklearn.model_selection import train_test_split

# Step 2: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LogisticRegression

# Step 3: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


import matplotlib.pyplot as plt

# Step 5: Plot the test predictions
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr', edgecolor='k', marker='o')
plt.title("Logistic Regression Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
