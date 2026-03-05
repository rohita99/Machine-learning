import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Training the model
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Predict
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[1],
                  [2],
                  [3],
                  [4],
                  [5]])

    y = np.array([2, 4, 6, 8, 10])  # y = 2x

    # Train model
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)

    # Predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("Weights:", model.weights)
    print("Bias:", model.bias)
