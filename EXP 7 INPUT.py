import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Sigmoid function
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Training the model
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Predict probabilities
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    # Predict class labels
    def predict(self, X):
        y_predicted_probs = self.predict_proba(X)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted_probs]
        return np.array(y_predicted_cls)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[0.5, 1.5],
                  [1.0, 1.0],
                  [1.5, 0.5],
                  [3.0, 2.0],
                  [2.0, 3.0],
                  [2.5, 2.5]])

    y = np.array([0, 0, 0, 1, 1, 1])

    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)

    # Predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)
