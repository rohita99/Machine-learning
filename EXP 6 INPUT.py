import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Gaussian Naive Bayes (From Scratch)
# -----------------------------
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian_probability(self, x, mean, var):
        eps = 1e-9  # To avoid division by zero
        coefficient = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
        exponent = np.exp(- (x - mean) ** 2 / (2 * (var + eps)))
        return coefficient * exponent

    def predict(self, X):
        predictions = []
        
        for x in X:
            posteriors = []
            
            for c in self.classes:
                prior = np.log(self.priors[c])
                conditional = np.sum(
                    np.log(self.gaussian_probability(x, self.mean[c], self.var[c]))
                )
                posterior = prior + conditional
                posteriors.append(posterior)
            
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)

# -----------------------------
# Train Model
# -----------------------------
model = GaussianNaiveBayes()
model.fit(X_train, y_train)

# -----------------------------
# Test Model
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy:", accuracy)
