import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Create Sample Bank Loan Dataset
# -----------------------------
data = {
    "Age": [25,35,45,20,35,52,23,40,60,48],
    "Income": [50000,60000,80000,20000,120000,180000,30000,90000,200000,150000],
    "CreditScore": [650,700,720,600,800,850,620,710,900,780],
    "LoanApproved": [0,1,1,0,1,1,0,1,1,1]
}

df = pd.DataFrame(data)

X = df.drop("LoanApproved", axis=1).values
y = df["LoanApproved"].values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Naive Bayes Implementation
# -----------------------------
class NaiveBayes:

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian_probability(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []

            for c in self.classes:
                prior = np.log(self.priors[c])
                class_conditional = np.sum(
                    np.log(self.gaussian_probability(c, x))
                )
                posterior = prior + class_conditional
                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)

# -----------------------------
# Train Model
# -----------------------------
model = NaiveBayes()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy)
