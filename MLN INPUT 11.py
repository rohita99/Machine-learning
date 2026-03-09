# Credit Score Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Income': [25000, 40000, 50000, 60000, 70000, 80000],
    'Loan': [5000, 10000, 15000, 20000, 25000, 30000],
    'Credit_History': [1, 1, 0, 1, 0, 1],
    'Credit_Score': ['Poor', 'Average', 'Poor', 'Good', 'Average', 'Good']
}

df = pd.DataFrame(data)

# Features and target
X = df[['Income', 'Loan', 'Credit_History']]
y = df['Credit_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Predicted Credit Score:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
