# Future Sales Prediction using Linear Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# -----------------------------
# Create Sample Sales Dataset
# -----------------------------
data = {
    "Advertising": [230, 44, 17, 151, 180, 8, 57, 120, 200, 90],
    "SocialMedia": [37, 39, 45, 41, 10, 48, 32, 20, 25, 30],
    "MarketingBudget": [69, 45, 69, 58, 89, 57, 40, 70, 90, 60],
    "Sales": [22, 10, 9, 18, 20, 7, 11, 15, 21, 14]
}

df = pd.DataFrame(data)

# Features and Target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Predict Future Sales
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------
print("Actual Sales:", y_test.values)
print("Predicted Sales:", predictions)

# Accuracy Metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, predictions))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, predictions))

# -----------------------------
# Predict Future Example
# -----------------------------
future_data = [[150, 35, 75]]  # Advertising, SocialMedia, MarketingBudget
future_sales = model.predict(future_data)

print("Predicted Future Sales:", future_sales[0])
