# House Price Prediction using Support Vector Regression (SVR)
# No CSV file used

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------
# Create Dataset Manually
# ----------------------------------

data = {
    "Area_sqft": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    "Age": [15, 10, 8, 7, 5, 6, 4, 3, 2, 1],
    "Location_Rating": [5, 6, 6, 7, 8, 7, 8, 9, 9, 10],
    "Price": [150000, 200000, 230000, 300000, 350000, 400000, 420000, 500000, 550000, 650000]
}

df = pd.DataFrame(data)

# ----------------------------------
# Split Features and Target
# ----------------------------------

X = df.drop("Price", axis=1)
y = df["Price"]

# ----------------------------------
# Train-Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# Feature Scaling
# ----------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------
# Train Model (SVR Algorithm)
# ----------------------------------

model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# ----------------------------------
# Prediction
# ----------------------------------

y_pred = model.predict(X_test)

# ----------------------------------
# Model Evaluation
# ----------------------------------

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ----------------------------------
# Predict Price for a New House
# ----------------------------------
# Format: [Area_sqft, Bedrooms, Bathrooms, Age, Location_Rating]

new_house = np.array([[2100, 3, 2, 4, 8]])
new_house_scaled = scaler.transform(new_house)

predicted_price = model.predict(new_house_scaled)

print("Predicted House Price:", predicted_price[0])
