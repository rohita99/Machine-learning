# Car Price Prediction Model without CSV file

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# -------------------------------
# Create Dataset Manually
# -------------------------------

data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2014, 2013, 2020, 2021, 2012],
    "Present_Price": [5.5, 6.0, 7.2, 8.0, 9.5, 4.0, 3.5, 10.0, 11.5, 3.0],
    "Kms_Driven": [40000, 30000, 20000, 15000, 10000, 50000, 60000, 12000, 8000, 70000],
    "Fuel_Type": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],   # 0=Petrol, 1=Diesel
    "Seller_Type": [0, 0, 1, 0, 1, 0, 1, 0, 0, 1], # 0=Dealer, 1=Individual
    "Transmission": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],# 0=Manual, 1=Automatic
    "Owner": [0, 0, 1, 0, 0, 1, 2, 0, 0, 3],
    "Selling_Price": [3.5, 4.0, 5.0, 6.2, 7.0, 2.5, 2.0, 8.5, 9.0, 1.5]
}

df = pd.DataFrame(data)

# -------------------------------
# Split Features and Target
# -------------------------------

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------

print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

# -------------------------------
# Predict New Car Price
# -------------------------------

# Example: Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
new_car = np.array([[2019, 9.0, 15000, 0, 0, 0, 0]])

predicted_price = model.predict(new_car)

print("Predicted Car Selling Price:", predicted_price[0])
