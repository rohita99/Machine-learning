# Mobile Price Prediction using Random Forest

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample mobile dataset
# Features: RAM(GB), Storage(GB), Battery(mAh), Camera(MP)
X = np.array([
    [2, 32, 3000, 8],
    [3, 32, 3200, 12],
    [4, 64, 4000, 16],
    [6, 128, 4500, 48],
    [8, 128, 5000, 64],
    [12, 256, 6000, 108],
    [4, 64, 4200, 20],
    [6, 128, 4800, 32],
    [8, 256, 5200, 64],
    [3, 32, 3000, 13]
])

# Price category
# 0 = Low price
# 1 = Medium price
# 2 = High price
y = np.array([0,0,1,1,2,2,1,2,2,0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Predict price for a new mobile
# Example mobile: 8GB RAM, 128GB Storage, 5000mAh Battery, 64MP Camera
new_mobile = [[8,128,5000,64]]

predicted_price = model.predict(new_mobile)

print("Predicted Price Category:", predicted_price[0])

if predicted_price[0] == 0:
    print("Low Price Mobile")
elif predicted_price[0] == 1:
    print("Medium Price Mobile")
else:
    print("High Price Mobile")
