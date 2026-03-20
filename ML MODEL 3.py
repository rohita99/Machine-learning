import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Define dataset (IMPORTANT - this was missing)
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([3, 5, 7, 10])

# Linear Regression
model1 = LinearRegression()
model1.fit(X, y)
y1 = model1.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model2 = LinearRegression()
model2.fit(X_poly, y)
y2 = model2.predict(X_poly)

# MSE Calculation
mse_linear = mean_squared_error(y, y1)
mse_poly = mean_squared_error(y, y2)

print("Linear MSE:", mse_linear)
print("Polynomial MSE:", mse_poly)
