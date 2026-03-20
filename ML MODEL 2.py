from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([3, 5, 7, 10])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Polynomial Regression")
plt.show()
