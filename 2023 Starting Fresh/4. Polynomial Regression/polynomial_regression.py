import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

"""plt.scatter(X[:, [1]], y, color="red")
plt.show()"""

# Fitting linear regression to compare
linRegressor = LinearRegression()
linRegressor.fit(X, y)
y_linRegressor = linRegressor.predict(X)

"""plt.scatter(X, y, color="red")
plt.plot(X, y_linRegressor, color="blue")
plt.title("Simple Linear Regression")
plt.show()"""

# Fitting Polynomial Regression Model
polyRegressor = PolynomialFeatures(degree=5)
XPoly = polyRegressor.fit_transform(X)
linRegressor2 = LinearRegression()
linRegressor2.fit(XPoly, y)

XGrid = np.arange(min(X), max(X), 0.1)
XGrid = XGrid.reshape((len(XGrid), 1))
"""plt.plot(XGrid, linRegressor2.predict(polyRegressor.fit_transform(XGrid)), color="green")
plt.show()"""

# Predicting the new result with Linear Regression
"""print("Linear Regression predictor: ", linRegressor.predict(6.5))"""

# Predicting the new result with Polynomial Linear Regression
print("Polynomial Regression predictor: ", linRegressor2.predict(polyRegressor.fit_transform(np.full(1, 1), 6.5)))
