import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Polynomial Regression Model
polyRegressor = PolynomialFeatures(degree=5)
XPoly = polyRegressor.fit_transform(X)
linRegressor = LinearRegression()
linRegressor.fit(XPoly, y)

# Predicting the new result with Linear Regression
"""print("Linear Regression predictor: ", linRegressor.predict(6.5))"""

"""plt.scatter(X, y, color="red")
plt.plot(X, y_linRegressor, color="blue")
plt.title("Simple Linear Regression")
plt.show()"""
