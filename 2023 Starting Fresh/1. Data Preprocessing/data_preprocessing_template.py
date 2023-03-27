import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Missing Data
simpleImputer = SimpleImputer(missing_values=np.nan, strategy="mean")
simpleImputer = simpleImputer.fit(X[:, 1:])
X[:, 1:] = simpleImputer.transform(X[:, 1:])

# Encoding categorical data
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Divide in column to avoid hierarchy in encoded data
columnTransformer = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')
X = columnTransformer.fit_transform(X)

# Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
