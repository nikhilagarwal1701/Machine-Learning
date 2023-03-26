import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

simpleImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
simpleImputer = simpleImputer.fit(X[:, 1:])
X[:, 1:] = simpleImputer.transform(X[:, 1:])
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

