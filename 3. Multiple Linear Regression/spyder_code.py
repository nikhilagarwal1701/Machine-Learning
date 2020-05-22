#Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import data set
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#Categorical Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
columntransform = ColumnTransformer([("State", OneHotEncoder(), [3])],remainder = 'passthrough')
X = columntransform.fit_transform(X)
X=np.array(X,dtype=float)

#Splitting Dataset into Test and Training Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the Test set Results
Y_pred=regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()

#regressor_ols.summary() 

X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()

regressor_ols.summary() 