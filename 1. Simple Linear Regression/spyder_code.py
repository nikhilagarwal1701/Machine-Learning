#Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset  
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#slitting data set into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3, random_state=0)

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set results
Y_pred=regressor.predict(X_test)

#Visualizing the Training Set Results
plt.scatter(X_train, Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Years of Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test Set Results
plt.scatter(X_test, Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Years of Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()








