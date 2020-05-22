# Multiple Linear Regression

#importing dataset
dataset=read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State=factor(dataset$State,
                     levels=c('New York','California','Florida'),
                     labels=c(1,2,3))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor=lm(formula = Profit~.,
             data = training_set)

# summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using Backward Elimination
regressor=lm(formula = Profit~R.D.Spend,
             data = training_set)