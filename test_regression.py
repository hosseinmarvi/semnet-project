# Code source: Jaques Grobler
# License: BSD 3 clause
"""
===========================================================================
Regression of a dataset with 1-Linear regression,2-Ridge regression
===========================================================================

This example computes:1-Mean squared error, 2-Correlation, 3-Coefficient of
determination, 4-explained_variance_score for a given dataset. The dataset
has coeficceint of 1 in "Input1", and coeficceint of 1e-10 in "input2". The
argument "normalize" in "RidgeCV" needs to be set as "True" for Input2.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import RidgeCV
from sklearn import preprocessing

# Load the diabetes dataset
diabetes_XX, diabetes_y = datasets.load_diabetes(return_X_y=True)

##...................... Input1 ..........................##

# # Use only one feature
# diabetes_X = diabetes_XX[:, np.newaxis, 2]
# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

##...................... Input2 ..........................##

# Use only one feature
diabetes_X = diabetes_XX[:, np.newaxis, 2]*1e-10
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_train2=  preprocessing.scale(diabetes_X_train )
diabetes_X_test = diabetes_X[-20:]
diabetes_X_test2=  preprocessing.scale(diabetes_X_test )

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]*1e-10
diabetes_y_train2=  preprocessing.scale(diabetes_y_train )

diabetes_y_test = diabetes_y[-20:]*1e-10
diabetes_y_test2=  preprocessing.scale(diabetes_y_test )


##.....................................................##

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train2, diabetes_y_train2)
# Make predictions using the testing set with ordinary regression
diabetes_y_pred = regr.predict(diabetes_X_test2)



# Create Ridge linear regression object
regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=True)
# regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=True,cv=10)

# Train the model using the training sets
regrCV.fit(diabetes_X_train2, diabetes_y_train2)
# Make predictions using the testing set with Ridge regression
diabetes_y_pred_CV = regrCV.predict(diabetes_X_test2)



# y=diabetes_X_test*regr.coef_
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Coefficients (RR): \n', regrCV.coef_)

# The mean squared error
print('Mean squared error    : ',
       mean_squared_error(diabetes_y_test2, diabetes_y_pred))


print('Mean squared error(RR): '
      , mean_squared_error(diabetes_y_test2, diabetes_y_pred_CV))


print('Correlation    : '
      , np.corrcoef(diabetes_y_test2, diabetes_y_pred)[0,1])
print('Correlation(RR): '
      , np.corrcoef(diabetes_y_test2, diabetes_y_pred_CV)[0,1])


# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination    : '
      , r2_score(diabetes_y_test2, diabetes_y_pred))

print('Coefficient of determination(RR): '
      , r2_score(diabetes_y_test2, diabetes_y_pred_CV))

print('explained_variance_score    : '
      , explained_variance_score(diabetes_y_test2, diabetes_y_pred))

print('explained_variance_score(RR):'
      , explained_variance_score(diabetes_y_test2, diabetes_y_pred_CV))


# print('regularization factor: %.16f' % regrCV.alpha_)
print('regularization factor: ', regrCV.alpha_)

# Plot outputs
plt.figure()
plt.scatter(diabetes_X_test2, diabetes_y_test2,  color='black')
plt.plot(diabetes_X_test2, diabetes_y_pred, color='blue', linewidth=3)
# plt.plot(diabetes_X_test2, diabetes_y_pred_CV, color='r', linewidth=3)
# plt.plot(diabetes_X_test, y, color='g', linewidth=3)
plt.title('regularization factor: ' + str(regrCV.alpha_))
plt.xticks(())
plt.yticks(())
plt.show()