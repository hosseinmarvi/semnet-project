# Code source: Jaques Grobler
# License: BSD 3 clause
"""
===========================================================================
Regression of a dataset with 1-Linear regression,2-Ridge regression
===========================================================================

This example computes:1-Mean squared error, 2-Correlation, 3-Coefficient of
determination, 4-explained_variance_score for a given dataset. The
argument "normalize" in "RidgeCV" needs to be set as "True" for Input2.
"""
from sklearn.model_selection import cross_validate
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import RidgeCV
from sklearn import preprocessing
import numpy.matlib
from sklearn.model_selection import (train_test_split, KFold)
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
import math
from sklearn.model_selection import LeaveOneOut
# Load the diabetes dataset
diabetes_XX, diabetes_y = datasets.load_diabetes(return_X_y=True)

##...................... Input1 ..........................##
# # choose the size of X and Y
# r, c= [200,100]#[20,5]
# # r, c= [20,10]

# d=str(c)+'x'+str(c)
# # number of iteration
# S=1

# # choose the coefficient matrix
# T1= np.identity(c)

# T2= np.identity(c)
# T2[1,2]=1
# T2[2,1]=-1
# T2[2,3]=2
# T2[3,4]=3
# T2[4,2]=2

# T3= np.identity(c)
# # choose the number of additional elements to be added to indentity matrix
# # e=[10,25] for data of size [20,5]
# # e=[200,1000,5000,10000] for data of size [200,100]
# e=10
# for c in np.random.randint(0,c,[e,2]):
#     T3[c[0],c[1]]= np.random.randint(-5,5,1)[0]

# MSE_snr= np.zeros([11,2])
# corr_snr= np.zeros([11,2])
# r2_snr= np.zeros([11,2])
# var_snr= np.zeros([11,2])
# noise_pow=[0,1,4,9,16,25,36,49,64,81,100]
# T_OLS_t=[0]*11
# T_RR_t=[0]*11
# normalize=True

# f='Linear'
# T= T1

# for j,s in enumerate(np.linspace(0,1,11)):
# # for j,s in enumerate([0.3]):

#     MSE= np.zeros([S,2])
#     corr= np.zeros([S,2])
#     r2= np.zeros([S,2])
#     var= np.zeros([S,2])
#     noise = np.random.normal(0,s,(r,c))
#     T_OLS= np.zeros([c,c])
#     T_RR= np.zeros([c,c])

#     for i in np.arange(0,S):

#         XX= np.random.normal(0,1,(r,c))
#         YY= np.matmul(XX, T)
#         YY= YY+noise
#         # X= XX- np.matlib.repmat(np.mean(XX,0),r,1)
#         # Y= YY- np.matlib.repmat(np.mean(YY,0),r,1)

#         X= XX
#         Y= YY

#         X_train, X_test, Y_train, y_test = train_test_split(
#               X, Y, test_size=0.33)

#         # Create linear regression object
#         # Train the model using the training sets
#         # Make predictions using the testing set with ordinary regression
#         regr = linear_model.LinearRegression()
#         regr.fit(X_train, Y_train)
#         y_pred = regr.predict(X_test)
#         # print(regr.coef_)

#         regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=normalize)
#         regrCV.fit(X_train, Y_train)
#         y_pred_CV = regrCV.predict(X_test)

#         T_OLS= T_OLS+ regr.coef_
#         T_RR= T_RR+ regrCV.coef_

#         MSE[i,0]= mean_squared_error(y_test, y_pred)
#         MSE[i,1]= mean_squared_error(y_test, y_pred_CV)

#         corr[i,0]= np.corrcoef(y_test.transpose(), y_pred.transpose())[0,1]
#         corr[i,1]= np.corrcoef(y_test.transpose(), y_pred_CV.transpose())[0,1]

#         r2[i,0]= r2_score(y_test, y_pred)
#         r2[i,1]= r2_score(y_test, y_pred_CV)

#         var[i,0]= explained_variance_score(y_test, y_pred)
#         var[i,1]= explained_variance_score(y_test, y_pred_CV)
#     # averaging across iteration
#     T_OLS_t[j]=  T_OLS/S
#     T_RR_t[j]=  T_RR/S
#     MSE_snr[j,0]= np.round(np.mean(MSE[:,0]),4)
#     MSE_snr[j,1]= np.round(np.mean(MSE[:,1]),4)
#     corr_snr[j,0]= np.round(np.mean(corr[:,0]),4)
#     corr_snr[j,1]= np.round(np.mean(corr[:,1]),4)
#     r2_snr[j,0]= np.round(np.mean(r2[:,0]),4)
#     r2_snr[j,1]= np.round(np.mean(r2[:,1]),4)
#     var_snr[j,0]= np.round(np.mean(var[:,0]),4)
#     var_snr[j,1]= np.round(np.mean(var[:,1]),4)

# #############################################################
# ##...................... Input2 ..........................##
# f='Nonlinear'
# for j,s in enumerate(np.linspace(0,1,11)):
#     MSE= np.zeros([r,2])
#     corr= np.zeros([r,2])
#     r2= np.zeros([r,2])
#     var= np.zeros([r,2])
#     noise = np.random.normal(0,s,(r,c))
#     T_OLS= np.zeros([c,c])
#     T_RR= np.zeros([c,c])

#     for i in np.arange(0,r):

#         XX= np.random.normal(0,1,(r,c))
#         YY= np.matmul(XX, T1)

#         for n in np.arange(0,r):
#             YY[n,0]= XX[n,0]*np.exp(-np.square(XX[n,1]))- 0.8*XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]+ noise[n,0]

#             YY[n,1]= XX[n,0]*np.exp(-np.square(XX[n,1]))+ XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]*np.exp(-XX[n,0])+ noise[n,1]

#             YY[n,2]= XX[n,0]*np.exp(-np.square(XX[n,1]))- 0.6* XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]+noise[n,2]

#             YY[n,3]= XX[n,0]*np.exp(-np.square(XX[n,3]))+ XX[n,2]+np.square(XX[n,4])+\
#                 XX[n,1]*np.exp(-XX[n,4])+ noise[n,3]

#             YY[n,4]= XX[n,0]*np.exp(-np.square(XX[n,1]))+ 0.9* XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]+ noise[n,4]

#             # YY[n,50]= XX[n,0]*np.exp(-np.square(XX[n,10]))- 0.8*XX[n,20]+np.square(XX[n,30])+\
#             #     XX[n,40]+ XX[n,50]*XX[n,60] + XX[n,70]*np.exp(-np.square(XX[n,80]))- 0.5*XX[n,90]+ noise[n,0]

#             # YY[n,60]= XX[n,0]*np.exp(-np.square(XX[n,10]))+ XX[n,20]+np.square(XX[n,30])+\
#             #     XX[n,40]*np.exp(-XX[n,50])+ XX[n,60]*XX[n,70]+ XX[n,80]*np.exp(-np.square(XX[n,90]))+ noise[n,1]

#             # YY[n,70]= XX[n,0]*np.exp(-np.square(XX[n,10]))- 0.6* XX[n,20]+np.square(XX[n,30])+\
#             #     XX[n,40]+ XX[n,50]*XX[n,60]+ XX[n,70]*np.exp(-np.square(XX[n,80]))- 0.5*XX[n,90]+ noise[n,2]

#             # YY[n,80]= XX[n,10]*np.exp(-np.square(XX[n,30]))+ XX[n,40]+np.square(XX[n,50])+\
#             #     XX[n,60]*np.exp(-XX[n,70])+ XX[n,80]*XX[n,90]+ noise[n,3]

#             # YY[n,90]= XX[n,10]*np.exp(-np.square(XX[n,30]))+ 0.9* XX[n,40]+np.square(XX[n,60])+\
#             #     XX[n,70]+ XX[n,55]*XX[n,80]+ noise[n,90]

#         X=XX- np.matlib.repmat(np.mean(XX,0),r,1)
#         Y=YY- np.matlib.repmat(np.mean(YY,0),r,1)
#         X_train, X_test, Y_train, y_test = train_test_split(
#               X, Y, test_size=0.33)

#         # Create linear regression object
#         # Train the model using the training sets
#         # Make predictions using the testing set with ordinary regression
#         regr = linear_model.LinearRegression()
#         regr.fit(X_train, Y_train)
#         y_pred = regr.predict(X_test)
#         # print(regr.coef_)

#         regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=normalize)
#         regrCV.fit(X_train, Y_train)
#         y_pred_CV = regrCV.predict(X_test)

#         T_OLS= T_OLS+ regr.coef_
#         T_RR= T_RR+ regrCV.coef_

#         MSE[i,0]= mean_squared_error(y_test, y_pred)
#         MSE[i,1]= mean_squared_error(y_test, y_pred_CV)

#         corr[i,0]= np.corrcoef(y_test.transpose(), y_pred.transpose())[0,1]
#         corr[i,1]= np.corrcoef(y_test.transpose(), y_pred_CV.transpose())[0,1]

#         r2[i,0]= r2_score(y_test, y_pred)
#         r2[i,1]= r2_score(y_test, y_pred_CV)

#         var[i,0]= explained_variance_score(y_test, y_pred)
#         varSNR_connectivity)(T,repeat,c,r,std)[i,1]= explained_variance_score(y_test, y_pred_CV)

#     T_OLS_t[j]=  T_OLS/r
#     T_RR_t[SNR_connectivity)(T,repeat,c,r,std)j]=  T_RR/r
#     MSE_snr[j,0]= np.round(np.mean(MSE[:,0]),4)
#     MSE_snr[j,1]= np.round(np.mean(MSE[:,1]),4)
#     corr_snSNR_connectivity)(T,repeat,c,r,std)r[j,0]= np.round(np.mean(corr[:,0]),4)
#     corr_snr[j,1]= np.round(np.mean(corr[:,1]),4)
#     r2_snr[j,0]= np.round(np.mean(r2[:,0]),4)
#     r2_snr[j,1]= np.round(np.mean(r2[:,1]),4)
#     var_snr[j,0]= np.round(np.mean(var[:,0]),4)
#     var_snr[j,1]= np.round(np.mean(var[:,1]),4)


# plt.figure(figsize=(8,4))
# plt.title('MSE'+' - '+f)
# plt.plot(noise_pow,MSE_snr[:,0],'ro',label='OLS')
# plt.plot(noise_pow,MSE_snr[:,0],'r--')
# plt.plot(noise_pow,MSE_snr[:,1],'bo',label='RR')
# plt.plot(noise_pow,MSE_snr[:,1],'b--')
# plt.xlabel('Noise Power')
# plt.legend(loc='upper right')
# plt.savefig('/home/sr05/Method_dev/method_fig/MSE_T7_'+d+'_'+f)

# plt.figure(figsize=(8,4))
# plt.title('Explained Variance'+' - '+f)
# plt.plot(noise_pow,var_snr[:,0],'ro',label='OLS')
# plt.plot(noise_pow,var_snr[:,0],'r--')
# plt.plot(noise_pow,var_snr[:,1],'bo',label='RR')
# plt.plot(noise_pow,var_snr[:,1],'b--')
# plt.xlabel('Noise Power')
# plt.legend(loc='upper right')
# # plt.savefig('/home/sr05/Method_dev/method_fig/Var_T7_'+d+'_'+f)


# plt.figure(figsize=(8,4))
# plt.title('R2_score'+' - '+f)
# plt.plot(noise_pow,r2_snr[:,0],'ro',label='OLS')
# plt.plot(noise_pow,r2_snr[:,0],'r--')
# plt.plot(noise_pow,r2_snr[:,1],'bo',label='RR')
# plt.plot(noise_pow,r2_snr[:,1],'b--')
# plt.xlabel('Noise Power')
# plt.legend(loc='upper right')
# # plt.savefig('/home/sr05/Method_dev/method_fig/R2_T7_'+d+'_'+f)


# plt.figure(figsize=(8,4))
# plt.title('Correlation'+' - '+f)
# plt.plot(noise_pow,corr_snr[:,0],'ro',label='OLS')
# plt.plot(noise_pow,corr_snr[:,0],'r--')
# plt.plot(noise_pow,corr_snr[:,1],'bo',label='RR')
# plt.plot(noise_pow,corr_snr[:,1],'b--')
# plt.xlabel('Noise Power')
# plt.legend(loc='upper right')
# # plt.savefig('/home/sr05/Method_dev/method_fig/Corr_T7_'+d+'_'+f)

# # plt.close('all')


#####################################################################
# noise_H=[np.random.normal(0,s,(200,100)) for s in [.5,.6,.7,.8,.9,1]]
# noise_L=[np.random.normal(0,s,(200,100)) for s in [0,.1,.2,.3,.4]]
# MSE= np.zeros(100)
# var= np.zeros(100)
# r2= np.zeros(100)

# T= np.zeros([100,100])


# H= np.delete(np.arange(0,100),np.arange(40,70))

# for i in np.arange(0,100):
#     XX= np.random.normal(0,1,(200,100))
#     T=np.random.normal(0,1,[100,100])
#     YY= np.matmul(XX, T)

#     if i in H:
#         Noise= noise_H[np.random.randint(0,6)]
#         YY= YY+ Noise
#     if i in np.arange(40,70):
#         Noise= noise_L[np.random.randint(0,3)]
#         YY= YY+ Noise

#     X= XX- np.matlib.repmat(np.mean(XX,0),200,1)
#     Y= YY- np.matlib.repmat(np.mean(YY,0),200,1)
#     X_train, X_test, Y_train, y_test = train_test_split(
#           X, Y, test_size=0.33)

#     regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=normalize)
#     regrCV.fit(X_train, Y_train)
#     y_pred_CV = regrCV.predict(X_test)

#     MSE[i]= mean_squared_error(y_test, y_pred_CV)
#     r2[i]= r2_score(y_test, y_pred_CV)
#     var[i]= explained_variance_score(y_test, y_pred_CV)

# plt.figure(figsize=(8,4))
# plt.plot(MSE,'r')

# plt.figure(figsize=(8,4))
# plt.plot(r2,'b')

# plt.figure(figsize=(8,4))
# plt.plot(var,'g')


######################################################################
# r, c = [200, 100]
# noise_H = [np.random.normal(0, s, (r, c)) for s in [.5, .6, .7, .8, .9, 1]]
# noise_L = [np.random.normal(0, s, (r, c)) for s in [0, .1, .2, .3, .4]]
# MSE = np.zeros([c, 2])
# var = np.zeros([c, 2])
# r2 = np.zeros([c, 2])
# H = np.delete(np.arange(0, 100), np.arange(40, 70))

# for i in np.arange(0, 100):
#     XX = np.random.normal(0, 1, (r, c))
#     T = np.random.normal(0, 1, (c, c))
#     YY = np.matmul(XX, T)
#     noise = noise_L[np.random.randint(0, 3)]

#     if i in H:
#         for n in np.arange(0, r):
#             YY[n, 50] = XX[n, 0]*np.exp(-np.square(XX[n, 10])) - 0.8*XX[n, 20]+np.square(XX[n, 30]) +\
#                 XX[n, 40] + XX[n, 50]*XX[n, 60] + XX[n, 70] * \
#                 np.exp(-np.square(XX[n, 80])) - 0.5*XX[n, 90] + noise[n, 0]

#             YY[n, 60] = XX[n, 0]*np.exp(-np.square(XX[n, 10])) + XX[n, 20]+np.square(XX[n, 30]) +\
#                 XX[n, 40]*np.exp(-XX[n, 50]) + XX[n, 60]*XX[n, 70] + \
#                 XX[n, 80]*np.exp(-np.square(XX[n, 90])) + noise[n, 1]

#             YY[n, 70] = XX[n, 0]*np.exp(-np.square(XX[n, 10])) - 0.6 * XX[n, 20]+np.square(XX[n, 30]) +\
#                 XX[n, 40] + XX[n, 50]*XX[n, 60] + XX[n, 70] * \
#                 np.exp(-np.square(XX[n, 80])) - 0.5*XX[n, 90] + noise[n, 2]

#             YY[n, 80] = XX[n, 10]*np.exp(-np.square(XX[n, 30])) + XX[n, 40]+np.square(XX[n, 50]) +\
#                 XX[n, 60]*np.exp(-XX[n, 70]) + XX[n, 80] * \
#                 XX[n, 90] + noise[n, 3]

#             YY[n, 90] = XX[n, 10]*np.exp(-np.square(XX[n, 30])) + 0.9 * XX[n, 40]+np.square(XX[n, 60]) +\
#                 XX[n, 70] + XX[n, 55]*XX[n, 80] + noise[n, 90]

#     if i in np.arange(40, 70):
#         YY = YY + noise

#     X = XX
#     Y = YY
#     X_train, X_test, Y_train, y_test = train_test_split(
#         X, Y, test_size=0.33)

#     regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100), normalize=True)
#     regrCV.fit(X_train, Y_train)
#     y_pred_CV = regrCV.predict(X_test)

#     regr = MLPRegressor(activation='tanh',  solver='lbfgs',
#                         max_iter=500, hidden_layer_sizes=100).\
#         fit(X_train, Y_train)
#     y_pred = regr.predict(X_test)

#     MSE[i, 0] = mean_squared_error(y_test, y_pred_CV)
#     MSE[i, 1] = mean_squared_error(y_test, y_pred)

#     r2[i, 0] = r2_score(y_test, y_pred_CV)
#     r2[i, 1] = r2_score(y_test, y_pred)

#     var[i, 0] = explained_variance_score(y_test, y_pred_CV)
#     var[i, 1] = explained_variance_score(y_test, y_pred)


# plt.figure(figsize=(8, 4))
# plt.title('MSE')
# plt.plot(MSE[:, 0], 'r')
# plt.plot(MSE[:, 1], 'b')

# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_MSE')

# plt.figure(figsize=(8, 4))
# plt.title('R2 score')
# plt.plot(r2[:, 0], 'r')
# plt.plot(r2[:, 1], 'b')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_R2score')

# plt.figure(figsize=(8, 4))
# plt.title('Explained Variance')
# plt.plot(var[:, 0], 'r')
# plt.plot(var[:, 1], 'b')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_var')

######################################################################
# noise_T=[np.random.normal(0,s,(r,c)) for s in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]]

# MSE= np.zeros(c)
# var= np.zeros(c)
# r2= np.zeros(c)
# corr= np.zeros(c)
# r_error= np.zeros(c)

# # T= np.identity(c)

# H= np.delete(np.arange(0,100),np.arange(40,70))

# for i in np.arange(0,100):
#     print(i)
#     XX= np.random.normal(0,1,(r,c))
#     T= np.random.normal(0,1,(c,c))
#     YY= np.matmul(XX, T)
#     noise= noise_T[np.random.randint(0,7)]
#     # noise= np.random.normal(0,.2,(r,c))

#     if i in np.arange(0,40):

#         for n in np.arange(0,r):
#             YY[n,50]= XX[n,0]*np.exp(-np.square(XX[n,10]))- 0.8*XX[n,20]+np.square(XX[n,30])+\
#                 XX[n,40]+ XX[n,50]*XX[n,60] + XX[n,70]*np.exp(-np.square(XX[n,80]))- 0.5*XX[n,90]+ noise[n,0]

#             YY[n,60]= XX[n,0]*np.exp(-np.square(XX[n,10]))+ XX[n,20]+np.square(XX[n,30])+\
#                 XX[n,40]*np.exp(-XX[n,50])+ XX[n,60]*XX[n,70]+ XX[n,80]*np.exp(-np.square(XX[n,90]))+ noise[n,1]

#             YY[n,70]= XX[n,0]*np.exp(-np.square(XX[n,10]))- 0.6* XX[n,20]+np.square(XX[n,30])+\
#                 XX[n,40]+ XX[n,50]*XX[n,60]+ XX[n,70]*np.exp(-np.square(XX[n,80]))- 0.5*XX[n,90]+ noise[n,2]

#             YY[n,80]= XX[n,10]*np.exp(-np.square(XX[n,30]))+ XX[n,40]+np.square(XX[n,50])+\
#                 XX[n,60]*np.exp(-XX[n,70])+ XX[n,80]*XX[n,90]+ noise[n,3]

#             YY[n,90]= XX[n,10]*np.exp(-np.square(XX[n,30]))+ 0.9* XX[n,40]+np.square(XX[n,60])+\
#                 XX[n,70]+ XX[n,55]*XX[n,80]+ noise[n,90]
#     if i in np.arange(40,70):
#         YY= YY+ noise
#     if i in np.arange(70,100):
#         YY= np.random.normal(0,1,(r,c)) + noise


#     X= XX
#     Y= YY

#     n_splits=10
#     MSE_s=np.zeros([n_splits])
#     var_s=np.zeros([n_splits])
#     r2_s=np.zeros([n_splits])
#     r_error_s=np.zeros([n_splits])
#     corr_s=np.zeros([n_splits])

#     kf = KFold(n_splits=n_splits)
#     for s, (train, test) in enumerate(kf.split(X, Y)):
#         # print(s,train, test)
#         regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=normalize)
#         regrCV.fit(X[train,:], Y[train,:])
#         y_pred_CV = regrCV.predict(X[test,:])

#         MSE_s[s]= mean_squared_error(Y[test,:], y_pred_CV)
#         r2_s[s]= r2_score(Y[test,:], y_pred_CV)
#         var_s[s]= explained_variance_score(Y[test,:], y_pred_CV)
#         corr_s[s]= np.corrcoef(Y[test,:], y_pred_CV)[0,1]
#         r_error_s[s]= np.mean(np.abs((Y[test,:]- y_pred_CV)/Y[test,:]))

#     MSE[i]= np.mean(MSE_s)
#     r2[i]= np.mean(r2_s)
#     var[i]= np.mean(var_s)
#     corr[i]= np.mean(corr_s)
#     r_error[i]= np.mean(r_error_s)

# plt.figure(figsize=(8,4))
# plt.title('MSE')
# plt.plot(MSE,'r')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_MSE')

# plt.figure(figsize=(8,4))
# plt.title('R2 score')
# plt.plot(r2,'b')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_R2score')

# plt.figure(figsize=(8,4))
# plt.title('Explained Variance')
# plt.plot(var,'g')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_var')


# plt.figure(figsize=(8,4))
# plt.title('Correlation')
# plt.plot(corr,'g')
# plt.xlabel('time')


# plt.figure(figsize=(8,4))
# plt.title('Relative erorr')
# plt.plot(r_error,'g')
# plt.xlabel('time')
# # plt.close('all')

# ##############################################################
# noise_T=[np.random.normal(0,s,(r,c)) for s in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]]

# MSE= np.zeros(100)
# var= np.zeros(100)
# r2= np.zeros(100)
# corr= np.zeros(100)
# r_error= np.zeros(100)

# # T= np.identity(c)

# H= np.delete(np.arange(0,100),np.arange(40,70))

# for i in np.arange(0,100):
#     print(i)
#     XX= np.random.normal(0,1,(r,c))
#     T= np.random.normal(0,1,(c,c))
#     YY= np.matmul(XX, T)
#     noise= noise_T[np.random.randint(0,7)]
#     # noise= np.random.normal(0,.2,(r,c))

#     if i in np.arange(0,40):

#         for n in np.arange(0,r):
#             YY[n,0]= XX[n,0]*np.exp(-np.square(XX[n,1]))- 0.8*XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]+ noise[n,0]

#             YY[n,1]= XX[n,0]*np.exp(-np.square(XX[n,1]))+ XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]*np.exp(-XX[n,0])+ noise[n,1]

#             YY[n,2]= XX[n,0]*np.exp(-np.square(XX[n,1]))- 0.6* XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]+ noise[n,2]

#             YY[n,3]= XX[n,0]*np.exp(-np.square(XX[n,1]))+ XX[n,2]+np.square(XX[n,3])+\
#                 XX[n,4]*np.exp(-XX[n,0])+  noise[n,3]

#             YY[n,4]= XX[n,0]*np.exp(-np.square(XX[n,2]))+ 0.9* XX[n,3]+np.square(XX[n,4])+\
#                  noise[n,4]
#     if i in np.arange(40,70):
#         YY= YY+ noise
#     if i in np.arange(70,100):
#         YY= np.random.normal(0,1,(r,c)) + noise


#     X= XX
#     Y= YY

#     n_splits=10
#     MSE_s=np.zeros([n_splits])
#     var_s=np.zeros([n_splits])
#     r2_s=np.zeros([n_splits])
#     r_error_s=np.zeros([n_splits])
#     corr_s=np.zeros([n_splits])

#     kf = KFold(n_splits=n_splits)
#     for s, (train, test) in enumerate(kf.split(X, Y)):
#         # print(s,train, test)
#         regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=normalize)
#         regrCV.fit(X[train,:], Y[train,:])
#         y_pred_CV = regrCV.predict(X[test,:])

#         MSE_s[s]= mean_squared_error(Y[test,:], y_pred_CV)
#         r2_s[s]= r2_score(Y[test,:], y_pred_CV)
#         var_s[s]= explained_variance_score(Y[test,:], y_pred_CV)
#         corr_s[s]= np.corrcoef(Y[test,:], y_pred_CV)[0,1]
#         r_error_s[s]= np.mean(np.abs((Y[test,:]- y_pred_CV)/Y[test,:]))

#     MSE[i]= np.mean(MSE_s)
#     r2[i]= np.mean(r2_s)
#     var[i]= np.mean(var_s)
#     corr[i]= np.mean(corr_s)
#     r_error[i]= np.mean(r_error_s)

# plt.figure(figsize=(8,4))
# plt.title('MSE')
# plt.plot(MSE,'r')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_MSE')

# plt.figure(figsize=(8,4))
# plt.title('R2 score')
# plt.plot(r2,'b')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_R2score')

# plt.figure(figsize=(8,4))
# plt.title('Explained Variance')
# plt.plot(var,'g')
# plt.xlabel('time')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_var')


# plt.figure(figsize=(8,4))
# plt.title('Correlation')
# plt.plot(corr,'g')
# plt.xlabel('time')


# plt.figure(figsize=(8,4))
# plt.title('Relative erorr')
# plt.plot(r_error,'g')
# plt.xlabel('time')
# # plt.close('all')


##############################################################
# connectivity of two random noise with different numbers of trails and vertices
# repeat = 100
# normalize = True
# # R=[10,20,50,100,200,500,1000]
# # R=[10,20,50,100,200,500,1000]
# R = [10, 20, 50, 100, 200, 500, 1000]

# C = [50, 100, 150]
# # C=[50,100,150]
# # MSE= np.zeros(len(R),len(C))
# # var= np.zeros(len(R),len(C))
# # r2= np.zeros(len(R),len(C))
# # corr= np.zeros(len(R),len(C))
# # r_error= np.zeros(len(R),len(C))
# # # T= np.identity(c)


# def noise_connectivity(c, r, repeat):
#     MSE_c = np.zeros([repeat])
#     var_c = np.zeros([repeat])
#     r2_c = np.zeros([repeat])
#     corr_c = np.zeros([repeat])
#     r_error_c = np.zeros([repeat])

#     for i in np.arange(0, repeat):
#         print('r, c , i: ', r, ', ', c, ', ', i)
#         X = np.random.normal(0, 1, (r, c))
#         Y = np.random.normal(0, 1, (r, c))
#         if r >= c:
#             if (r/5) < 10:
#                 n_splits = math.floor(r/5)
#             else:
#                 n_splits = 10
#         else:
#             n_splits = np.max([math.floor(r/5)-1, 2])

#         MSE_s = np.zeros([n_splits])
#         var_s = np.zeros([n_splits])
#         r2_s = np.zeros([n_splits])
#         r_error_s = np.zeros([n_splits])
#         corr_s = np.zeros([n_splits])

#         kf = KFold(n_splits=n_splits)
#         for s, (train, test) in enumerate(kf.split(X, Y)):
#             # print(s,train, test)
#             regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100),
#                              normalize=normalize)
#             regrCV.fit(X[train, :], Y[train, :])
#             y_pred_CV = regrCV.predict(X[test, :])

#             MSE_s[s] = mean_squared_error(Y[test, :], y_pred_CV)
#             r2_s[s] = r2_score(Y[test, :], y_pred_CV)
#             var_s[s] = explained_variance_score(Y[test, :], y_pred_CV)
#             corr_s[s] = np.corrcoef(Y[test, :], y_pred_CV)[0, 1]
#             r_error_s[s] = np.mean(np.abs((Y[test, :] - y_pred_CV)/Y[test, :]))

#         MSE_c[i] = np.mean(MSE_s)
#         r2_c[i] = np.mean(r2_s)
#         var_c[i] = np.mean(var_s)
#         corr_c[i] = np.mean(corr_s)
#         r_error_c[i] = np.mean(r_error_s)

#     MSE = np.mean(MSE_c)
#     r2 = np.mean(r2_c)
#     var = np.mean(var_c)
#     corr = np.mean(corr_c)
#     r_error = np.mean(r_error_c)
#     return {'MSE': MSE, 'r2': r2, 'var': var, 'corr': corr, 'r_error': r_error}


# s = time.time()

# GOF_ave = Parallel(n_jobs=-1)(delayed(noise_connectivity)(c, r, repeat)for c in C for r in R)
# e = time.time()
# print(e-s)

# MSE = np.zeros([len(C), len(R)])
# r2 = np.zeros([len(C), len(R)])
# var = np.zeros([len(C), len(R)])
# corr = np.zeros([len(C), len(R)])
# r_error = np.zeros([len(C), len(R)])

# for i in range(len(C)):
#     for j in range(len(R)):
#         MSE[i, j] = GOF_ave[i*len(R)+j]['MSE']
#         r2[i, j] = GOF_ave[i*len(R)+j]['r2']
#         var[i, j] = GOF_ave[i*len(R)+j]['var']
#         corr[i, j] = GOF_ave[i*len(R)+j]['corr']
#         r_error[i, j] = GOF_ave[i*len(R)+j]['r_error']

# plt.figure(figsize=(8, 4))
# plt.title('MSE')
# plt.plot(R, MSE[0], 'ro', label='50')
# plt.plot(R, MSE[0], 'r--')
# plt.plot(R, MSE[1], 'bs', label='100')
# plt.plot(R, MSE[1], 'b--')
# plt.plot(R, MSE[2], 'gx', label='150')
# plt.plot(R, MSE[2], 'g--')
# plt.legend(loc='upper right')
# plt.xlabel('Number of Trials')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_noise_connectivity_MSE2')

# plt.figure(figsize=(8, 4))
# plt.title('R2 scorre')
# plt.plot(R, r2[0], 'ro', label='50')
# plt.plot(R, r2[0], 'r--')
# plt.plot(R, r2[1], 'bs', label='100')
# plt.plot(R, r2[1], 'b--')
# plt.plot(R, r2[2], 'gx', label='150')
# plt.plot(R, r2[2], 'g--')
# plt.legend(loc='upper right')
# plt.xlabel('Number of Trials')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_noise_connectivity_R22')
# #
# plt.figure(figsize=(8, 4))
# plt.title('Explained Variance')
# plt.plot(R, var[0], 'ro', label='50')
# plt.plot(R, var[0], 'r--')
# plt.plot(R, var[1], 'bs', label='100')
# plt.plot(R, var[1], 'b--')
# plt.plot(R, var[2], 'gx', label='150')
# plt.plot(R, var[2], 'g--')
# plt.legend(loc='upper right')
# plt.xlabel('Number of Trials')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_noise_connectivity_var2')

# plt.figure(figsize=(8, 4))
# plt.title('Correlation')
# plt.plot(R, corr[0], 'ro', label='50')
# plt.plot(R, corr[0], 'r--')
# plt.plot(R, corr[1], 'bs', label='100')
# plt.plot(R, corr[1], 'b--')
# plt.plot(R, corr[2], 'gx', label='150')
# plt.plot(R, corr[2], 'g--')
# plt.legend(loc='upper right')
# plt.xlabel('Number of Trials')
# plt.savefig(
# '/home/sr05/Method_dev/method_fig/simulation_noise_connectivity_corr2')

# plt.figure(figsize=(8, 4))
# plt.title('Relative Error')
# plt.plot(R, r_error[0], 'ro', label='50')
# plt.plot(R, r_error[0], 'r--')
# plt.plot(R, r_error[1], 'bs', label='100')
# plt.plot(R, r_error[1], 'b--')
# plt.plot(R, r_error[2], 'gx', label='150')
# plt.plot(R, r_error[2], 'g--')
# plt.legend(loc='upper right')
# plt.xlabel('Number of Trials')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_noise_connectivity_RE2')

# # # plt.close('all')

###########################################################
# T= np.identity(100)
# T=np.random.normal(0,1,(100,100))
# std_pow=[2.5,2,1.5,1,.5,0,-.5,-1,-1.5,-2]
# y_pow=np.zeros([len(std_pow),100])
# n_pow=np.zeros([len(std_pow),100])
# x_pow=np.zeros([len(std_pow),100])
# for s, std in enumerate(std_pow):
#     noise = np.random.normal(0,10**std,(200,100))

#     for i in np.arange(0,100):
#         print ('s , i: ',s,i)
#         X= np.random.normal(0,1,(200,100))
#         Y= np.matmul(X, T)
#         Y= Y+noise
#         y_pow[s,i]=np.var(Y)
#         n_pow[s,i]=np.var(noise)
#         x_pow[s,i]=np.var(np.matmul(X, T))
# sig=x_pow.copy().mean(1)
# n=n_pow.copy().mean(1)
# snr=sig/n
############################################################
# SNR Effect on model performance
# r, c= [200,100]

# # number of iteration
# repeat=100

# # choose the coefficient matrix
# T1= np.identity(c)
# std_pow=[2.5,2,1.5,1,.5,0,-.5,-1,-1.5,-2]
# # snr=np.arange(-40,50,10)
# # SNR=snr.copy().transpose()
# normalize=True
# n_pow=np.zeros([repeat,1])
# x_pow=np.zeros([repeat,1])
# f='Linear'
# Ts= [T1 ,np.random.normal(0,1,(c,c))]

# def SNR_connectivity(T,std,repeat,r,c):
#     MSE= np.zeros([repeat,1])
#     var= np.zeros([repeat,1])
#     r2= np.zeros([repeat,1])
#     corr= np.zeros([repeat,1])
#     r_error= np.zeros([repeat,1])

#     noise = np.random.normal(0,10**std,(r,c))

#     for i in np.arange(0,repeat):
#         print ('r, c , i: ',r,', ', c,', ',i)
#         X= np.random.normal(0,1,(r,c))
#         Y= np.matmul(X, T)
#         Y= Y+noise
#         n_pow[i,0]=np.var(noise)
#         x_pow[i,0]=np.var(np.matmul(X, T))
#         if r<=10:
#             n_splits=2
#         else:
#             n_splits=10

#         MSE_s=np.zeros([n_splits])
#         var_s=np.zeros([n_splits])
#         r2_s=np.zeros([n_splits])
#         r_error_s=np.zeros([n_splits])
#         corr_s=np.zeros([n_splits])

#         kf = KFold(n_splits=n_splits)
#         for s, (train, test) in enumerate(kf.split(X, Y)):
#             # print(s,train, test)
#             regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=normalize)
#             regrCV.fit(X[train,:], Y[train,:])
#             y_pred_CV = regrCV.predict(X[test,:])

#             MSE_s[s]= mean_squared_error(Y[test,:], y_pred_CV)
#             r2_s[s]= r2_score(Y[test,:], y_pred_CV)
#             var_s[s]= explained_variance_score(Y[test,:], y_pred_CV)
#             corr_s[s]= np.corrcoef(Y[test,:], y_pred_CV)[0,1]
#             r_error_s[s]= np.mean(np.abs((Y[test,:]- y_pred_CV)/Y[test,:]))

#         MSE[i]= np.mean(MSE_s)
#         r2[i]= np.mean(r2_s)
#         var[i]= np.mean(var_s)
#         corr[i]= np.mean(corr_s)
#         r_error[i]= np.mean(r_error_s)

#     sig=x_pow.copy().mean(0)
#     n=n_pow.copy().mean(0)
#     snr=sig/n
#     # averaging across iteration
#     # T_RR_t[j]=  T_RR/repeat
#     MSE_snr= np.round(np.mean(MSE[:,0]),4)
#     corr_snr= np.round(np.mean(corr[:,0]),4)
#     r2_snr= np.round(np.mean(r2[:,0]),4)
#     var_snr= np.round(np.mean(var[:,0]),4)
#     r_error_snr= np.round(np.mean(r_error[:,0]),4)
#     return {'MSE':MSE_snr,'r2':r2_snr, 'var':var_snr, 'corr':corr_snr,\
#             'r_error':r_error_snr,'sig_pow':sig,'n_pow':n,'snr':snr}

# s=time.time()
# GOF_ave=Parallel(n_jobs=-1)(delayed(SNR_connectivity)(T,std,repeat,r,c)for T in Ts\
#                             for std in std_pow)
# e=time.time()
# print(e-s)


# MSE=np.zeros([2,len(std_pow)])
# r2=np.zeros([2,len(std_pow)])
# var=np.zeros([2,len(std_pow)])
# corr=np.zeros([2,len(std_pow)])
# r_error=np.zeros([2,len(std_pow)])
# snr=np.zeros([2,len(std_pow)])

# for i in range(2):
#     for j in range(len(std_pow)):
#         MSE[i,j]=GOF_ave[i*len(std_pow)+j]['MSE']
#         r2[i,j]=GOF_ave[i*len(std_pow)+j]['r2']
#         var[i,j]=GOF_ave[i*len(std_pow)+j]['var']
#         corr[i,j]=GOF_ave[i*len(std_pow)+j]['corr']
#         r_error[i,j]=GOF_ave[i*len(std_pow)+j]['r_error']
#         snr[i,j]=GOF_ave[i*len(std_pow)+j]['snr']
# # plt.rcParams['font.size'] = '12'

# plt.figure(figsize=(8,4))
# plt.title('MSE')
# plt.plot(10*np.log10(snr[0,:]),MSE[0,:],'ro',label='T: Identity')
# plt.plot(10*np.log10(snr[0,:]),MSE[0,:],'r--')
# plt.plot(10*np.log10(snr[1,:]),MSE[1,:],'bo',label='T: Random')
# plt.plot(10*np.log10(snr[1,:]),MSE[1,:],'b--')
# plt.xlabel('SNR(db)')
# plt.legend(loc='upper right')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNReffect_MSE')

# plt.figure(figsize=(8,4))
# plt.title('Correlation')
# plt.plot(10*np.log10(snr[0,:]),corr[0,:],'ro',label='T: Identity')
# plt.plot(10*np.log10(snr[0,:]),corr[0,:],'r--')
# plt.plot(10*np.log10(snr[1,:]),corr[1,:],'bo',label='T: Random')
# plt.plot(10*np.log10(snr[1,:]),corr[1,:],'b--')
# plt.xlabel('SNR(db)')
# plt.legend(loc='upper right')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNReffect_corr')

# plt.figure(figsize=(8,4))
# plt.title('R2 score')
# plt.plot(10*np.log10(snr[0,:]),r2[0,:],'ro',label='T: Identity')
# plt.plot(10*np.log10(snr[0,:]),r2[0,:],'r--')
# plt.plot(10*np.log10(snr[1,:]),r2[1,:],'bo',label='T: Random')
# plt.plot(10*np.log10(snr[1,:]),r2[1,:],'b--')
# plt.xlabel('SNR(db)')
# plt.legend(loc='upper right')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNReffect_R2')

# plt.figure(figsize=(8,4))
# plt.title('Explained Variance')
# plt.plot(10*np.log10(snr[0,:]),var[0,:],'ro',label='T: Identity')
# plt.plot(10*np.log10(snr[0,:]),var[0,:],'r--')
# plt.plot(10*np.log10(snr[1,:]),var[1,:],'bo',label='T: Random')
# plt.plot(10*np.log10(snr[1,:]),var[1,:],'b--')
# plt.xlabel('SNR(db)')
# plt.legend(loc='upper right')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNReffect_var')

# plt.figure(figsize=(8,4))
# plt.title('Relative Error')
# plt.plot(10*np.log10(snr[0,:]),r_error[0,:],'ro',label='T: Identity')
# plt.plot(10*np.log10(snr[0,:]),r_error[0,:],'r--')
# plt.plot(10*np.log10(snr[1,:]),r_error[1,:],'bo',label='T: Random')
# plt.plot(10*np.log10(snr[1,:]),r_error[1,:],'b--')
# plt.xlabel('SNR(db)')
# plt.legend(loc='upper right')
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNReffect_re')

# # # plt.close('all')

############################################################
# ### SNR and trials Effect on model performance
# c = 50
# # R = [40]
# R = [40, 50, 100, 150, 300, 500, 1000]

# # number of iteration
# repeat = 100

# # choose the coefficient matrix
# std_pow = [2.5, 2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2]
# # std_pow=[0,-.5,-1,-1.5,-2]

# # snr=np.arange(-40,50,10)
# # snr=np.arange(0,50,10)

# # SNR=snr.copy().transpose()
# # MSE_snr= np.zeros([len(std_pow),len(R)])
# # corr_snr= np.zeros([len(std_pow),len(R)])
# # r2_snr= np.zeros([len(std_pow),len(R)])
# # var_snr= np.zeros([len(std_pow),len(R)])
# # r_error_snr= np.zeros([len(std_pow),len(R)])
# n_pow = np.zeros([repeat, 1])
# x_pow = np.zeros([repeat, 1])
# normalize = True


# def SNR_trials_connectivity(repeat, r, c, std):
#     MSE = np.zeros([repeat, 1])
#     var = np.zeros([repeat, 1])
#     r2 = np.zeros([repeat, 1])
#     corr = np.zeros([repeat, 1])
#     r_error = np.zeros([repeat, 1])

#     noise = np.random.normal(0, 10**std, (r, c))
#     T = np.random.normal(0, 1, (c, c))

#     for i in np.arange(0, repeat):
#         # print ('r, c , i: ',r,', ', c,', ',i)
#         X = np.random.normal(0, 1, (r, c))
#         Y = np.matmul(X, T)
#         Y = Y+noise
#         n_pow[i, 0] = np.var(noise)
#         x_pow[i, 0] = np.var(np.matmul(X, T))

#         if r >= c:
#             if (r/5) < 10:
#                 n_splits = math.floor(r/5)
#             else:
#                 n_splits = 10
#         else:
#             n_splits = np.max([math.floor(r/5)-1, 2])

#         MSE_s = np.zeros([n_splits])
#         var_s = np.zeros([n_splits])
#         r2_s = np.zeros([n_splits])
#         r_error_s = np.zeros([n_splits])
#         corr_s = np.zeros([n_splits])

#         kf = KFold(n_splits=n_splits)
#         for s, (train, test) in enumerate(kf.split(X, Y)):
#             # print(s,train, test)
#             regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100),
#                              normalize=normalize)
#             regrCV.fit(X[train, :], Y[train, :])
#             y_pred_CV = regrCV.predict(X[test, :])

#             MSE_s[s] = mean_squared_error(Y[test, :], y_pred_CV)
#             r2_s[s] = r2_score(Y[test, :], y_pred_CV)
#             var_s[s] = explained_variance_score(Y[test, :], y_pred_CV)
#             corr_s[s] = np.corrcoef(Y[test, :], y_pred_CV)[0, 1]
#             r_error_s[s] = np.mean(np.abs((Y[test, :] - y_pred_CV)/Y[test, :]))

#         MSE[i] = np.mean(MSE_s)
#         r2[i] = np.mean(r2_s)
#         var[i] = np.mean(var_s)
#         corr[i] = np.mean(corr_s)
#         r_error[i] = np.mean(r_error_s)

#     sig = x_pow.copy().mean(0)
#     n = n_pow.copy().mean(0)
#     snr = sig/n
#     # averaging across iteration
#     # T_RR_t[j]=  T_RR/repeat
#     MSE_snr = np.round(np.mean(MSE[:, 0]), 4)
#     corr_snr = np.round(np.mean(corr[:, 0]), 4)
#     r2_snr = np.round(np.mean(r2[:, 0]), 4)
#     var_snr = np.round(np.mean(var[:, 0]), 4)
#     r_error_snr = np.round(np.mean(r_error[:, 0]), 4)
#     return {'MSE': MSE_snr, 'r2': r2_snr, 'var': var_snr, 'corr': corr_snr,
#             'r_error': r_error_snr, 'sig_pow': sig, 'n_pow': n, 'snr': snr}


# s = time.time()
# GOF_ave = Parallel(n_jobs=-1)(delayed(SNR_trials_connectivity)(repeat, r, c, std)
#                               for r in R
#                               for std in std_pow)
# e = time.time()
# print(e-s)


# MSE = np.zeros([len(R), len(std_pow)])
# r2 = np.zeros([len(R), len(std_pow)])
# var = np.zeros([len(R), len(std_pow)])
# corr = np.zeros([len(R), len(std_pow)])
# r_error = np.zeros([len(R), len(std_pow)])
# snr = np.zeros([len(R), len(std_pow)])
# for i in range(len(R)):
#     for j in range(len(std_pow)):
#         MSE[i, j] = GOF_ave[i*len(std_pow)+j]['MSE']
#         r2[i, j] = GOF_ave[i*len(std_pow)+j]['r2']
#         var[i, j] = GOF_ave[i*len(std_pow)+j]['var']
#         corr[i, j] = GOF_ave[i*len(std_pow)+j]['corr']
#         r_error[i, j] = GOF_ave[i*len(std_pow)+j]['r_error']
#         snr[i, j] = GOF_ave[i*len(std_pow)+j]['snr']

# SNR = 10*np.log10(np.mean(snr, 0))
# plt.rcParams['font.size'] = '12'
# my_colors = ['bo', 'b--', 'y*', 'y--', 'go', 'g--',
#              'ms', 'm--', 'bs', 'b--', 'rs', 'r--', 'gs', 'g--']
# my_labels = ['40', '50', '100', '150', '300', '500', '1000']

# # p = 0
# # plt.figure(figsize=(8, 4))
# # for i in np.arange(0, len(R)):
# #     plt.title('MSE')
# #     plt.plot(SNR, MSE[p, :], my_colors[i+p], label=my_labels[i])
# #     plt.plot(SNR, MSE[p, :], my_colors[i+1+p])
# #     p += 1
# #     plt.legend(loc='upper right')

# # plt.xlabel('SNR(db)')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNR_trials_effect_MSE')

# # p = 0
# # plt.figure(figsize=(8, 4))
# # for i in np.arange(0, len(R)):
# #     plt.title('Relative Error')
# #     plt.plot(SNR, r_error[p, :], my_colors[i+p], label=my_labels[i])
# #     plt.plot(SNR, r_error[p, :], my_colors[i+1+p])
# #     p += 1
# #     plt.legend(loc='upper right')

# # plt.xlabel('SNR(db)')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNR_trials_effect_re')


# # p = 0
# # plt.figure(figsize=(8, 4))
# # for i in np.arange(0, len(R)):
# #     plt.title('R2 score')
# #     plt.plot(SNR, r2[p, :], my_colors[i+p], label=my_labels[i])
# #     plt.plot(SNR, r2[p, :], my_colors[i+1+p])
# #     p += 1
# #     plt.legend(loc='upper right')

# # plt.xlabel('SNR(db)')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNR_trials_effect_r2')


# p = 0
# plt.figure(figsize=(8, 4))
# for i in np.arange(0, len(R)):
#     plt.title('Explained Variance')
#     plt.plot(SNR, var[p, :], my_colors[i+p], label=my_labels[i])
#     plt.plot(SNR, var[p, :], my_colors[i+1+p])
#     p += 1
#     plt.legend(loc='upper left')

# plt.xlabel('SNR(db)')
# plt.savefig(
#     '/home/sr05/Method_dev/method_fig/simulation_SNR_trials_effect_var-50voxels')


# # p = 0
# # plt.figure(figsize=(8, 4))
# # for i in np.arange(0, len(R)):
# #     plt.title('PCC')
# #     plt.plot(SNR, corr[p, :], my_colors[i+p], label=my_labels[i])
# #     plt.plot(SNR, corr[p, :], my_colors[i+1+p])
# #     p += 1
# #     plt.legend(loc='upper right')

# # plt.xlabel('SNR(db)')
# # plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNR_trials_effect_corr')
# # plt.close('all')
#################################################################
# # repeat=10
# # noise=np.random.normal(0,.5,(200,140))
# # var= np.zeros([repeat,2])
# # r2= np.zeros([repeat,2])
# # for i in np.arange(0,repeat):
# #       print ('i: ',i)
# #       X= np.random.normal(0,1,(200,100))
# #       T= np.random.normal(0,1,(100,140))
# #       Y= np.matmul(X, T)
# #       Y= Y+noise

# #       X_uv=(X.copy().mean(1)).reshape(200,1)
# #       Y_uv=(Y.copy().mean(1)).reshape(200,1)
# #       T_uv=np.ones([1,140])
# #       n_splits=10

# #       var_s=np.zeros([n_splits,2])
# #       r2_s=np.zeros([n_splits,2])

# #       kf = KFold(n_splits=n_splits)
# #       for s, (train, test) in enumerate(kf.split(X, Y)):
# #           # print(s,train, test)
# #           regrCV = RidgeCV(alphas=np.logspace(-10,10,100),normalize=True)
# #           regrCV.fit(X[train,:], Y[train,:])
# #           y_pred_CV = regrCV.predict(X[test,:])

# #           r2_s[s,0]= r2_score(Y[test,:], y_pred_CV)
# #           var_s[s,0]= explained_variance_score(Y[test,:], y_pred_CV)

# #           regr = RidgeCV(alphas=np.logspace(-10,10,100),normalize=True)
# #           regr.fit(X_uv[train], Y_uv[train])
# #           y_pred_CV = regr.predict(X_uv[test])
# #           y_pred= np.matmul(y_pred_CV,T_uv)

# #           r2_s[s,1]= r2_score(Y[test,:], y_pred)
# #           var_s[s,1]= explained_variance_score(Y_uv[test], y_pred_CV)


# #       r2[i,0]= np.mean(r2_s[:,0])
# #       var[i,0]= np.mean(var_s[:,0])

# #       r2[i,1]= np.mean(r2_s[:,1])
# #       var[i,1]= np.mean(var_s[:,1])

# ###############################################################################
# ## Effect of K in K-fold on explained variance as a function of SNRs
# r, c = [40, 20]
# plt.close('all')
# repeat = 10
# # std_pow = [2.5, 2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2]
# std_pow = [0, -.5, -1, -1.5, -2]

# NS = [2,5,8,10,40]

# for ns in NS:

#     snr = np.zeros([len(std_pow), 1])
#     var_snr = np.zeros([len(std_pow), 1])

#     for st, std in enumerate(std_pow):
#         print(ns, st)
#         noise = np.random.normal(0, 10**std, (r, c))
#         T = np.random.normal(0, 1, (c, c))
#         n_pow = np.zeros([repeat, 1])
#         x_pow = np.zeros([repeat, 1])
#         var = np.zeros([repeat, 1])
#         r2 = np.zeros([repeat, 1])
#         for i in np.arange(0, repeat):
#             # print ('r, c , i: ',r,', ', c,', ',i)
#             X = np.random.normal(0, 1, (r, c))
#             Y = np.matmul(X, T)
#             Y = Y+noise
#             n_pow[i, 0] = np.var(noise)
#             x_pow[i, 0] = np.var(np.matmul(X, T))

#             if r <= 10:
#                 n_splits = 2
#             else:
#                 n_splits = ns

#             var_s = np.zeros([n_splits])

#             kf = KFold(n_splits=n_splits)
#             for s, (train, test) in enumerate(kf.split(X, Y)):
#                 # print(s,train, test)
#                 regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100),
#                                   normalize=True)
#                 regrCV.fit(X[train, :], Y[train, :])
#                 y_pred_CV = regrCV.predict(X[test, :])
#                 var_s[s] = explained_variance_score(Y[test, :], y_pred_CV)

#             # r2[i] = np.mean(r2_s)
#             var[i] = np.mean(var_s)

#         sig = x_pow.copy().mean(0)
#         n = n_pow.copy().mean(0)
#         snr[st] = sig/n
#         var_snr[st] = np.round(np.mean(var[:, 0]), 4)
#     SNR = 10*np.log10(snr)
#     # r2_snr = np.round(np.mean(r2[:, 0]), 4)
#     print('snr: ', snr, '   / var: ', var_snr)
#     plt.figure()
#     plt.plot(SNR, var_snr,'bs')
#     plt.plot(SNR, var_snr,'b--')
#     plt.title(str(ns)+ '-fold cross validation')
#     plt.xlabel('SNR(db)')
#     plt.ylabel('Explained Variance')


# # plt.close('all')


###############################################################################
# # Effect of K in K-fold on explained variance as a function of SNRs
# r, c = [40, 20]
# # plt.close('all')
# repeat = 50
# # std_pow = [2.5, 2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2]
# std_pow = [-.5]

# K = np.array([2, 5, 8, 10, 20, r-1])

# for std in std_pow:

#     snr = np.zeros([len(std_pow), 1])
#     var_k = np.zeros([len(K), 1])
#     mse_k = np.zeros([len(K), 1])
#     mse_std = np.zeros([len(K), 1])
#     noise = np.random.normal(0, 10**std, (r, c))
#     T = np.random.normal(0, 1, (c, c))
#     n_pow = np.zeros([repeat, 1])
#     x_pow = np.zeros([repeat, 1])

#     for j, k in enumerate(K):
#         print(k)
#         var = np.zeros([repeat, 1])
#         mse = np.zeros([repeat, 1])
#         for i in np.arange(0, repeat):
#             # print ('r, c , i: ',r,', ', c,', ',i)
#             X = np.random.normal(0, 1, (r, c))
#             Y = np.matmul(X, T)
#             Y = Y+noise
#             n_pow[i, 0] = np.var(noise)
#             x_pow[i, 0] = np.var(np.matmul(X, T))
#             n_splits = k

#             var_s = np.zeros([n_splits])
#             mse_s = np.zeros([n_splits])

#             kf = KFold(n_splits=n_splits)
#             for s, (train, test) in enumerate(kf.split(X, Y)):
#                 # print(s,train, test)
#                 regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100),
#                                  normalize=True)
#                 regrCV.fit(X[train, :], Y[train, :])
#                 y_pred_CV = regrCV.predict(X[test, :])
#                 var_s[s] = explained_variance_score(Y[test, :], y_pred_CV)
#                 mse_s[s] = mean_squared_error(Y[test, :], y_pred_CV)
#             var[i] = np.mean(var_s)
#             mse[i] = np.mean(mse_s)

#         sig = x_pow.copy().mean(0)
#         n = n_pow.copy().mean(0)
#         # snr[st] = sig/n
#         var_k[j] = np.round(np.mean(var[:, 0]), 4)
#         mse_k[j] = np.round(np.mean(mse[:, 0]), 4)
#         mse_std[j] = np.round(np.std(mse[:, 0]), 4)

#     # SNR = 10*np.log10(snr)
#     # r2_snr = np.round(np.mean(r2[:, 0]), 4)
#     # print('snr: ', snr, '   / var: ', var_snr)
# plt.figure()
# plt.plot(np.delete(K, [-2]), np.delete(var_k, [-2]), 'bs')
# plt.plot(np.delete(K, [-2]), np.delete(var_k, [-2]), 'b--')
# # plt.plot(K, var_k, 'bs')
# # plt.plot(K, var_k, 'b--')
# plt.title('X and Y dimension: [' + str(r) + ', '+str(c)+']')
# plt.xlabel('K')
# plt.ylabel('Explained Variance')

# mean = mse_k
# error = mse_std

# # my_color=['mediumblue','r','teal','orange']

# # labels = ['SD_lATL', 'SD_rATL','LD_lATL', 'LD_rATL']
# # x_pos = np.arange(len(K))
# # x_pos = np.arange(len(K))
# x_pos = K

# fig, ax = plt.subplots(figsize=(9, 5))
# for i in np.arange(0, len(K)):
#     ax.bar(x_pos[i], mean[i],
#            yerr=error[i],
#            align='center',
#            alpha=0.4,
#            ecolor='black',
#            capsize=10,
#            # color=my_color[i],
#            width=0.4)
# ax.set_ylabel('MSE')
# ax.set_xlabel('K')

# ax.set_xticks(K)
# ax.plot(K, mean, 'g--')
# # ax.set_xticklabels(labels)
# ax.set_title('X and Y dimension: [' + str(r) + ', '+str(c)+']')
# # ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()


# plt.figure()
# plt.plot(K, mse_k, 'rs')
# plt.plot(K, mse_k, 'r--')
# plt.title('X and Y dimension: [' + str(r) + ', '+str(c)+']')
# plt.xlabel('K')
# plt.ylabel('MSE')

# # plt.close('all')
################################################################
# Effect of leave-one-out
r, c = [40, 20]
# plt.close('all')
repeat = 50
# std_pow = [2.5, 2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2]
std_pow = [.5]


for std in std_pow:

    # snr = np.zeros([len(std_pow), 1])
    # var_k = np.zeros([len(K), 1])
    # mse_k = np.zeros([len(K), 1])
    # mse_std = np.zeros([len(K), 1])
    noise = np.random.normal(0, 10**std, (r, c))
    T = np.random.normal(0, 1, (c, c))
    # n_pow = np.zeros([repeat, 1])
    # x_pow = np.zeros([repeat, 1])

    var = np.zeros([repeat, 1])
    mse = np.zeros([repeat, 1])
    for i in np.arange(0, repeat):
        # print ('r, c , i: ',r,', ', c,', ',i)
        X = np.random.normal(0, 1, (r, c))
        Y = np.matmul(X, T)
        Y = Y+noise
        # n_pow[i, 0] = np.var(noise)
        # x_pow[i, 0] = np.var(np.matmul(X, T))
        # n_splits = k

        var_s = np.zeros([r])
        mse_s = np.zeros([r])

        loo = LeaveOneOut()
        regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100),
                         normalize=True)
        scores = cross_validate(regrCV, X, Y, scoring='explained_variance',
                                cv=loo, n_jobs=-1, return_train_score=True)
    #     for s, (train, test) in enumerate(loo.split(X,Y)):
    #         # print(s,train, test)
    #         regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100),
    #                          normalize=True)
    #         regrCV.fit(X[train, :], Y[train, :])
    #         y_pred_CV = regrCV.predict(X[test, :])return_train_score
    #         var_s[s] = explained_variance_score(Y[test, :], y_pred_CV)
    #         mse_s[s] = mean_squared_error(Y[test, :], y_pred_CV)
    #     var[i] = np.mean(var_s)
    #     mse[i] = np.mean(mse_s)

    #     # sig = x_pow.copy().mean(0)
    #     # n = n_pow.copy().mean(0)
    #     # snr[st] = sig/n
    # print(np.round(np.mean(var[:, 0]), 4))
        # mse_k[j] = np.round(np.mean(mse[:, 0]), 4)
        # mse_std[j] = np.round(np.std(mse[:, 0]), 4)


############################################################
# # ### Y=XT and X=YT
# r_x, c_x = 100, 70
# r_x, c_y = 100, 50

# # number of iteration
# repeat = 10

# # choose the coefficient matrix
# std_pow = [2.5, 2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2]

# normalize = True


# def my_regressor(X, Y):
#     r = X.shape[0]
#     c = np.max([X.shape[1], Y.shape[1]])
#     if r >= c:
#         if (r/5) < 10:
#             n_splits = math.floor(r/5)
#         else:
#             n_splits = 10
#     else:
#         n_splits = np.max([math.floor(r/5)-1, 2])

#     var_s = np.zeros([n_splits])

#     kf = KFold(n_splits=n_splits)
#     for s, (train, test) in enumerate(kf.split(X, Y)):
#         # print(s,train, test)
#         regrCV = RidgeCV(alphas=np.logspace(-10, 10, 100),
#                          normalize=normalize)
#         regrCV.fit(X[train, :], Y[train, :])
#         y_pred_CV = regrCV.predict(X[test, :])
#         var_s[s] = explained_variance_score(Y[test, :], y_pred_CV)

#     return np.mean(var_s)


# std_x = -1
# std_y = -1

# var_x2y = np.zeros([repeat, 1])
# var_y2x = np.zeros([repeat, 1])
# snr_x = np.zeros([repeat, 1])
# snr_y = np.zeros([repeat, 1])
# noise_x = np.random.normal(0, 10**std_x, (r_x, c_x))
# noise_y = np.random.normal(0, 10**std_y, (r_x, c_y))

# T = np.random.normal(0, 1, (c_x, c_y))
# # T = np.identity(c_y)

# for i in np.arange(0, repeat):
#     # print ('r, c , i: ',r,', ', c,', ',i)
#     x = np.random.normal(0, 1, (r_x, c_x))
#     y = np.matmul(x, T)

#     # snr_x[i] = 10*np.log(np.var(x)/np.var(noise_x))
#     # snr_y[i] = 10*np.log(np.var(y)/np.var(noise_y))
#     snr_x[i] = np.var(x)/np.var(noise_x)
#     snr_y[i] = np.var(y)/np.var(noise_y)
#     X = x+noise_x
#     Y = y+noise_y
#     # X = y
#     # Y = x

#     var_x2y[i] = my_regressor(X, Y)
#     var_y2x[i] = my_regressor(Y, X)

# var_snr_x2y = np.round(np.mean(var_x2y[:, 0]), 4)
# var_snr_y2x = np.round(np.mean(var_y2x[:, 0]), 4)
# print('var_snr_x2y: ', var_snr_x2y)
# print('var_snr_y2x: ', var_snr_y2x)

# print('snr_x: ', np.round(np.mean(snr_x), 2))
# print('snr_y: ', np.round(np.mean(snr_y), 2))
