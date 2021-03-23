#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:50:08 2021

@author: sr05
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import (cross_validate, KFold)
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
############################################################################
# X = np.arange(0.0, 1, 0.01).reshape(-1, 1)
# y = np.sin(2 * np.pi * X).ravel()
# # X = np.arange(0,100,1).reshape(-1, 1)
# # y = .8*X
# n_splits = 10

# kf = KFold(n_splits=n_splits)
# MSE_s = np.zeros([n_splits])
# var_s = np.zeros([n_splits])
# activation = 'tanh'

# for s, (train, test) in enumerate(kf.split(X, y)):
#     # print(s,train, test)
#     nn = MLPRegressor(hidden_layer_sizes=(3),
#                       activation=activation, solver='lbfgs')
#     nn.fit(X[train], y[train])
#     y_pred_CV = nn.predict(X[test])

#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.scatter(X, y, s=5, c='b', marker="o", label='real')
#     ax1.plot(X[test], y_pred_CV, c='r', label='NN Prediction')

#     plt.legend()
#     plt.show()

#     MSE_s[s] = mean_squared_error(y[test], y_pred_CV)
#     var_s[s] = explained_variance_score(y[test], y_pred_CV)
# print( np.mean(var_s))

# nn = MLPRegressor(hidden_layer_sizes=(3),
#                   activation=activation, solver='lbfgs')
# scores = cross_validate(
#     nn, X, y, scoring=('explained_variance'), cv=kf, n_jobs=-1)
# score = np.mean(scores['test_score'])
# print(score)

###########################################################################
normalize = True
r, c = [200, 50]
y1 = np.zeros([r, c])
hidden_layer_sizes = 100

noise = np.random.normal(0, 1, (r, c))
T = np.random.normal(0, 1, (c, c))
X1 = np.random.normal(0, 1, (r, c))
y1 = np.matmul(X1, T)  # + noise
# y1 = np.sin(X1)

# y1 = y1 + noise
# scalerx = StandardScaler()
# scalery = StandardScaler()

# scalerx.fit(X1)
# scalery.fit(y1)

# X = scalerx.transform(X1)
# y = scalerx.transform(y1)


# y = np.matmul(X, T)
# print('SNR: ', np.var(y1)/np.var(noise))
# score_nh = np.zeros([len(hidden_layer_size), 1])

r = X1.shape[0]
if (r/5) > 10:
    n_splits = 10
else:
    n_splits = 5

kf = KFold(n_splits=n_splits)
MSE_s = np.zeros([n_splits])
var_s = np.zeros([n_splits])
activation = 'identity'

for s, (train, test) in enumerate(kf.split(X1, y1)):
    # print(s,train, test)
    mlp = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                     activation=activation, solver='lbfgs', max_iter=1000))
    # nn = MLPRegressor(hidden_layer_sizes=(100),
    #                   activation=activation, solver='lbfgs', max_iter=1000)
    mlp.fit(X1[train, :], y1[train, :])
    y_pred = mlp.predict(X1[test, :])

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(X1[test, 0], y1[test, 0], s=5, c='b', marker="s", label='real')
    # ax1.scatter(X1[test, 0], y_pred[:, 0], c='r',
    #             marker="o", label='NN Prediction')

    # plt.legend()
    # plt.show()

    MSE_s[s] = mean_squared_error(y1[test, 0], y_pred[:, 0])
    var_s[s] = explained_variance_score(y1[test, 1], y_pred[:, 1])


print(np.mean(var_s))

nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                  activation=activation, solver='lbfgs', max_iter=1000)
scores = cross_validate(
    nn, X1, y1, scoring=('explained_variance'), cv=kf, n_jobs=-1)
score = np.mean(scores['test_score'])
print(score)

# plt.close('all')


# from keras.models import Sequential
# from keras.layers import Dense
# model = Sequential()
# input_size = len(X[0])
# model.add(Dense(200, activation="relu", input_dim=input_size))
# model.add(Dense(200, activation="relu"))
# model.add(Dense(1, activation="linear"))
# model.compile(optimizer="adam", loss="mse")
# model.fit(X_train, y_train, epochs=25, verbose=1)
# y_pred = model.predict(X_test)
