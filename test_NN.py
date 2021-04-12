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
from sklearn.metrics import mean_squared_error, r2_score, \
    explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

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
r, c = [200, 100]
y1 = np.zeros([r, c])
hidden_layer_sizes = [2, 5, 10, int(np.floor(c / 3)), int(np.floor(c / 2)), c]

noise = np.random.normal(0, 1, (r, c))
T = np.random.normal(0, .5, (c, c))
X = np.random.normal(0, 1, (r, c))
Y = np.matmul(X, T)  # + noise

r = X.shape[0]
n_splits = 5
if (r / 5) > 10:
    n_splits = 10
var_L = np.zeros([len(hidden_layer_sizes), 2])
var_r = np.zeros([len(hidden_layer_sizes), 2])
var_t = np.zeros([len(hidden_layer_sizes), 2])

kf = KFold(n_splits=n_splits)
for n_h, h in enumerate(hidden_layer_sizes):
    print(f'hidden_layer_sizes: {h}')
    var_s_L = np.zeros([n_splits, 2])
    var_s_r = np.zeros([n_splits, 2])
    var_s_t = np.zeros([n_splits, 2])

    for s, (train, test) in enumerate(kf.split(X, Y)):
        # print(s,train, test)
        mlp_t = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation='tanh', solver='lbfgs',
                             max_iter=1000, alpha=1e-04)
        mlp_r = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation='relu', solver='lbfgs',
                             max_iter=1000, alpha=1e-04)
        mlp_L = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation='identity', solver='lbfgs',
                             max_iter=1000, alpha=1e-04)

        x_train = StandardScaler().fit(
            X[train, :]).transform(X[train, :])
        y_train = StandardScaler().fit(
            Y[train, :]).transform(Y[train, :])

        x_test = StandardScaler().fit(
            X[train, :]).transform(X[test, :])
        y_test = StandardScaler().fit(
            Y[train, :]).transform(Y[test, :])

        mlp_t.fit(X[train, :], Y[train, :])
        y_pred_t = mlp_t.predict(X[test, :])
        y_pred_t_train = mlp_t.predict(X[train, :])

        mlp_r.fit(X[train, :], Y[train, :])
        y_pred_r = mlp_r.predict(X[test, :])
        y_pred_r_train = mlp_r.predict(X[train, :])

        mlp_L.fit(X[train, :], Y[train, :])
        y_pred_L = mlp_L.predict(X[test, :])
        y_pred_L_train = mlp_L.predict(X[train, :])

        var_s_t[s, 0] = explained_variance_score(Y[test, :], y_pred_t)
        var_s_r[s, 0] = explained_variance_score(Y[test, :], y_pred_r)
        var_s_L[s, 0] = explained_variance_score(Y[test, :], y_pred_L)

        var_s_t[s, 1] = explained_variance_score(Y[train, :], y_pred_t_train)
        var_s_r[s, 1] = explained_variance_score(Y[train, :], y_pred_r_train)
        var_s_L[s, 1] = explained_variance_score(Y[train, :], y_pred_L_train)

    var_t[n_h, 0] = np.mean(var_s_t[:, 0])
    var_r[n_h, 0] = np.mean(var_s_r[:, 0])
    var_L[n_h, 0] = np.mean(var_s_L[:, 0])

    var_t[n_h, 1] = np.mean(var_s_t[:, 1])
    var_r[n_h, 1] = np.mean(var_s_r[:, 1])
    var_L[n_h, 1] = np.mean(var_s_L[:, 1])
plt.figure()
plt.plot(hidden_layer_sizes, var_t[:, 0], 'r', label='test')
plt.plot(hidden_layer_sizes, var_t[:, 1], 'r--', label='train')
plt.legend(loc='upper right')
plt.title('tanh')

plt.figure()
plt.plot(hidden_layer_sizes, var_r[:, 0], 'b', label='test')
plt.plot(hidden_layer_sizes, var_r[:, 1], 'b--', label='train')
plt.legend(loc='upper right')
plt.title('relu')

plt.figure()
plt.plot(hidden_layer_sizes, var_L[:, 0], 'g', label='test')
plt.plot(hidden_layer_sizes, var_L[:, 1], 'g--', label='train')
plt.legend(loc='upper right')
plt.title('identity')

plt.show()
# plt.close('all')

# # from keras.models import Sequential
# # from keras.layers import Dense
# # model = Sequential()
# # input_size = x_train.shape[0]
# # model.add(Dense(100, activation="tanh", input_dim=input_size))
# # model.add(Dense(input_size , activation="linear"))
# # model.compile(optimizer="adam", loss="mse")
# # model.fit(x_train, y_train, epochs=25, verbose=1)
# # y_pred = model.predict(x_test)
