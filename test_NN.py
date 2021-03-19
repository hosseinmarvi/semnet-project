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

r, c = [200, 100]
y1 = np.zeros([r, c])
hidden_layer_size = np.arange(int(r/5), int(3*r), 50)
# [int(r/5), int(r/2), int(r),
#                      int(1.5*r), int(2*r), int(3*r), int(5*r), int(10*r)]
noise = np.random.normal(0, .1, (r, c))
T = np.random.normal(0, 1, (c, c))
X = np.random.normal(0, 1, (r, c))
# y = np.matmul(X, T) + noise
for i in np.arange(0, r):
    for j in np.arange(0, c):
        y1[i, j] = math.tanh(X[i, j])
y = y1 + noise

# y = np.matmul(X, T)
print('SNR: ', np.var(y1)/np.var(noise))
score_nh = np.zeros([len(hidden_layer_size), 1])

r = X.shape[0]
if (r/5) > 10:
    n_splits = 10
else:
    n_splits = 5

kf = KFold(n_splits=n_splits)
regrCV = RidgeCV(alphas=np.logspace(-5, 5, 100), normalize=True)
scoresCV = cross_validate(
    regrCV, X, y, scoring=('explained_variance'), cv=kf, n_jobs=-1)
score_CV = np.mean(scoresCV['test_score'])
print('RR score: ', score_CV)
# for i, nh in enumerate(hidden_layer_size):
#     # for nh in [40,80,120,200,1000]:

#     print('hidden_layer_size: ', nh)
#     regrMLP = MLPRegressor(hidden_layer_sizes=(nh), max_iter=500,
#                            activation='identity', solver='lbfgs')
#     scoresMLP = cross_validate(
#         regrMLP, X, y, scoring=('explained_variance'), cv=kf, n_jobs=-1)
#     score_nh[i, 0] = np.mean(scoresMLP['test_score'])
#     print('MPL score: ', np.mean(scoresMLP['test_score']))

# plt.figure()
# plt.plot(hidden_layer_size, score_nh)

regrMLP = MLPRegressor(hidden_layer_sizes=r+1000, max_iter=500,
                       activation='tanh', solver='lbfgs')
scoresMLP = cross_validate(
    regrMLP, X, y, scoring=('explained_variance'), cv=kf, n_jobs=-1)
print('MPL score: ', np.mean(scoresMLP['test_score']))

# plt.figure()
# plt.plot(hidden_layer_size, score_nh)
# plt.close('all')
