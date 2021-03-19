#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:12:04 2021

@author: sr05
"""
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (cross_validate, KFold)

x = np.arange(0.0, 10, 0.01).reshape(-1, 1)
y = np.sin(2 * np.pi * x).ravel()

nn = MLPRegressor(hidden_layer_sizes=(3), 
                  activation='tanh', solver='lbfgs')

n = nn.fit(x, y)
test_x = np.arange(-0.1, 1.1, 0.01).reshape(-1, 1)
test_y = nn.predict(test_x)

kf = KFold(n_splits=10)

scores =cross_validate(
            nn, x, y, scoring=('explained_variance'), cv=kf, n_jobs=-1)
score = np.mean(scores['test_score'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=5, c='b', marker="o", label='real')
ax1.plot(test_x,test_y, c='r', label='NN Prediction')

plt.legend()
plt.show()