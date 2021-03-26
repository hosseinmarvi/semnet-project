#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:40:05 2021

@author: sr05
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

x1 = np.zeros([100, 1])
x2 = np.zeros([100, 1])
x3 = np.zeros([100, 1])

x1[0, 0], x1[1, 0] = np.random.random_sample(2)
x2[0, 0], x2[1, 0] = np.random.random_sample(2)
x3[0, 0], x3[1, 0] = np.random.random_sample(2)

for n in np.arange(2, 100):
    x1[n, 0] = 0.5*x1[n-1, 0]-0.7*x1[n-2, 0]-0.6*x2[n-1, 0]
    x2[n, 0] = 0.2*x1[n-1, 0]+0.7*x2[n-1, 0]-0.5*x2[n-2, 0]+0.8*x3[n-1, 0]
    x3[n, 0] = 0.8*x3[n-1, 0]
data = [x1, x2, x3]

roi_y = 2
roi_x = 1
labels = [0, 1, 2]

def mvgc_io(data, roi_x, roi_y, d):

    y = data[roi_y][d:]

    x_total = mvgc_x(data, labels, d)
    labels.remove(roi_x)
    x = mvgc_x(data, labels, d)
    return y, x_total, x


def mvgc_x(data, labels, d):

    for r, roi in enumerate(labels):
        x_roi = data[roi][0:-d]
        if r == 0:
            x_total = x_roi
        else:
            x_total = np.append(x_total, x_roi, axis=1)
    return x_total


def mvgc(y, x_total, x, roi_x, roi_y, normalize=True):

    r = x.shape[0]
    if (r/5) > 10:
        n_splits = 10
    else:
        n_splits = 5
    gc_s = np.zeros(n_splits)
    kf = KFold(n_splits=n_splits)
    for s, (train, test) in enumerate(kf.split(x, y)):
        error_x = mvgc_reg(x, y, train, test, normalize)
        error_total = mvgc_reg(x_total, y, train, test, normalize)
        gc = np.zeros(error_x.shape[1])
        for i in np.arange(error_x.shape[1]):
            gc[i] = np.log(np.var(error_x[:, i])/np.var(error_total[:, i]))
        gc_s[s] = np.mean(gc)
    gc_total = np.mean(gc_s)
    return gc_total


def mvgc_reg(x, y, train, test, normalize):
    regr = RidgeCV(alphas=np.logspace(-5, 5, 100),
                   normalize=True)
    regr.fit(x[train, :], y[train, :])
    y_pred = regr.predict(x[test, :])
    error = y[test, :] - y_pred
    return error


y, x_total, x = mvgc_io(data, 0, 1, 2)
gc = mvgc(y, x_total, x, 0, 1, True)
