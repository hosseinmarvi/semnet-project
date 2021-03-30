#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:40:05 2021

@author: sr05
"""
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# for n in np.arange(2, trails):
#     x0[n, :] = 2*x0[n-1, :]*np.exp(-1*x0[n-1, :]**2)-0.8*x0[n-2, :]
#     x1[n, :] = 0.8*x0[n-1, :]**2+x1[n-1, :] * \
#         np.exp(-1*x1[n-1, :]**2)-0.8*x1[n-2, :]
#     x2[n, :] = 0.8*x0[n-1, :]**2+x1[n-1, :] * \
#         np.exp(-1*x2[n-1, :]**2)-0.8*x2[n-2, :]
# data = [x0, x1, x2]


# labels = [0, 1]


def mvgc_io(data, roi_x, roi_y, delay):

    time_series_length = data[roi_y].shape[0]
    time_series_range = np.arange(delay, time_series_length)
    y = data[roi_y][time_series_range, :]

    x_total = mvgc_x(data, labels, delay, time_series_range)
    labels.remove(roi_x)
    x = mvgc_x(data, labels, delay, time_series_range)
    return y, x_total, x
# y, x_total, x = mvgc_io(data, roi_x, roi_y, delay)


def mvgc_x(data, labels, delay, time_series_range):

    for r, roi in enumerate(labels):
        x_roi = np.zeros([time_series_range.shape[0],
                          (data[roi].shape[1]*delay)])
        for d in np.arange(1, delay + 1):
            x_roi[:, data[roi].shape[1] *
                  (d-1):data[roi].shape[1]*d] = data[roi][time_series_range-d, :]

        if r == 0:
            x_total = x_roi
        else:
            x_total = np.append(x_total, x_roi, axis=1)
    return x_total


def mvgc(y, x_total, x, roi_x, roi_y,  hidden_layer_sizes, activation,
         normalize=True):

    r = x.shape[0]
    if (r/5) > 10:
        n_splits = 10
    else:
        n_splits = 5
    gc_s = np.zeros(n_splits)
    gc_s_l = np.zeros(n_splits)
    gc_s_nl = np.zeros(n_splits)

    kf = KFold(n_splits=n_splits)
    for s, (train, test) in enumerate(kf.split(x, y)):
        print(s)
        error_x = mvgc_reg(x, y, train, test, normalize)
        error_total = mvgc_reg(x_total, y, train, test, normalize)

        error_x_l, error_x_nl = mvgc_nonlinear_reg(
            x, y, train, test, hidden_layer_sizes, activation)
        error_total_l, error_total_nl = mvgc_nonlinear_reg(
            x_total, y, train, test, hidden_layer_sizes, activation)

        gc = np.zeros(error_x.shape[1])
        gc_l = np.zeros(error_x.shape[1])
        gc_nl = np.zeros(error_x.shape[1])

        for i in np.arange(error_x.shape[1]):
            gc[i] = np.log(np.var(error_x[:, i])/np.var(error_total[:, i]))
            gc_l[i] = np.log(np.var(error_x_l[:, i]) /
                             np.var(error_total_l[:, i]))
            gc_nl[i] = np.log(np.var(error_x_nl[:, i]) /
                              np.var(error_total_nl[:, i]))

        gc_s[s] = np.mean(gc)
        gc_s_l[s] = np.mean(gc_l)
        gc_s_nl[s] = np.mean(gc_nl)

    gc_total = np.mean(gc_s)
    gc_total_l = np.mean(gc_s_l)
    gc_total_nl = np.mean(gc_s_nl)

    return gc_total, gc_total_l, gc_total_nl


def mvgc_reg(x, y, train, test, normalize):
    regr = RidgeCV(alphas=np.logspace(-4, 4, 100),
                   normalize=True)
    regr.fit(x[train, :], y[train, :])
    y_pred = regr.predict(x[test, :])
    error = y[test, :] - \
        y_pred.reshape(y[test, :].shape[0], y[test, :].shape[1])
    return error


def mvgc_nonlinear_reg(x, y, train, test, hidden_layer_sizes, activation):
    x_train = StandardScaler().fit(
        x[train, :]).transform(x[train, :])
    y_train = StandardScaler().fit(
        y[train, :]).transform(y[train, :])

    x_test = StandardScaler().fit(
        x[train, :]).transform(x[test, :])
    y_test = StandardScaler().fit(
        y[train, :]).transform(y[test, :])

    mlp_l = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         activation='identity', learning_rate='adaptive',
                         solver='lbfgs', max_iter=1000, alpha=1e-04)
    mlp_nl = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                          activation=activation, learning_rate='adaptive',
                          solver='lbfgs', max_iter=1000, alpha=1e-04)

    mlp_l.fit(x_train, y_train)
    mlp_nl.fit(x_train, y_train)
    y_pred_l = mlp_l.predict(x_test)
    y_pred_nl = mlp_nl.predict(x_test)

    error_l = y_test - y_pred_l.reshape(y_test.shape[0], y_test.shape[1])
    error_nl = y_test - y_pred_nl.reshape(y_test.shape[0], y_test.shape[1])

    return error_l, error_nl


roi_y = 1
roi_x = 2
delay = 2
repeat = 10
hidden_layer_sizes = 100
activation = 'tanh'
gc = np.zeros(repeat)
gc_l = np.zeros(repeat)
gc_nl = np.zeros(repeat)

start = time.time()
for m in np.arange(repeat):
    print(f'repeat: {m}')
    trails = 200
    vertices = 2
    x0 = np.zeros([trails, vertices])
    x1 = np.zeros([trails, vertices])
    x2 = np.zeros([trails, vertices])

    x0[0, :], x0[1, :] = np.random.randn(2, vertices)
    x1[0, :], x1[1, :] = np.random.randn(2, vertices)
    x2[0, :], x2[1, :] = np.random.randn(2, vertices)
    labels = [0, 1, 2]

    for n in np.arange(2, trails):
        x0[n, :] = 0.5*x0[n-1, :]-0.7 * x0[n-2, :]-0.6*x1[n-1, :]
        x1[n, :] = 0.2*x0[n-1, :]+0.7*x1[n-1, :] - \
            0.5*x1[n-2, :]+0.8*x2[n-1, :]
        x2[n, :] = 0.8*x2[n-1, :]

    # for n in np.arange(2, trails):
    #     x0[n, :] = 2*x0[n-1, :]*np.exp(-1*x0[n-1, :]**2)-0.8*x0[n-2, :]
    #     x1[n, :] = 0.8*x0[n-1, :]**2+x1[n-1, :] * \
    #         np.exp(-1*x1[n-1, :]**2)-0.8*x1[n-2, :]
    #     x2[n, :] = 0.8*x0[n-1, :]**2+x1[n-1, :] * \
    #         np.exp(-1*x2[n-1, :]**2)-0.8*x2[n-2, :]
    data = [x0, x1, x2]

    y, x_total, x = mvgc_io(data, roi_x, roi_y, delay)
    gc[m], gc_l[m], gc_nl[m] = mvgc(y, x_total, x, roi_x, roi_y,
                                    hidden_layer_sizes, activation,
                                    normalize=True)
print(f"Granger Causality X{roi_x} to X{roi_y}: {np.mean(gc)}")
print(f"NN Granger Causality X{roi_x} to X{roi_y}: {np.mean(gc_l)}")
print(f"NL NN Granger Causality X{roi_x} to X{roi_y}: {np.mean(gc_nl)}")
end = time.time()
print(end - start)
