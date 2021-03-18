
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:34:18 2020

@author: sr05
"""
import numpy as np
import math
from numpy.linalg import inv
X = np.random.rand(10, 4)


def create_time_series(x, y, delay):

    time_series_length = x.shape[0]
    time_series_range = np.arange(delay, time_series_length)
    X1 = np.zeros([time_series_range.shape[0], (x.shape[1]*delay)])

    X1 = np.zeros([time_series_range.shape[0], (x.shape[1]*delay)])

    for d in np.arange(1, delay + 1):
        X1[:, x.shape[1]*(d-1):x.shape[1]*d] = x[time_series_range-d, :]
    Y1 = y[time_series_range]

    return X1, Y1


def bivariate_error_value(X, delay):

    if X.shape[0] < X.shape[1]:
        X = X.transpose()
    bv_error_value = np.zeros([X.shape[1], X.shape[1]])

    for c1 in np.arange(0, X.shape[1]):
        for c2 in np.arange(0, X.shape[1]):
            if c1 != c2:
                x = X[:, [c1, c2]]
                y = X[:, c1]
                [Input, Target] = create_time_series(x, y, delay)
                beta_coef = np.matmul(inv(np.matmul(Input.transpose(), Input)),
                                      np.matmul(Input.transpose(), Target))

                # error_value = np.sum(np.power(np.subtract(np.matmul(Input,beta_coef),Target),2))
                error_value = np.var(np.subtract(
                    np.matmul(Input, beta_coef), Target))
                bv_error_value[c1, c2] = error_value
            else:
                bv_error_value[c1, c2] = 0
    return bv_error_value


def univariate_error_value(X, delay):

    if X.shape[0] < X.shape[1]:
        X = X.transpose()
    uv_error_value = np.zeros([1, X.shape[1]])

    for c1 in np.arange(0, X.shape[1]):

        x = np.zeros([X.shape[0], 1])
        y = np.zeros([X.shape[0], 1])

        x[:, 0] = X[:, c1]
        y[:, 0] = X[:, c1]
        [Input, Target] = create_time_series(x, y, delay)
        beta_coef = np.matmul(inv(np.matmul(Input.transpose(), Input)),
                              np.matmul(Input.transpose(), Target))

        # error_value = np.sum(np.power(np.subtract(np.matmul(Input,beta_coef),
        # Target), 2))
        error_value = np.var(np.subtract(np.matmul(Input, beta_coef), Target))
        uv_error_value[0, c1] = error_value
    return uv_error_value


def linear_granger_causality(X, delay):
    uv_error_value = univariate_error_value(X, delay)
    bv_error_value = bivariate_error_value(X, delay)
    LGC = np.zeros([bv_error_value.shape[0], bv_error_value.shape[0]])
    for c1 in np.arange(0, LGC.shape[0]):
        for c2 in np.arange(0, LGC.shape[0]):
            if c1 != c2:
                a = math.log(
                    np.var(uv_error_value[0, c1])/np.var(bv_error_value[c1, c2]))
                if a > 0:
                    LGC[c1, c2] = a
                else:
                    LGC[c1, c2] = 0
            else:
                LGC[c1, c2] = 0

    return LGC


Z = linear_granger_causality(X, 3)
UVE = univariate_error_value(X, 3)
BVE = bivariate_error_value(X, 3)
