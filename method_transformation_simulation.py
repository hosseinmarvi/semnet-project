#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:26:44 2021

@author: sr05
"""
import math
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    cross_validate, LeaveOneOut, train_test_split, KFold)
from sklearn.metrics import (
    mean_squared_error, r2_score, explained_variance_score)

"""
===========================================================================
Effect of K in K-fold Cross-validation on explained variance and MSE
===========================================================================
"""
# R = [50]
# C = [50]
# repeat = 10
# std = -.5
# normalize = True

# for r in R:
#     for c in C:
#         K = np.array([3, 5, 8, 10, 20, r-5, r-2, r-1])

#         var_k = np.zeros([len(K), 1])
#         mse_k = np.zeros([len(K), 1])
#         mse_std = np.zeros([len(K), 1])
#         noise = np.random.normal(0, 10**std, (r, c))
#         T = np.random.normal(0, 1, (c, c))
#         n_pow = np.zeros([repeat, 1])
#         x_pow = np.zeros([repeat, 1])

#         for j, k in enumerate(K):
#             print(k)
#             var = np.zeros([repeat, 1])
#             mse = np.zeros([repeat, 1])
#             for i in np.arange(0, repeat):
#                 X = np.random.normal(0, 1, (r, c))
#                 Y = np.matmul(X, T)
#                 Y = Y+noise
#                 n_pow[i, 0] = np.var(noise)
#                 x_pow[i, 0] = np.var(np.matmul(X, T))
#                 kf = KFold(n_splits=k)
#                 regrCV = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                  normalize=normalize)
#                 scores = cross_validate(
#                     regrCV, X, Y, scoring=('explained_variance',
#                                            'neg_mean_squared_error'),
#                     cv=kf, n_jobs=-1)

#                 var[i] = np.mean(scores['test_explained_variance'])
#                 mse[i] = np.mean(np.abs(scores['test_neg_mean_squared_error']))

#             var_k[j] = np.round(np.mean(var[:, 0]), 4)
#             mse_k[j] = np.round(np.mean(mse[:, 0]), 4)
#             mse_std[j] = np.round(np.std(mse[:, 0]), 4)

#         plt.figure()
#         # plt.plot(np.delete(K, [-3, -2]), np.delete(var_k, [-3, -2]), 'bs')
#         # plt.plot(np.delete(K, [-3, -2]), np.delete(var_k, [-3, -2]), 'b--')
#         plt.plot(K, var_k, 'bs')
#         plt.plot(K, var_k, 'b--')
#         plt.title('X and Y dimension: [' + str(r) + ', '+str(c)+']')
#         plt.xlabel('K')
#         plt.ylabel('Explained Variance')

#         mean = mse_k
#         error = mse_std
#         x_pos = K
#         fig, ax = plt.subplots(figsize=(9, 5))
#         for i in np.arange(0, len(K)):
#             ax.bar(x_pos[i], mean[i],
#                    yerr=error[i],
#                    align='center',
#                    alpha=0.4,
#                    ecolor='black',
#                    capsize=10,
#                    width=0.4)
#         ax.set_ylabel('MSE')
#         ax.set_xlabel('K')
#         ax.set_xticks(K)
#         ax.plot(K, mean, 'g--')
#         ax.set_title('X and Y dimension: [' + str(r) + ', '+str(c)+']')
#         plt.tight_layout()
#         plt.show()
# # plt.close('all')
"""
===========================================================================
Transformation between two random noise (two independant variables)
===========================================================================
"""
repeat = 100
normalize = True
R = [30, 50, 100, 200, 500, 1000]
C = [50, 100, 150]
my_color = ['ro', 'r--', 'bs', 'b--', 'gx', 'g--']


def noise_connectivity(c, r, repeat):

    mse = np.zeros([repeat])
    var = np.zeros([repeat])

    for i in np.arange(0, repeat):
        print('r, c , i: ', r, ', ', c, ', ', i)
        X = np.random.normal(0, 1, (r, c))
        Y = np.random.normal(0, 1, (r, c))

        if (r/5) > 10:
            n_splits = 10
        else:
            n_splits = 5

        kf = KFold(n_splits=n_splits)
        regrCV = RidgeCV(alphas=np.logspace(-5, 5, 100),
                          normalize=normalize)
        scores = cross_validate(
            regrCV, X, Y, scoring=('explained_variance',
                                    'neg_mean_squared_error'),
            cv=kf, n_jobs=-1, return_train_score=True)

        var[i] = np.mean(scores['test_explained_variance'])
        mse[i] = np.mean(scores['test_neg_mean_squared_error'])

    return {'MSE': np.mean(mse), 'var': np.mean(var)}


s = time.time()
GOF_ave = Parallel(n_jobs=-1)(delayed(noise_connectivity)
                              (c, r, repeat)for c in C for r in R)
e = time.time()
print(e-s)

mse = np.zeros([len(C), len(R)])
var = np.zeros([len(C), len(R)])
for i in range(len(C)):
    for j in range(len(R)):
        mse[i, j] = np.abs(GOF_ave[i*len(R)+j]['MSE'])
        var[i, j] = GOF_ave[i*len(R)+j]['var']

plt.rcParams['font.size'] = '14'
plt.figure(figsize=(11, 5))
for i in np.arange(0, len(C)):
    plt.plot(R, mse[i], my_color[i*2], label=str(C[i]))
    plt.plot(R, mse[i], my_color[i*2+1])
plt.legend(loc='upper right')
plt.xlabel('Number of Trials')
plt.ylabel('MSE')
plt.savefig('/home/sr05/Method_dev/method_fig/simulation_noise_connectivity_MSE2')


plt.figure(figsize=(11, 5))
for i in np.arange(0, len(C)):
    plt.plot(R, var[i], my_color[i*2], label=str(C[i]))
    plt.plot(R, var[i], my_color[i*2+1])
plt.legend(loc='upper right')
plt.xlabel('Number of Trials')
plt.ylabel('Explained Variance')
plt.savefig('/home/sr05/Method_dev/method_fig/simulation_noise_connectivity_var2')

"""
===========================================================================
Linear Transformation between two variables
===========================================================================
"""
# repeat = 100
# c = 150
# R = [30, 50, 100, 150, 300, 500, 1000]
# std_pow = [2.5, 2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2]
# n_pow = np.zeros([repeat, 1])
# x_pow = np.zeros([repeat, 1])
# my_colors = ['bo', 'b--', 'yo', 'y--', 'go', 'g--',
#              'm*', 'm--', 'bs', 'b--', 'rs', 'r--', 'gs', 'g--']
# normalize = True


# def SNR_trials_connectivity(repeat, r, c, std):
#     mse = np.zeros([repeat, 1])
#     var = np.zeros([repeat, 1])

#     noise = np.random.normal(0, 10**std, (r, c))
#     T = np.random.normal(0, 1, (c, c))
#     for i in np.arange(0, repeat):
#         # print ('r, c , i: ',r,', ', c,', ',i)
#         X = np.random.normal(0, 1, (r, c))
#         Y = np.matmul(X, T)
#         Y = Y+noise
#         n_pow[i, 0] = np.var(noise)
#         x_pow[i, 0] = np.var(np.matmul(X, T))

#         if (r/5) > 10:
#             n_splits = 10
#         else:
#             n_splits = 5

#         kf = KFold(n_splits=n_splits)
#         regrCV = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                          normalize=normalize)
#         scores = cross_validate(
#             regrCV, X, Y, scoring=('explained_variance',
#                                    'neg_mean_squared_error'),
#             cv=kf, n_jobs=-1, return_train_score=True)

#         var[i] = np.mean(scores['test_explained_variance'])
#         mse[i] = np.mean(scores['test_neg_mean_squared_error'])

#     sig = x_pow.copy().mean(0)
#     n = n_pow.copy().mean(0)
#     snr = sig/n
#     mse_snr = np.round(np.mean(mse[:, 0]), 4)
#     var_snr = np.round(np.mean(var[:, 0]), 4)

#     return {'MSE': mse_snr, 'var': var_snr, 'sig_pow': sig, 'n_pow': n,
#             'snr': snr}


# s = time.time()
# GOF_ave = Parallel(n_jobs=-1)(delayed(SNR_trials_connectivity)(repeat, r, c,
#                                                                std)
#                               for r in R
#                               for std in std_pow)
# e = time.time()
# print(e-s)


# mse = np.zeros([len(R), len(std_pow)])
# var = np.zeros([len(R), len(std_pow)])
# snr = np.zeros([len(R), len(std_pow)])
# for i in range(len(R)):
#     for j in range(len(std_pow)):
#         mse[i, j] = np.abs(GOF_ave[i*len(std_pow)+j]['MSE'])
#         var[i, j] = GOF_ave[i*len(std_pow)+j]['var']
#         snr[i, j] = GOF_ave[i*len(std_pow)+j]['snr']

# SNR = 10*np.log10(np.mean(snr, 0))

# plt.rcParams['font.size'] = '14'
# plt.figure(figsize=(10, 5))
# for i in np.arange(0, len(R)):
#     plt.plot(SNR, mse[i, :], my_colors[i*2], label=str(R[i]))
#     plt.plot(SNR, mse[i, :], my_colors[i*2+1])
#     plt.legend(loc='upper right')
# plt.xlabel('SNR(db)')
# plt.ylabel('MSE')
# plt.title('Number of vertices: '+str(c))
# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNR_trials_effect_MSE_'+str(c))


# plt.figure(figsize=(10, 5))
# for i in np.arange(0, len(R)):
#     plt.plot(SNR, var[i, :], my_colors[i*2], label=str(R[i]))
#     plt.plot(SNR, var[i, :], my_colors[i*2+1])
#     plt.legend(loc='upper left')
# plt.xlabel('SNR(db)')
# plt.ylabel('Explained Variance')
# plt.title('Number of vertices: '+str(c))

# plt.savefig('/home/sr05/Method_dev/method_fig/simulation_SNR_trials_effect_var_'+str(c))
