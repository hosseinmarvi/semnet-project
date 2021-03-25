#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:26:44 2021

@author: sr05
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate, KFold


"""
===========================================================================
1. Transformation between two random noise (two independent variables)
===========================================================================
"""


def noise_connectivity_main(trials, vertices, repeat, normalize):
    # number of vertices
    noise_colors = ['ro', 'r--', 'bs', 'b--', 'gx', 'g--']

    gof = Parallel(n_jobs=-1)(delayed(noise_connectivity)
                              (t, v, repeat, normalize)
                              for v in vertices for t in trials)

    # extracts metrics from GOF_ave for different vertices (C) and trials(R)
    mse = np.zeros([len(vertices), len(trials)])
    explained_var = np.zeros([len(vertices), len(trials)])
    for i in range(len(vertices)):
        for j in range(len(trials)):
            mse[i, j] = np.abs(gof[i * len(trials) + j]['MSE'])
            explained_var[i, j] = gof[i * len(trials) + j]['explained_var']

    noise_connectivity_plot(vertices, trials, mse, noise_colors,
                            'Number of Trials', 'MSE')
    noise_connectivity_plot(vertices, trials, explained_var, noise_colors,
                            'Number of Trials', 'Explained Variance')


def noise_connectivity(trial, vertex, repeat, normalize):
    """
    this function computes the connectivity between 2 random noises using
    Ridge Regression and k-fold cross-validation.

    Parameters
    ----------
    normalize : bool

    vertex : int
        column-number of vertices
    trial : int
        row-number of trials
    repeat : int

    Returns
    -------
    MSE and explained_variance

    """
    mse = np.zeros([repeat])
    explained_var = np.zeros([repeat])

    for i in np.arange(repeat):
        print(f'trial, vertex, i: {trial}, {vertex}, {i}')
        x = np.random.normal(0, 1, (trial, vertex))
        y = np.random.normal(0, 1, (trial, vertex))
        explained_var[i], mse[i] = transformation(trial, x, y, normalize)

    return {'MSE': np.mean(mse), 'explained_var': np.mean(explained_var)}


def noise_connectivity_plot(vertices, x_axis, y_axis, colors,
                            x_label, y_label):
    plt.rcParams['font.size'] = '14'
    plt.figure(figsize=(11, 5))
    for i in np.arange(len(vertices)):
        plt.plot(x_axis, y_axis[i], colors[i * 2], label=str(vertices[i]))
        plt.plot(x_axis, y_axis[i], colors[i * 2 + 1])
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.savefig(f'/home/sr05/Method_dev/method_fig'
    #             f'/noise_connectivity_{y_label}')


"""
===========================================================================
2. Linear Transformation between two variables. 
   Investigationg the effects of different number of trials and SNRs
===========================================================================
"""


def linear_connectivity_main(vertices, trials, repeat, normalize):
    std_pow = [2.5, 2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2]
    connectivity_colors = ['bo', 'b--', 'yo', 'y--', 'go', 'g--',
                           'm*', 'm--', 'bs', 'b--', 'rs', 'r--', 'gs', 'g--']
    gof = Parallel(n_jobs=-1)(
        delayed(linear_connectivity)(vertex, t,  repeat, normalize, std)
        for vertex in vertices
        for t in trials
        for std in std_pow)

    mse = np.zeros([len(vertices), len(trials), len(std_pow)])
    explained_var = np.zeros([len(vertices), len(trials), len(std_pow)])
    snr = np.zeros([len(vertices), len(trials), len(std_pow)])
    # extracts metrics from gof for different vertices and trials
    for v in range(len(vertices)):
        for t in range(len(trials)):
            for s in range(len(std_pow)):
                mse[v, t, s] = np.abs(
                    gof[v * len(trials)*len(std_pow) + t * len(std_pow) + s]['MSE'])
                explained_var[v, t, s] = \
                    gof[v * len(trials)*len(std_pow) + t *
                        len(std_pow) + s]['explained_var']
                snr[v, t, s] = gof[v *
                                   len(trials)*len(std_pow) + t * len(std_pow) + s]['SNR']

    # snr = 10 * np.log10(np.mean(snr, 0))
    snr_db = 10 * np.log10(snr)
    linear_connectivity_plot(vertices, trials, snr_db, mse,
                             connectivity_colors, 'SNR(db)', 'MSE')
    linear_connectivity_plot(vertices, trials, snr_db, explained_var,
                             connectivity_colors, 'SNR(db)',
                             'Explained Variance')


def linear_connectivity(vertex, trial,  repeat, normalize, std):
    """
    this function computes the connectivity between 2 linearly-related matrices
    using Ridge Regression and k-fold cross-validation.

    Parameters
    ----------
    normalize : bool

    repeat : int
    trial : int
        row-number of trials
    vertex : int
        column-number of vertices
    std : float

    Returns
    -------
    dict
        MSE, explained_variance, signal power, noise power, and SNR.

    """
    mse = np.zeros([repeat, 1])
    explained_var = np.zeros([repeat, 1])
    noise_var = np.zeros([repeat, 1])
    signal_var = np.zeros([repeat, 1])
    noise = np.random.normal(0, 10 ** std, (trial, vertex))
    coef_matrix = np.random.normal(0, 1, (vertex, vertex))
    for i in np.arange(repeat):
        # print(f'trial, vertex, i: {trial}, {vertex}, {i}')
        x = np.random.normal(0, 1, (trial, vertex))
        y = np.matmul(x, coef_matrix)
        y = y+noise
        noise_var[i, 0] = np.var(noise)
        signal_var[i, 0] = np.var(np.matmul(x, coef_matrix))
        explained_var[i], mse[i] = transformation(trial, x, y, normalize)

    signal_var_avg = signal_var.copy().mean(0)
    noise_var_avg = noise_var.copy().mean(0)
    snr = signal_var_avg/noise_var_avg
    mse_snr = np.round(np.mean(mse[:, 0]), 4)
    explained_var_snr = np.round(np.mean(explained_var[:, 0]), 4)

    return {'MSE': mse_snr, 'explained_var': explained_var_snr,
            'sig_pow': signal_var_avg, 'noise_var': noise_var_avg, 'SNR': snr}


def linear_connectivity_plot(vertices, trials, x_axis, y_axis, colors,
                             x_label, y_label):

    for v in np.arange(len(vertices)):
        plt.rcParams['font.size'] = '14'
        plt.figure(figsize=(11, 5))
        for t in np.arange(len(trials)):
            # print( v , t)
            plt.plot(x_axis[v, t, :], y_axis[v, t, :], colors[t * 2],
                     label=str(trials[t]))
            plt.plot(x_axis[v, t, :], y_axis[v, t, :], colors[t * 2 + 1])
            plt.legend(loc='upper right')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Transformation: Vertex({vertices[v]})')
        # plt.savefig(f'/home/sr05/Method_dev/method_fig'
        #             f'/linear_connectivity_{y_label}_v{vertices[i]}')


def transformation(trial, x, y, normalize):
    n_splits = 5
    if (trial / 5) > 10:
        n_splits = 10
    kf = KFold(n_splits=n_splits)
    regr_cv = RidgeCV(alphas=np.logspace(-5, 5, 100),
                      normalize=normalize)
    scores = cross_validate(
        regr_cv, x, y,
        scoring=('explained_variance', 'neg_mean_squared_error'), cv=kf,
        n_jobs=-1, return_train_score=True
    )

    return (np.mean(scores['test_explained_variance']),
            np.mean(np.abs(scores['test_neg_mean_squared_error'])))


if __name__ == '__main__':
    n_repeats = 100
    trials_list = [30, 50, 100, 150, 300, 500, 1000]

    vertices_list = [50, 150, 100]
    normalization = True
    linear_connectivity_main(vertices_list, trials_list, n_repeats,
                             normalization)
    noise_connectivity_main(trials_list, vertices_list, n_repeats,
                            normalization)
# plt.close('all')
