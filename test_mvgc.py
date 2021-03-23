#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:17:04 2021

@author: sr05
"""
import os
import matplotlib.pyplot as plt
import mne
import time
import pickle
import numpy as np
import sn_config as C
from joblib import Parallel, delayed
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()
# ROI_x=1
# ROI_y=0
s = time.time()
fs = 1000
f_down_sampling = 100  # 100Hz, 20Hz
t_down_sampling = fs/f_down_sampling  # 10ms, 50ms


def mvgc_main(cond, i, d, normalize):

    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]

    # morph labels from fsaverage to each subject
    morphed_labels = mne.morph_labels(SN_ROI, subject_to=data_path+sub_to,
                                      subject_from='fsaverage',
                                      subjects_dir=data_path)
    # read epochs
    epo_name = data_path + meg + 'block_'+cond+'_words_epochs-epo.fif'

    epochs_cond = mne.read_epochs(epo_name, preload=True)

    # crop epochs
    epochs = epochs_cond['words'].copy(
    ).crop(-.200, .900).resample(f_down_sampling)
    inv_fname_epoch = data_path + meg + 'InvOp_'+cond+'_EMEG-inv.fif'

    output_roi = mvgc_roi(epochs, inv_fname_epoch, morphed_labels, sub_to)

    for roi_x in C.rois_labels:
        for roi_y in C.rois_labels:
            mvgc_parallel(epochs, output_roi, roi_x, roi_y, d, i, cond,
                         normalize)

    # gc = np.zeros([len(epochs.times)-d, 1])
    # for t in np.arange(d, len(epochs.times)):
    #     y, x_total, x = mvgc_io(output_roi, roi_x, roi_y, t, d)
    #     gc[t-d, 0] = mvgc(y, x_total, x, roi_x, roi_y, normalize)
    # with open(file_name, "wb") as fp:  # Pickling
    #     pickle.dump(gc, fp)
    # return gc


def mvgc_roi(epoch, inv_fname_epoch, labels, sub_to):
    """
    Extracts all ROIs' patterns over time.

    Parameters
    ----------
    epoch : TYPE
        DESCRIPTION.
    inv_fname_epoch : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    sub_to : TYPE
        DESCRIPTION.

    Returns
    -------
    output : a dictionary constitutes of numpy arrays of size [stimuli*vertices
                                                               *time-points].

    """
    output = {}
    # read inverse operator,apply inverse operator
    inv_op = read_inverse_operator(inv_fname_epoch)
    stc = apply_inverse_epochs(epoch, inv_op, lambda2, method='MNE',
                               pick_ori="normal", return_generator=False)

    for lb, label in enumerate(labels):
        label.subject = sub_to
        v, t = stc[0].in_label(label).data.shape
        X = np.zeros([len(stc), v, t])
        for s in np.arange(0, len(stc)):
            X[s, :, :] = stc[s].in_label(label).data
        output[C.rois_labels[lb]] = X
    return output


def mvgc_io(output_roi, roi_x, roi_y, t, d):
    """
    Creates variables X and Y in equation: Y[t]=X_total[t-d:t-1]T1 and 
    Y[t]=X_total[t-d:t-1]T2, where X_total has the d-previous patterns 
    of all ROIs, while in X, patterns of ROI X is dropped. 
    Parameters
    ----------
    output_roi : dictionary
        Patterns of all ROIs over time.
    roi_x : string
        The ROI whose effect on ROI Y is estimated.
    roi_y : string
        The ROI on which the effects of ROI X is estimated.
    t : int
        time-point in which Y is estimated.
    d : int
        model order.

    Returns
    -------
    y : array of size [stimuli*vertices_ROI_y]
        Patterns of output at one single time-point.
    x_total : array of size [stimuli*(vertices_allROIs*d)]
        d previous Patterns of all ROIs as the input.
    x : array of size [stimuli*(vertices_allROIs_minusROI_X*d)]
        d previous Patterns of all ROIs minus ROI_x as the input.

    """
    y = output_roi[roi_y][:, :, t]
    labels = C.rois_labels.copy()
    labels.remove(roi_x)
    x_total = mvgc_x(output_roi, C.rois_labels, d, t)
    x = mvgc_x(output_roi, labels, d, t)
    return y, x_total, x


def mvgc_x(output_roi, labels, d, t):
    """
    Creates input x_total as d previous patterns of ROIs in labels

    Parameters
    ----------
    output_roi : dictionary
        DESCRIPTION.
    labels : label
        DESCRIPTION.
    d : int
        DESCRIPTION.
    t : int
        DESCRIPTION.

    Returns
    -------
    x_total :  array of size [stimuli*(vertices_labelsROIs*d)]
        d previous Patterns of all ROIs in labels as the input.

    """
    for r, roi in enumerate(labels):
        x_roi = output_roi[roi][:, :, t -
                                d:t].reshape([output_roi[roi].shape[0],
                                              output_roi[roi].shape[1]*d])
        if r == 0:
            x_total = x_roi
        else:
            x_total = np.append(x_total, x_roi, axis=1)
    return x_total


def mvgc(y, x_total, x, roi_x, roi_y, normalize):
    """
    Computes Granger Causality value.

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x_total : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    normalize : TYPE
        DESCRIPTION.

    Returns
    -------
    gc_total : TYPE
        DESCRIPTION.

    """
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


def mvgc_parallel(epochs, output_roi, roi_x, roi_y, d, i, cond, normalize):
    file_name = os.path.expanduser('~') + '/my_semnet/json_files/GC/gc_' + \
        cond+'_x'+str(roi_x)+'-y'+str(roi_y)+'_sub_'+str(i) + \
        '_d_'+str(d)+'.json'
    gc = np.zeros([len(epochs.times)-d, 1])
    for t in np.arange(d, len(epochs.times)):
        y, x_total, x = mvgc_io(output_roi, roi_x, roi_y, t, d)
        gc[t-d, 0] = mvgc(y, x_total, x, roi_x, roi_y, normalize)
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(gc, fp)


Cond = ['fruit', 'odour', 'milk', 'LD']

d = 5
normalize = True



s = time.time()
gc = Parallel(n_jobs=-1)(delayed(mvgc_main)
                                (cond, i, d, normalize)
                         for cond in Cond
                         for i in range(18))

e = time.time()
print(e-s)

# for i in np.arange(0, 100):
#     if gc_matrix[i] < 0:
#         gc_matrix[i] = 0


# plt.plot(gc_matrix)
