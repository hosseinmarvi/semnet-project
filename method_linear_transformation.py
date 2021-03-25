#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:05:45 2020

@author: sr05
"""
import os
import sys

import mne
import time
import pickle
import numpy as np
import sn_config as c
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from SN_semantic_ROIs import SN_semantic_ROIs
from sklearn.model_selection import cross_validate, KFold
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator


# path to raw data
data_path = c.data_path
main_path = c.main_path
subjects = c.subjects
mri_sub = c.subjects_mri

# Parameters
lambda2 = c.lambda2_epoch
label_path = c.label_path
roi = SN_semantic_ROIs()
fs = 1000
f_down_sampling = 20  # 100Hz, 20Hz
t_down_sampling = fs / f_down_sampling  # 10ms, 50ms


def method_linear_transformation_main(cond, roi_y, roi_x, i, normalize):
    print('***Running sn_transformation_main...')
    meg = subjects[i]
    sub_to = mri_sub[i][1:15]

    # file_name to save the output
    file_name = os.path.expanduser('~') + \
        '/my_semnet/json_files/transformation/trans_' + cond + '_x' + \
        str(roi_x) + '-y' + str(roi_y) + '_sub_' + str(i) + '_' + \
        str(t_down_sampling) + '_bl.json'

    # morph labels from fsaverage to each subject
    morphed_labels = mne.morph_labels(roi, subject_to=data_path + sub_to,
                                      subject_from='fsaverage',
                                      subjects_dir=data_path)

    # read,crop and resample epochs
    epoch_name = data_path + meg + 'block_' + cond + '_words_epochs-epo.fif'
    epoch_condition = mne.read_epochs(epoch_name, preload=True)
    epochs = epoch_condition['words'].copy().crop(-.200, .900) \
        .resample(f_down_sampling)

    # inverse operator
    inverse_fname_epoch = data_path + meg + 'InvOp_' + cond + '_EMEG-inv.fif'

    print('***Running SN_transformation_io...')
    # prepares the patterns of roi_x and roi_y over time
    output = method_linear_transformation_io(epochs, inverse_fname_epoch,
                                             morphed_labels, sub_to, roi_x,
                                             roi_y)
    print('***Running SN_transformation...')
    # computes the connectivity (explained_variance) of the patterns of roi_x
    # and roi_y over time
    gof = method_linear_transformation(output[0], output[1], normalize)
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(gof, fp)
    return gof


def method_linear_transformation_io(epoch, inv_fname_epoch, labels, sub_to,
                                    roi_x, roi_y):
    # extracts patterns of roi_x and roi_y over time
    output = [0] * 2
    # read and apply inverse operator
    inverse_operator = read_inverse_operator(inv_fname_epoch)
    stc = apply_inverse_epochs(epoch, inverse_operator, lambda2, method='MNE',
                               pick_ori="normal", return_generator=False)

    for i, roi_idx in enumerate([roi_x, roi_y]):
        labels[roi_idx].subject = sub_to
        # defines matrix dimensions (vertices x timepoints), & initializing
        n_vertices, n_timepoints = stc[0].in_label(labels[roi_idx]).data.shape
        x = np.zeros([len(stc), n_vertices, n_timepoints])
        # creates output array of size (trials x vertices x timepoints)
        for n_trial, stc_trial in enumerate(stc):
            pattern = stc_trial.in_label(labels[roi_idx]).data
            x[n_trial, :, :] = pattern

        output[i] = x
    return output


def method_linear_transformation(x, y, normalize):
    # computes the explained_variance of different latencies

    gof = {}
    # initialize the explained variance array of size n_timepoints X n_timepoints
    gof_explained_variance = np.zeros([x.shape[-1], x.shape[-1]])
    for t1 in np.arange(x.shape[-1]):
        for t2 in np.arange(x.shape[-1]):
            print('timepoint: ', t1, t2)
            n_trials = x.shape[0]
            n_splits = 5
            if (n_trials / 5) > 10:
                n_splits = 10
            kf = KFold(n_splits=n_splits)
            regr_cv = RidgeCV(alphas=np.logspace(-5, 5, 100),
                              normalize=normalize)
            scores = cross_validate(regr_cv, x[:, :, t2], y[:, :, t1],
                                    scoring='explained_variance',
                                    cv=kf, n_jobs=-1)

            gof_explained_variance[t1, t2] = np.mean(scores['test_score'])

    gof['explained_variance'] = gof_explained_variance

    return gof


if __name__ == '__main__':
    option = ''
    while option not in ['0', '1']:
        option = input('Would you like to run it on the cluster?\n'
                       '[0]: No\n'
                       '[1]: Yes\n')
    conditions = ['fruit', 'odour', 'milk', 'LD']
    normalization = True
    if option == '0':
        start = time.time()
        Parallel(n_jobs=-1)(delayed(method_linear_transformation_main)
                            (cond, roi_y, roi_x, i, normalization)
                            for cond in conditions
                            for roi_y in range(len(roi))
                            for roi_x in range(len(roi))
                            for i in range(len(subjects)))
        end = time.time()
        print(end - start)

    # options = 1
    else:
        if len(sys.argv) == 1:
            conditions = conditions
        for cond in conditions:
            Parallel(n_jobs=-1)(delayed(method_linear_transformation_main)
                                (cond, roi_y, roi_x, i, normalization)
                                for roi_y in range(len(roi))
                                for roi_x in range(len(roi))
                                for i in range(len(subjects)))
