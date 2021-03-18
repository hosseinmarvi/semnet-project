#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:05:45 2020

@author: sr05
"""
import os
import mne
import time
import pickle
import numpy as np
import SN_config as C
from joblib import Parallel, delayed
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.model_selection import (cross_validate, KFold)
from sklearn.linear_model import RidgeCV

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_MRI
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


def SN_transformation_main(cond, ROI_y, ROI_x, i, normalize):
    print('***Running SN_transformation_main...')
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    file_name = os.path.expanduser('~') + '/my_semnet/json_files/transformation/trans_' + \
        cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i) + \
        '_'+str(t_down_sampling)+'_bl.json'
    # morph labels from fsaverage to each subject
    morphed_labels = mne.morph_labels(SN_ROI, subject_to=data_path+sub_to,
                                      subject_from='fsaverage', subjects_dir=data_path)

    # read epochs
    epo_name = data_path + meg + 'block_'+cond+'_words_epochs-epo.fif'

    epochs_cond = mne.read_epochs(epo_name, preload=True)

    # crop epochs
    epochs = epochs_cond['words'].copy(
    ).crop(-.200, .900).resample(f_down_sampling)

    # equalize trial counts to eliminate bias
    # equalize_epoch_counts([epochs_SD, epochs_LD])

    inv_fname_epoch = data_path + meg + 'InvOp_'+cond+'_EMEG-inv.fif'

    print('***Running SN_transformation_io...')
    output = SN_transformation_io(epochs, inv_fname_epoch, morphed_labels,
                                  sub_to, ROI_x, ROI_y)
    print('***Running SN_transformation...')
    GOF_ave = SN_transformation(output[0], output[1], normalize)
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(GOF_ave, fp)
    return GOF_ave


def SN_transformation_io(epoch, inv_fname_epoch, labels, sub_to, ROI_x, ROI_y):
    output = [0]*2
    # read inverse operator,apply inverse operator
    inv_op = read_inverse_operator(inv_fname_epoch)
    stc = apply_inverse_epochs(epoch, inv_op, lambda2, method='MNE',
                               pick_ori="normal", return_generator=False)

    for j, idx in enumerate([ROI_x, ROI_y]):
        labels[idx].subject = sub_to
        # define dimentions of matrix (vertices X timepoints), & initializing
        v, t = stc[0].in_label(labels[idx]).data.shape
        X = np.zeros([len(stc), v, t])
        # create output array of size (vertices X stimuli X timepoints)
        for s in np.arange(0, len(stc)):
            S = stc[s].in_label(labels[idx]).data
            # X[s,:,:]=S.copy() -np.matlib.repmat(np.mean(S,0).reshape(1,t),v,1)
            X[s, :, :] = S

        output[j] = X
    return output


def SN_transformation(X, Y, normalize):
    # initialize the explained variance array of sizr timepoints X 1
    # GOF_ave=np.zeros([X.shape[-1],1])
    GOF_ave = {}
    # initialize the correlation coefficeint array of sizr vertices X 1
    GOF_explained_variance = np.zeros([X.shape[-1], X.shape[-1]])
    for t1 in np.arange(0, X.shape[-1]):
        for t2 in np.arange(0, X.shape[-1]):
            print('time: ', t1, t2)
            r = X.shape[0]
            if (r/5) > 10:
                n_splits = 10
            else:
                n_splits = 5
            kf = KFold(n_splits=n_splits)
            regrCV = RidgeCV(alphas=np.logspace(-5, 5, 100),
                             normalize=normalize)
            scores = cross_validate(
                regrCV, X[:, :, t2], Y[:, :, t1], scoring=(
                    'explained_variance'),
                cv=kf, n_jobs=-1)

            GOF_explained_variance[t1, t2] = np.mean(
                scores['test_score'])

    GOF_ave['explained_variance'] = GOF_explained_variance

    return GOF_ave


# GOF=[[[0]*6 for x in range(6)]for c in range(2)]
Cond = ['fruit', 'odour', 'milk', 'LD']
# Cond = ['LD']

normalize = True

# def SN_transformation_main_parallel(cond):
#     GOF_ave=Parallel(n_jobs=-1)(delayed(SN_transformation_main)\
#                             (cond, ROI_y, ROI_x, i, normalize)\
#                                 for ROI_y in range(6)\
#                                     for ROI_x in range(6) for i in range(18))

s = time.time()
GOF_ave = Parallel(n_jobs=-1)(delayed(SN_transformation_main)
                              (cond, ROI_y, ROI_x, i, normalize)
                              for cond in Cond
                              for ROI_y in range(6)
                              for ROI_x in range(6) for i in range(18))

e = time.time()
print(e-s)


# if len(sys.argv) == 1:

#     Cond = Cond


# for c in Cond:
#     SN_transformation_main_parallel(c)
