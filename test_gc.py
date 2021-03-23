#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:56:29 2021

@author: sr05
"""

import os
import mne
import sys
import time
import pickle
import numpy as np
import sn_config as C
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import linear_model
from joblib import Parallel, delayed
from mne.epochs import equalize_epoch_counts
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import (train_test_split, KFold)
from sklearn.linear_model import RidgeCV
import multiprocessing
from functools import partial
from mne.stats import permutation_cluster_1samp_test, f_threshold_mway_rm,\
    summarize_clusters_stc, permutation_cluster_test,\
    f_mway_rm
from scipy import stats as stats
from matplotlib.colors import LinearSegmentedColormap
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

d = 5
X_SD = np.zeros([len(C.subjects), 6, 6, 105])
X_LD = np.zeros([len(C.subjects), 6, 6, 105])

for y, roi_y in enumerate(C.rois_labels):
    for x, roi_x in enumerate(C.rois_labels):
        if roi_y != roi_x:
            for i in np.arange(0, len(C.subjects)):
                print(y, x, i)
                gc = 0
                for cond in ['fruit', 'odour', 'milk']:
                    file_name = os.path.expanduser('~') + '/my_semnet/json_files/GC/gc_' + cond+'_x'+str(roi_x)+'-y'+str(roi_y)+'_sub_'+str(i) + \
                        '_d_'+str(d)+'.json'
                    with open(file_name, "rb") as fp:   # Unpickling
                        a = pickle.load(fp)
                    gc = gc + a

                    X_SD[i, y, x, :] = (gc/3).reshape(1, 105)

                for cond in ['LD']:
                    file_name = os.path.expanduser('~') + '/my_semnet/json_files/GC/gc_' + cond+'_x'+str(roi_x)+'-y'+str(roi_y)+'_sub_'+str(i) + \
                        '_d_'+str(d)+'.json'
                    with open(file_name, "rb") as fp:   # Unpickling
                        a = pickle.load(fp)
                    X_LD[i, y, x, :] = (a).reshape(1, 105)


for t in np.arange(0, 105):
    fig = plt.figure()
    im = plt.imshow(np.mean(X_SD[:, :, :, t].copy(), 0),
                    extent=[1, 6, 1, 6], aspect='equal',
                    origin='lower')
    plt.colorbar()
    # ax[0].set_title('SD')
    # ax[0, 0].set_ylabel(lb[ROI_y] + ' = '+lb[ROI_x] + ' * T', fontsize=14)

# plt.close('all')
