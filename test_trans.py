#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:19:42 2021

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

t_down_sampling = 50.0


def mask_function(X, cut_off=None):
    if cut_off is not None:
        r, c = X.shape
        for i in np.arange(0, r):
            for j in np.arange(0, c):
                if X[i, j] < cut_off:
                    X[i, j] = 0
    return X


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

# Cond= 'LD'
labels = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']

# for cond in ['fruit','odour','milk','LD']:
#     for ROI_y in np.arange(0,1):
#         for ROI_x in np.arange(1,2):
#             GOF_r2score=[0]

#             for i in np.arange(0,len(C.subjects)):

#                 # file_name=os.path.expanduser('~') +'/my_semnet/json_files/transformation/trans_'+Cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'_10.json'
#                 # with open(file_name, "rb") as fp:   # Unpickling
#                 #    a  = pickle.load(fp)
#                 #    GOF_r2score_10=GOF_r2score_10 + a['r2score']

#                 file_name=os.path.expanduser('~') +'/my_semnet/json_files/transformation/trans_'+cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'_50.json'
#                 with open(file_name, "rb") as fp:   # Unpickling
#                    a  = pickle.load(fp)
#                    GOF_r2score=GOF_r2score + a['r2score']


#             print(cond + str(ROI_y) + str(ROI_x))
#             # x_pos = np.arange(15)
#             # fig, ax = plt.subplots(figsize=[9,9])

#               # Set general font size
#             plt.rcParams['font.size'] = '10'
#             r2score = np.round(GOF_r2score.copy()/len(subjects),4)
#             # ax.matshow(leakage_ave, cmap=plt.cm.GnBu)
#             plt.matshow(r2score, cmap=plt.cm.seismic,vmin=-np.max(r2score), vmax=np.max(r2score), extent=[-0,900,900,0])
#             plt.title(cond+ '( Y: '+labels[ROI_y]+'/ X: '+ labels[ROI_x]+' )')
#             plt.colorbar()

#     # plt.close('all')

###########################################################

# GOF='r2score'
GOF = 'explained_variance'
total_max = []
total_min = []
for ROI_y in np.arange(0, 6):
    for ROI_x in np.arange(0, 6):
        if ROI_y != ROI_x:
            GOF_r2score_SD = 0
            GOF_r2score_LD = 0
            for i in np.arange(0, len(C.subjects)):
                GOF_r2score = 0
                for cond in ['fruit', 'odour', 'milk']:
                    file_name = os.path.expanduser('~') + '/my_semnet/json_files/transformation/trans_' + \
                        cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i) + \
                        '_'+str(t_down_sampling)+'_bl.json'
                    with open(file_name, "rb") as fp:   # Unpickling
                        a = pickle.load(fp)
                        GOF_r2score = GOF_r2score + a[GOF]

                GOF_r2score_SD = GOF_r2score_SD + GOF_r2score

                for cond in ['LD']:
                    file_name = os.path.expanduser('~') + '/my_semnet/json_files/transformation/trans_' + \
                        cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i) + \
                        '_'+str(t_down_sampling)+'_bl.json'
                    with open(file_name, "rb") as fp:   # Unpickling
                        a = pickle.load(fp)
                        GOF_r2score_LD = GOF_r2score_LD + a[GOF]

            r2score_SD = np.round(GOF_r2score_SD/(len(subjects)*3), 4)
            r2score_LD = np.round(GOF_r2score_LD/(len(subjects)), 4)
            v_max = max(np.max(r2score_SD), np.max(r2score_LD))
            v_min = min(np.min(r2score_SD), np.min(r2score_LD))

            # v_max = np.max(r2score_SD)
            # v_min = np.min(r2score_SD)
            vmax = max(abs(v_max), abs(v_min))
            total_max.append(vmax)
            total_min.append(v_min)

# for ROI_y in np.arange(0,1):
#     for ROI_x in np.arange(0,6):
#         if ROI_y!=ROI_x:
#             GOF_r2score_SD=0
#             GOF_r2score_LD=0
#             for i in np.arange(0,len(C.subjects)):
#                 GOF_r2score=0
#                 for cond in ['fruit','odour','milk']:
#                     file_name=os.path.expanduser('~') +'/my_semnet/json_files/transformation/trans_'+cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'_50.json'
#                     with open(file_name, "rb") as fp:   # Unpickling
#                         a  = pickle.load(fp)
#                     GOF_r2score=GOF_r2score + a[GOF]

#                 GOF_r2score_SD= GOF_r2score_SD + GOF_r2score

#                 for cond in ['LD']:
#                     file_name=os.path.expanduser('~') +'/my_semnet/json_files/transformation/trans_'+cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'_50.json'
#                     with open(file_name, "rb") as fp:   # Unpickling
#                         a  = pickle.load(fp)
#                     GOF_r2score_LD=GOF_r2score_LD + a[GOF]

#         print(cond + str(ROI_y) + str(ROI_x))
#         # Set general font size
#         plt.rcParams['font.size'] = '10'
#         r2score_SD = np.round(GOF_r2score_SD.copy()/(len(subjects)*3),4)
#         r2score_LD = np.round(GOF_r2score_LD.copy()/(len(subjects)),4)
#         # v_max= max(np.max(r2score_SD),np.max(r2score_LD))
#         # v_min= min(np.min(r2score_SD),np.min(r2score_LD))
#         # vmax=max(abs(v_max),abs(v_min))
#         vmax=max(total_max)
#         # ax.matshow(leakage_ave, cmap=plt.cm.GnBu)
#         plt.matshow(r2score_SD, cmap=plt.cm.seismic,vmin=-vmax, vmax=vmax, extent=[0,900,0,900], origin='lower')
#         plt.title('SD ( Y: '+labels[ROI_y]+'/ X: '+ labels[ROI_x]+' )')
#         plt.colorbar()


#         plt.rcParams['font.size'] = '10'
#         # ax.matshow(leakage_ave, cmap=plt.cm.GnBu)
#         plt.matshow(r2score_LD, cmap=plt.cm.seismic,vmin=-vmax, vmax=vmax, extent=[0,900,0,900], origin='lower')
#         plt.title('LD ( Y: '+labels[ROI_y]+'/ X: '+ labels[ROI_x]+' )')
#         plt.colorbar()


# # plt.close('all')

###########################################################################
# d = 22
# X_SD = np.zeros([len(C.subjects), 30, d, d])
# X_LD = np.zeros([len(C.subjects), 30, d, d])
# cut_off = 0
# p = 0
# for ROI_y in np.arange(0, 6):
#     for ROI_x in np.arange(0, 6):
#         if ROI_y != ROI_x:

#             print(p)
#             for i in np.arange(0, len(C.subjects)):
#                 GOF_SD = np.zeros([d, d])
#                 for cond in ['fruit', 'odour', 'milk']:
#                     file_name = os.path.expanduser('~') + '/my_semnet/json_files/transformation/trans_'+cond+'_x'+str(
#                         ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'_50_bl.json'
#                     with open(file_name, "rb") as fp:   # Unpickling
#                         a = pickle.load(fp)
#                     GOF_SD = GOF_SD + a[GOF]

#                 X_SD[i, p, :, :] = mask_function(GOF_SD/3, cut_off)
#                 # X_SD[i, p, :, :] = mask_function(GOF_SD/3)

#                 for cond in ['LD']:
#                     file_name = os.path.expanduser('~') + '/my_semnet/json_files/transformation/trans_'+cond+'_x'+str(
#                         ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'_50_bl.json'
#                     with open(file_name, "rb") as fp:   # Unpickling
#                         a = pickle.load(fp)
#                     X_LD[i, p, :, :] = mask_function(a[GOF], cut_off)
#                     # X_LD[i, p, :, :] = mask_function(a[GOF])

#             p += 1

d = 22
X_SD = np.zeros([len(C.subjects), 36, d, d])
X_LD = np.zeros([len(C.subjects), 36, d, d])
cut_off = 0.0
p = 0
for ROI_y in np.arange(0, 6):
    for ROI_x in np.arange(0, 6):
        # if ROI_y != ROI_x:

        print(p)
        for i in np.arange(0, len(C.subjects)):
            GOF_SD = np.zeros([d, d])
            for cond in ['fruit', 'odour', 'milk']:
                file_name = os.path.expanduser('~') + '/my_semnet/json_files/transformation/trans_' + \
                    cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i) + \
                    '_'+str(t_down_sampling)+'_bl.json'
                with open(file_name, "rb") as fp:   # Unpickling
                    a = pickle.load(fp)
                GOF_SD = GOF_SD + a[GOF]

            # X_SD[i, p, :, :] = mask_function(GOF_SD/3, cut_off)
            X_SD[i, p, :, :] = mask_function(GOF_SD/3)

            for cond in ['LD']:
                file_name = os.path.expanduser('~') + '/my_semnet/json_files/transformation/trans_' + \
                    cond+'_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i) + \
                    '_'+str(t_down_sampling)+'_bl.json'
                with open(file_name, "rb") as fp:   # Unpickling
                    a = pickle.load(fp)
                # X_LD[i, p, :, :] = mask_function(a[GOF], cut_off)
                X_LD[i, p, :, :] = mask_function(a[GOF])

        p += 1
#############################################################################
# lb = C.ROIs_lables
# t1, t2 = [-200, 900]
# # # difference of SD (0:6) and LD(6:12) for aech ROI and individual
# Z = X_SD-X_LD
# # Z= X_SD

# tail = 0
# t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., len(C.subjects) - 1)
# k = 0
# for ROI_y in np.arange(0, 6):
#     for ROI_x in np.arange(0, 6):
#         if ROI_y != ROI_x:
#             T_obs, clusters, cluster_p_values, H0 = \
#                 permutation_cluster_1samp_test(Z[:, k, :, :], n_permutations=C.n_permutations,
#                                                threshold=t_threshold, tail=tail, out_type='mask',
#                                                verbose=True)

#             T_obs_plot = np.nan * np.ones_like(T_obs)
#             for c, p_val in zip(clusters, cluster_p_values):
#                 if p_val <= C.pvalue:
#                     T_obs_plot[c] = T_obs[c]

#             # plt.figure(figsize=(3,10))
#             fig, ax = plt.subplots(2, 2, figsize=(8, 8))
#             # # plotting the t-values
#             vmax = max(total_max)
#             # plt.subplot(311)
#             im = ax[0, 0].imshow(np.mean(X_SD[:, k, :, :].copy(), 0), cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2], aspect='equal',
#                                  origin='lower', vmin=0, vmax=vmax)
#             fig.colorbar(im, ax=ax[0, 0])
#             ax[0, 0].set_ylabel('SD')
#             fig.suptitle('Y: '+lb[ROI_y] + ' | X: '+lb[ROI_x])

#             im = ax[0, 1].imshow(np.mean(X_LD[:, k, :, :].copy(), 0), cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2],aspect='equal',
#                                  origin='lower', vmin=0, vmax=vmax)
#             fig.colorbar(im, ax=ax[0, 1])
#             ax[0, 1].set_ylabel('LD')
#             # ax[0,1].title('Power of '+lb[k])

#             im = ax[1, 0].imshow(np.mean(Z[:, k, :, :].copy(), 0), cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2],
#                                  aspect='equal', origin='lower', vmin=-vmax, vmax=vmax)
#             fig.colorbar(im, ax=ax[1, 0])
#             ax[1, 0].set_ylabel('SD-LD')
#             # ax[0,1].title('Power of '+lb[k])

#             vmax = np.max(np.nan_to_num(T_obs))
#             vmin = np.min(np.nan_to_num(T_obs))
#             v = max(abs(vmax), abs(vmin))
#             # plt.subplot(313)
#             ax[1, 1].imshow(T_obs, cmap=plt.cm.gray,
#                             extent=[t1, t2, t1, t2],
#                             aspect='equal', origin='lower', vmin=-v, vmax=v)
#             im = ax[1, 1].imshow(T_obs_plot, cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2],
#                                  aspect='equal', origin='lower', vmin=-v, vmax=v)
#             fig.colorbar(im, ax=ax[1, 1])
#             ax[1, 1].set_ylabel('t-test')
#             # plt.title('Power of '+lb[k])

#             k += 1
# # plt.close('all')

#############################################################################
# import matplotlib
# cmap = matplotlib.colors.ListedColormap(['Blues', 'YlOrRd'])
# cmap.set_over('darkred')
# cmap.set_under('black')
# bounds = [min(total_min), 0, max(total_max)]

# norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
# cb = matplotlib.colorbar.ColorbarBase(ax[0,1], cmap=cmap,
#                                 norm=norm,
#                                 boundaries=bounds,
#                                 extend='both',
#                                 extendfrac='auto',
#                                 ticks=bounds,
#                                 spacing='uniform',
#                                 orientation='vertical')


lb = C.rois_labels
t1, t2 = [-200, 900]
# # difference of SD (0:6) and LD(6:12) for aech ROI and individual
Z = X_SD-X_LD
# Z= X_SD
# colors = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]  # R -> G -> B
# colors = ['black','cadetblue','red','orange','yellow','white']
colors = ['black', 'cadetblue', 'darkred', 'red', 'darkorange',
          'orange', 'gold', 'khaki', 'yellow', 'lightyellow', 'white']

colors2 = ['blue', 'green', 'white', 'yellow', 'red']
colors3 = ['black', 'gray']

# colors = ['black','teal','red','orange','yellow','white']  # R -> G -> B

cmap_name = 'my_list'
n_bin = 100
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
cm2 = LinearSegmentedColormap.from_list(cmap_name, colors2, N=n_bin)
cm3 = LinearSegmentedColormap.from_list(cmap_name, colors3, N=n_bin)

tail = 0
t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., len(C.subjects) - 1)
k = 0
for ROI_y in np.arange(0, 1):
    for ROI_x in np.arange(ROI_y+1, 6):
    # for ROI_x in np.arange(2, 3):

        k1 = ROI_y*6+ROI_x
        k2 = ROI_x*6+ROI_y
        print(k1, k2)
        Z1 = X_SD[:, k1, :, :]-X_LD[:, k1, :, :]
        Z2 = X_SD[:, k2, :, :]-X_LD[:, k2, :, :]

        T_obs1, clusters1, cluster_p_values1, H01 = \
            permutation_cluster_1samp_test(Z1, n_permutations=C.n_permutations,
                                           threshold=t_threshold, tail=tail, out_type='mask',
                                           verbose=True)

        T_obs_plot1 = np.nan * np.ones_like(T_obs1)
        for c, p_val in zip(clusters1, cluster_p_values1):
            if p_val <= C.pvalue:
                T_obs_plot1[c] = T_obs1[c]

        T_obs2, clusters2, cluster_p_values2, H02 = \
            permutation_cluster_1samp_test(Z2, n_permutations=C.n_permutations,
                                           threshold=t_threshold, tail=tail, out_type='mask',
                                           verbose=True)

        T_obs_plot2 = np.nan * np.ones_like(T_obs2)
        for c, p_val in zip(clusters2, cluster_p_values2):
            if p_val <= C.pvalue:
                T_obs_plot2[c] = T_obs2[c]

        # plt.figure(figsize=(3,10))
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))

        # # plotting the t-values
        # vmax = max(total_max)
        # vmin = min(total_min)

        vmax = 1
        vmin = -.10
        # plt.subplot(311)
        im = ax[0, 0].imshow(np.mean(X_SD[:, k1, :, :].copy(), 0), cmap=cm,
                             extent=[t1, t2, t1, t2], aspect='equal',
                             origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[0, 0])
        ax[0, 0].set_title('SD')

        ax[0, 0].set_ylabel(lb[ROI_y] + ' = '+lb[ROI_x] + ' * T', fontsize=14)
        # fig.suptitle('Y: '+lb[ROI_y] + ' | X: '+lb[ROI_x])

        im = ax[0, 1].imshow(np.mean(X_LD[:, k1, :, :].copy(), 0), cmap=cm,
                             extent=[t1, t2, t1, t2], aspect='equal',
                             origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[0, 1])
        ax[0, 1].set_title('LD')
        # ax[0,1].title('Power of '+lb[k])

        im = ax[1, 0].imshow(np.mean(X_SD[:, k2, :, :].copy(), 0), cmap=cm,
                             extent=[t1, t2, t1, t2],
                             aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[1, 0])
        ax[1, 0].set_title('SD')

        ax[1, 0].set_ylabel(lb[ROI_x] + ' = '+lb[ROI_y] + ' * T', fontsize=14)

        im = ax[1, 1].imshow(np.mean(X_LD[:, k2, :, :].copy(), 0), cmap=cm,
                             extent=[t1, t2, t1, t2],
                             aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[1, 1])
        ax[1, 1].set_title('LD')
        # ax[0,1].title('Power of '+lb[k])

        vmax = np.max(np.nan_to_num(T_obs1))
        vmin = np.min(np.nan_to_num(T_obs1))
        v = max(abs(vmax), abs(vmin))
        # plt.subplot(313)
        ax[0, 2].imshow(T_obs1, cmap=cm3,
                        extent=[t1, t2, t1, t2],
                        aspect='equal', origin='lower', vmin=-v, vmax=v)
        im = ax[0, 2].imshow(T_obs_plot1, cmap=cm2,
                             extent=[t1, t2, t1, t2],
                             aspect='equal', origin='lower', vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax[0, 2])
        ax[0, 2].set_title('t-test')

        vmax = np.max(np.nan_to_num(T_obs2))
        vmin = np.min(np.nan_to_num(T_obs2))
        v = max(abs(vmax), abs(vmin))
        # plt.subplot(313)
        ax[1, 2].imshow(T_obs2, cmap=cm3,
                        extent=[t1, t2, t1, t2],
                        aspect='equal', origin='lower', vmin=-v, vmax=v)
        im = ax[1, 2].imshow(T_obs_plot2, cmap=cm2,
                             extent=[t1, t2, t1, t2],
                             aspect='equal', origin='lower', vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax[1, 2])
        ax[1, 2].set_title('t-test')
        # plt.title('Power of '+lb[k])
        # plt.savefig('/home/sr05/Method_dev/method_fig/Linear_Transformation_'+lb[ROI_y] + '_'+lb[ROI_x])

        k += 1
# plt.close('all')

#############################################################################
# # yx vs xy
# lb = C.ROIs_lables
# t1, t2 = [-200, 900]
# # # difference of SD (0:6) and LD(6:12) for aech ROI and individual
# # Z= X_SD

# tail = 0
# t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., len(C.subjects) - 1)
# k = 0
# for ROI_y in np.arange(0, 6):
#     for ROI_x in np.arange(ROI_y+1, 6):
#         k1 = ROI_y*6+ROI_x
#         k2 = ROI_x*6+ROI_y
#         print(k1, k2)
#         Z = X_LD[:, k1, :, :]-X_LD[:, k2, :, :]
#         if ROI_y != ROI_x:
#             T_obs, clusters, cluster_p_values, H0 = \
#                 permutation_cluster_1samp_test(Z, n_permutations=C.n_permutations,
#                                                threshold=t_threshold, tail=tail, out_type='mask',
#                                                verbose=True)

#             T_obs_plot = np.nan * np.ones_like(T_obs)
#             for c, p_val in zip(clusters, cluster_p_values):
#                 if p_val <= C.pvalue:
#                     T_obs_plot[c] = T_obs[c]

#             # plt.figure(figsize=(3,10))
#             fig, ax = plt.subplots(2, 2, figsize=(9, 9))
#             # # plotting the t-values
#             vmax = max(total_max)
#             # plt.subplot(311)
#             im = ax[0, 0].imshow(np.mean(X_LD[:, k1, :, :].copy(), 0), cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2], aspect='equal', origin='lower', vmin=-vmax, vmax=vmax)
#             fig.colorbar(im, ax=ax[0, 0])
#             ax[0, 0].set_ylabel('Y: '+lb[ROI_y] + ' | X: '+lb[ROI_x])
#             # fig.suptitle('Y: '+lb[ROI_y] + ' | X: '+lb[ROI_x])

#             im = ax[0, 1].imshow(np.mean(X_LD[:, k2, :, :].copy(), 0), cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2],
#                                  aspect='equal', origin='lower', vmin=-vmax, vmax=vmax)
#             fig.colorbar(im, ax=ax[0, 1])
#             ax[0, 1].set_ylabel('Y: '+lb[ROI_x] + ' | X: '+lb[ROI_y])
#             # ax[0,1].title('Power of '+lb[k])

#             im = ax[1, 0].imshow(np.mean(Z.copy(), 0), cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2],
#                                  aspect='equal', origin='lower', vmin=-vmax, vmax=vmax)
#             fig.colorbar(im, ax=ax[1, 0])
#             ax[1, 0].set_ylabel('k1-k2')
#             # ax[0,1].title('Power of '+lb[k])

#             vmax = np.max(np.nan_to_num(T_obs))
#             vmin = np.min(np.nan_to_num(T_obs))
#             v = max(abs(vmax), abs(vmin))
#             # plt.subplot(313)
#             ax[1, 1].imshow(T_obs, cmap=plt.cm.gray,
#                             extent=[t1, t2, t1, t2],
#                             aspect='equal', origin='lower', vmin=-v, vmax=v)
#             im = ax[1, 1].imshow(T_obs_plot, cmap=plt.cm.seismic,
#                                  extent=[t1, t2, t1, t2],
#                                  aspect='equal', origin='lower', vmin=-v, vmax=v)
#             fig.colorbar(im, ax=ax[1, 1])
#             ax[1, 1].set_ylabel('t-test')
#             # plt.title('Power of '+lb[k])

#             k += 1
# # plt.close('all')

##############################################################################
# lb = C.ROIs_lables
# t1, t2 = [-200, 900]
# # # difference of SD (0:6) and LD(6:12) for aech ROI and individual
# Z = X_SD

# tail = 0
# t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., len(C.subjects) - 1)
# k = 0
# for ROI_y in np.arange(0, 1):
#     for ROI_x in np.arange(0, 6):
#         if ROI_y != ROI_x:
#             T_obs, clusters, cluster_p_values, H0 = \
#                 permutation_cluster_1samp_test(Z[:, k, :, :], n_permutations=C.n_permutations,
#                                                threshold=t_threshold, tail=tail, out_type='mask',
#                                                verbose=True)

#             T_obs_plot = np.nan * np.ones_like(T_obs)
#             for c, p_val in zip(clusters, cluster_p_values):
#                 if p_val <= C.pvalue:
#                     T_obs_plot[c] = T_obs[c]

#             # plt.figure(figsize=(3,10))
#             fig, ax = plt.subplots(2, 1, figsize=(8, 8))
#             # # plotting the t-values
#             vmax = max(total_max)
#             # plt.subplot(311)
#             im = ax[0].imshow(np.mean(X_SD[:, k, :, :].copy(), 0), cmap=plt.cm.seismic,
#                               extent=[t1, t2, t1, t2], aspect='equal', origin='lower', vmin=-vmax, vmax=vmax)
#             fig.colorbar(im, ax=ax[0])
#             ax[0].set_ylabel('SD')
#             fig.suptitle('Y: '+lb[ROI_y] + ' | X: '+lb[ROI_x])

#             vmax = np.max(T_obs)
#             vmin = np.min(T_obs)
#             v = max(abs(vmax), abs(vmin))
#             # plt.subplot(313)
#             ax[1].imshow(T_obs, cmap=plt.cm.gray,
#                          extent=[t1, t2, t1, t2],
#                          aspect='equal', origin='lower', vmin=-v, vmax=v)
#             im = ax[1].imshow(T_obs_plot, cmap=plt.cm.seismic,
#                               extent=[t1, t2, t1, t2],
#                               aspect='equal', origin='lower', vmin=-v, vmax=v)
#             fig.colorbar(im, ax=ax[1])
#             ax[1].set_ylabel('t-test')
#             # plt.title('Power of '+lb[k])

#             k += 1
