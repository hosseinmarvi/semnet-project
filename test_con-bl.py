#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 18:36:08 2020

@author: sr05
"""
import os
import pickle
import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse_epochs, apply_inverse, read_inverse_operator
from mne.connectivity import spectral_connectivity,seed_target_indices, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from mne.stats import (permutation_cluster_1samp_test,spatio_temporal_cluster_test,
                       summarize_clusters_stc,permutation_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from scipy import stats as stats
from mne.epochs import equalize_epoch_counts
import time

from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

method='coh'
# my_stc_coh_SD=[[[[0]*4 for k in range(6)] for w in range(2)] for i in range(18)]
# my_stc_coh_LD=[[[[0]*4 for k in range(6)] for w in range(2)] for i in range(18)]
my_stc_coh_SD=[[]for i in range(18)]
my_stc_coh_LD=[[]for i in range(18)]
bl_coh_SD=[[]for i in range(18)]
bl_coh_LD=[[]for i in range(18)]
for i in np.arange(0,len(C.subjects)):
    
    stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'200_bands_SD_sub'+str(i)+'.json'
    stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'200_bands_LD_sub'+str(i)+'.json'
    
    
    with open(stc_SD_file_name, "rb") as fp:   # Unpickling
        my_stc_coh_SD[i] = pickle.load(fp)
    
    with open(stc_LD_file_name, "rb") as fp:   # Unpickling
        my_stc_coh_LD[i] = pickle.load(fp)
       
       
    stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'_bl_bands_SD_sub'+str(i)+'.json'
    stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'_bl_bands_LD_sub'+str(i)+'.json'

    with open(stc_SD_file_name, "rb") as fp:   # Unpickling
        bl_coh_SD[i] = pickle.load(fp)
    
    with open(stc_LD_file_name, "rb") as fp:   # Unpickling
        bl_coh_LD[i] = pickle.load(fp)
#########################################################

stc_kmax_SD=[]
stc_kmax_LD=[]

stc_kmin_SD=[]
stc_kmin_LD=[]
w_label=[' 50-250ms',' 250-450ms']
f_label=['theta','alpha','beta','gamma']
 


# for w in np.arange(0,1):
#     for k in np.arange(0,1): 
#         for f in np.arange(1,2): 
#             for i in np.arange(0,18):
#                 if i==0:
#                     stc_t_SD=my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=my_stc_coh_LD[i][w][k][f]
#                     bl_SD= bl_coh_SD[i][0][k][f]
#                     bl_LD= bl_coh_LD[i][0][k][f]

#                 else:
#                     stc_t_SD=stc_t_SD+my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=stc_t_LD+my_stc_coh_LD[i][w][k][f]
#                     bl_SD= bl_SD+ bl_coh_SD[i][0][k][f]
#                     bl_LD= bl_LD+ bl_coh_LD[i][0][k][f]
#             stc_T_SD=stc_t_SD/len(C.subjects)
#             stc_T_LD=stc_t_LD/len(C.subjects)
#             bl_SD=bl_SD/len(C.subjects)
#             bl_LD=bl_LD/len(C.subjects)
            
#             # print('Coherence_SD : '+C.ROIs_lables[k]+'-'+w_label[w])
#             brain = (stc_T_SD-bl_SD).plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='percent', lims=(70,85,100)))
            
#             brain = bl_SD.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='percent', lims=(70,85,100)))
#             brain = (stc_T_LD-bl_LD).plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='percent', lims=(70,85,100)))
            # # brain.save_image(C.pictures_path_Source_estimate+method+'_SD-'+C.ROIs_lables[k]+'_'+w_label[w][:-3]+'_'+f_label[f]+'.png')
            
            # print('Coherence_LD : '+C.ROIs_lables[k]+'-'+w_label[w]+'-'+f_label[f])
    
            # brain = stc_T_LD.plot(surface='inflated', hemi='split',
            #           time_label=method+'_LD : '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
            #           subjects_dir=C.data_path,size=([800,400]))
            # brain.save_image(C.pictures_path_Source_estimate+method+'_LD-'+C.ROIs_lables[k]+'_'+w_label[w][:-3]+'_'+f_label[f]+'.png')
  
#########################################################
# ## cluster-based permutation: SD vs LD in each band
t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., len(C.subjects) - 1)
not_sig=[]
for w in np.arange(1,2):
    for k in np.arange(1,2):
      # for k in np.array([0,4]):

        for f in np.arange(0,1):
            print('Clustering: ',f_label[f],'/ k:',k, '/ w: ',w)
            X_SD=np.zeros([18,1,20484])
            X_LD=np.zeros([18,1,20484])
            for i in np.arange(0,18):
                X_SD[i,:,:]=np.transpose((( my_stc_coh_SD[i][w][k][f]- bl_coh_SD[i][0][k][f]).data), [1, 0])
                X_LD[i,:,:]=np.transpose((( my_stc_coh_LD[i][w][k][f] -bl_coh_LD[i][0][k][f]).data), [1, 0])
            
            Y=X_SD-X_LD
            source_space = mne.grade_to_tris(5)
            connectivity = mne.spatial_tris_connectivity(source_space)
            tstep = my_stc_coh_LD[i][w][k][f].tstep
               
            #     print('Clustering.')
            T_obs, clusters, cluster_p_values, H0 = clu = \
                  spatio_temporal_cluster_1samp_test(Y, connectivity= connectivity,\
                  n_jobs=10,threshold=t_threshold,n_permutations=5000,step_down_p=0.05,t_power=1)
            
            if len(np.where(cluster_p_values<0.05)[0])!=0:           
                print('significant!')
                fsave_vertices = [np.arange(10242),np.arange(10242)]            
                stc_all_cluster_vis = summarize_clusters_stc(clu,tstep=tstep*1000,\
                                      vertices = fsave_vertices)
                 
                idx = stc_all_cluster_vis.time_as_index(times=stc_all_cluster_vis.times)
                data = stc_all_cluster_vis.data[:, idx]
                thresh = max(([abs(data.min()) , abs(data.max())]))
                
                brain = stc_all_cluster_vis.plot(surface='inflated', hemi='split', subject =\
                    'fsaverage', subjects_dir=C.data_path, clim=dict(kind='value', pos_lims=\
                    [thresh/10,thresh/5,thresh]), size=(800,400), colormap='mne', time_label=\
                      'SD-LD: ' + C.rois_labels[k] + '_' + w_label[w] + '_' + f_label[f], views='lateral')
                
                # brain.save_image(C.pictures_path_Source_estimate+'t-test_'+method+'_abs_'+\
                #                   C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f]+'.png')
            else:
                not_sig.append([w,k,f])
           

        