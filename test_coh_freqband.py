
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:42:09 2020

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

start=time.time()
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = C.label_path
method='coh'

stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/stc_'+method+'_bands_SD.json'
stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/stc_'+method+'_bands_LD.json'



with open(stc_SD_file_name, "rb") as fp:   # Unpickling
    my_stc_coh_SD = pickle.load(fp)

with open(stc_LD_file_name, "rb") as fp:   # Unpickling
    my_stc_coh_LD = pickle.load(fp)


stc_kmax_SD=[]
stc_kmax_LD=[]

stc_kmin_SD=[]
stc_kmin_LD=[]
w_label=[' 50-300ms',' 300-550ms']
f_label=['theta','alpha','beta','gamma']
#########################################################
## Plot average coh maps for two windows, four frequency
##bands, and two conditions
# for w in np.arange(1,2):
#     vmax=[]
#     vmin=[]
#     for k in np.arange(0,6):
#         stc_max_SD=[]
#         stc_max_LD=[]
#         stc_min_SD=[]
#         stc_min_LD=[]
#         for f in np.arange(0,4): 
#             stc_t_SD=0
#             stc_t_LD=0
#             for i in np.arange(0,18):                
#                 if i==0:
#                     stc_t_SD=my_stc_coh_SD[i][k][f][w]
#                     stc_t_LD=my_stc_coh_LD[i][k][f][w]
    
#                 else:
#                     stc_t_SD=stc_t_SD+my_stc_coh_SD[i][k][f][w]
#                     stc_t_LD=stc_t_LD+my_stc_coh_LD[i][k][f][w]
    
#             stc_max_SD.append((stc_t_SD/len(subjects)).data.max())
#             stc_max_LD.append((stc_t_LD/len(subjects)).data.max())
#             stc_min_SD.append((stc_t_SD/len(subjects)).data.min())
#             stc_min_LD.append((stc_t_LD/len(subjects)).data.min())
#         vmax.append(max(max(stc_max_SD),max(stc_max_LD)))
#         vmin.append(min(min(stc_min_SD),min(stc_min_LD)))

#     for k in np.arange(5,6): 
#         for f in np.arange(0,4): 
#             for i in np.arange(0,18):
#                 if i==0:
#                     stc_t_SD=my_stc_coh_SD[i][k][f][w]
#                     stc_t_LD=my_stc_coh_LD[i][k][f][w]
    
#                 else:
#                     stc_t_SD=stc_t_SD+my_stc_coh_SD[i][k][f][w]
#                     stc_t_LD=stc_t_LD+my_stc_coh_LD[i][k][f][w]
    
#             stc_T_SD=stc_t_SD/len(subjects)
#             stc_T_LD=stc_t_LD/len(subjects)
           
#             v_max=vmax[k]
#             v_min=vmin[k]
#             # vmid=(v_max+v_min)/4

#             v_max=max(abs(v_max),abs(v_min))
#             v_min=v_max/10
#             vmid=v_max/5
            
#             print('Coherence_SD : '+C.ROIs_lables[k]+'-'+w_label[w])
#             brain = stc_T_SD.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='value', pos_lims=(v_min,vmid,v_max)))
#             brain.save_image(C.pictures_path_Source_estimate+method+'_SD-'+C.ROIs_lables[k]+'_'+w_label[w][:-3]+'_'+f_label[f]+'.png')
            
#             print('Coherence_LD : '+C.ROIs_lables[k]+'-'+w_label[w]+'-'+f_label[f])
    
#             brain = stc_T_LD.plot(surface='inflated', hemi='split',
#                       time_label=method+'_LD : '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='value', pos_lims=(v_min,vmid,v_max)))
#             brain.save_image(C.pictures_path_Source_estimate+method+'_LD-'+C.ROIs_lables[k]+'_'+w_label[w][:-3]+'_'+f_label[f]+'.png')

#########################################################
## cluster-based permutation: SD vs LD in each band
t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., len(C.subjects) - 1)
not_sig=[]
for w in np.arange(1,2):
    # for k in np.arange(0,6):
     for k in np.array([0,4]):

        for f in np.arange(3,4):
            print('Clustering: ',f_label[f],'/ k:',k, '/ w: ',w)
            X_SD=np.zeros([18,1,20484])
            X_LD=np.zeros([18,1,20484])
            for i in np.arange(0,18):
                X_SD[i,:,:]=np.transpose(my_stc_coh_SD[i][k][f][w].data, [1, 0])
                X_LD[i,:,:]=np.transpose(my_stc_coh_LD[i][k][f][w].data, [1, 0])
            
            Y=X_SD-X_LD
            source_space = mne.grade_to_tris(5)
            connectivity = mne.spatial_tris_connectivity(source_space)
            tstep = my_stc_coh_LD[i][k][f][w].tstep
               
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
                    'fsaverage', subjects_dir=data_path, clim=dict(kind='value', pos_lims=\
                    [thresh/10,thresh/5,thresh]), size=(800,400), colormap='mne', time_label=\
                      'SD-LD: ' + C.rois_labels[k] + '_' + w_label[w] + '_' + f_label[f])
                
                brain.save_image(C.pictures_path_Source_estimate +'t-test_' + method + \
                                 C.rois_labels[k] + '_' + w_label[w] + '_' + f_label[f] + '.png')
            else:
                not_sig.append([w,k,f])
           





#########################################################
# tail=0                     
# n_subjects = len(subjects)  
# factor_levels=[4,2]
# # effects=['A','B']
# # y_label=['frequency bands','SD-LD']
# effects=['A:B']
# y_label=['frequency bands']
# A_all=np.arange(0,8)
# source_space = mne.grade_to_tris(5)
# connectivity = mne.spatial_tris_connectivity(source_space)
# t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., n_subjects - 1)
# w_label=[' (50-300ms)',' (300-550ms)']
#######################################################
# for w in np.arange(1,2):    
#     for k in np.arange(5,6): 
#         S=[]
#         for f in np.arange(0,4):
#             X_SD=np.zeros([18,1,20484])
#             X_LD=np.zeros([18,1,20484])
#             for i in np.arange(0,18):
#                 X_SD[i,:,:]=np.transpose(my_stc_coh_SD[i][k][f][w].data, [1, 0])
#                 X_LD[i,:,:]=np.transpose(my_stc_coh_LD[i][k][f][w].data, [1, 0])
            
#             S.append(X_SD)
#             S.append(X_LD)
               
#     for e , effect in enumerate(effects):
       
#         # computing f threshold
#         f_thresh = f_threshold_mway_rm(n_subjects, factor_levels,\
#                     effects=effect,pvalue=C.pvalue )
#         p=0
#         def stat_fun(*args):
#             return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
#                               effects=effect, return_pvals=False)[0] 
#         tstep = my_stc_coh_SD[i][k][f][w].tstep
#         n_vertices_sample, n_times = my_stc_coh_SD[i][k][f][w].data.shape

#         fsave_vertices = [np.arange(10242), np.arange(10242)]

       
#         T_obs, clusters, cluster_p_values, h0 = clu = mne.stats.spatio_temporal_cluster_test(
#             S, stat_fun=stat_fun, threshold=f_thresh, tail=tail,\
#             n_jobs=10, n_permutations=C.n_permutations, buffer_size=None,\
#             step_down_p=0.05,\
#             connectivity=connectivity,t_power=1)
#         print(effects)
#         stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu,tstep=tstep,p_thresh=0.05,\
#                               subject='fsaverage',vertices=fsave_vertices)
         
#         idx = stc_all_cluster_vis.time_as_index(times=stc_all_cluster_vis.times)
#         data = stc_all_cluster_vis.data[:, idx]
#         thresh = max(([abs(data.min()) , abs(data.max())]))
        
#         brain = stc_all_cluster_vis.plot(surface='inflated', hemi='split',subject =\
#             'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', pos_lims=\
#               [0,thresh/5,thresh]),size=(800,400),colormap='mne')
       
        # brain.save_image(C.pictures_path_Source_estimate+'f-test_Coh_SD-LD_'+C.ROIs_lables[k]+'.png')
       
        
    
#######################################################


# w=0
# for k in np.arange(0,1):
#     X_SD=np.zeros([18,1,20484])
#     X_LD=np.zeros([18,1,20484])
    
#     for i in np.arange(0,18):
#         wf0_sd=np.transpose(my_stc_coh_SD[i][k][0][w].data, [1, 0])            
#         wf1_sd=np.transpose(my_stc_coh_SD[i][k][1][w].data, [1, 0])            
#         wf2_sd=np.transpose(my_stc_coh_SD[i][k][2][w].data, [1, 0])            
#         wf3_sd=np.transpose(my_stc_coh_SD[i][k][3][w].data, [1, 0])            

#         wf0_ld=np.transpose(my_stc_coh_LD[i][k][0][w].data, [1, 0])            
#         wf1_ld=np.transpose(my_stc_coh_LD[i][k][1][w].data, [1, 0])            
#         wf2_ld=np.transpose(my_stc_coh_LD[i][k][2][w].data, [1, 0])            
#         wf3_ld=np.transpose(my_stc_coh_LD[i][k][3][w].data, [1, 0])            

        
#         X_SD[i,:,:]=(wf0_sd+ wf1_sd +wf2_sd+ wf3_sd)/4
#         X_LD[i,:,:]=(wf0_ld+ wf1_ld +wf2_ld+ wf3_ld)/4
      

    
#     S=X_SD-X_LD

  
#     tstep = my_stc_coh_SD[i][k][1][0].tstep
#     n_vertices_sample, n_times = my_stc_coh_SD[i][k][1][0].data.shape
#     fsave_vertices = [np.arange(10242), np.arange(10242)]
#     # print('clustering win: ',w)
#     T_obs, clusters, cluster_p_values, h0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
#         S,threshold=t_threshold, tail=tail,\
#         n_jobs=6, n_permutations=C.n_permutations, buffer_size=None,\
#         step_down_p=0.05,connectivity=connectivity,t_power=1)

#     fsave_vertices = [np.arange(n_vertices_sample/2),
#                   np.arange(n_vertices_sample/2)]
    

#     stc_cluster_vis = mne.stats.summarize_clusters_stc(clu,tstep=tstep,p_thresh=0.05,\
#                           subject='fsaverage',vertices=fsave_vertices)
 
#     # idx = stc_cluster_vis.time_as_index(times=stc_cluster_vis.times)
#     # data = stc_cluster_vis.data[:, idx]
#     # thresh = max(([abs(data.min()) , abs(data.max())]))

#     brain = stc_cluster_vis.plot(surface='inflated', hemi='split',subject =\
#         'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', pos_lims=\
#           [0.0002,0.001,.002]),size=(800,400),colormap='mne')

#     brain = stc_cluster_vis.plot(surface='inflated', hemi='split',subject =\
#         'fsaverage',  subjects_dir=data_path,size=(800,400),colormap='mne')
      
#     # brain.save_image(C.pictures_path_Source_estimate+'Connectivity_Coh_'+\
#     #                   C.ROIs_lables[k]+'.png')
           
    
    
    