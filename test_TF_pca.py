#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================
Compute power and phase lock value in 6 ROIs 
=========================================================

Compute time-frequency maps of power and phase lock in the
source space.
The inverse method is linear based on MNE inverse operator.

There are 4 sections needed to be uncomment based on the 
purpose:1) 2-way RM to compare SD/LD and interaction by ROIs.
2) t-test on each ROI to compare SD/LD. 3) 2-way RM to compare
SD/LD and lATL/rATL and the their interaction. 4) t-test on any 
combination of: SD_lATL, SD_rATL, LD_lATL, LD_rATL

@author: sr05
"""
import time
import mne
import pickle
import os
import numpy as np
import SN_config as C
from scipy import stats as stats
from matplotlib import pyplot as plt
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse, read_inverse_operator,\
                             source_induced_power
from mne.stats import permutation_cluster_1samp_test,f_threshold_mway_rm,\
                      summarize_clusters_stc,permutation_cluster_test,\
                      f_mway_rm
# from sklearn.decomposition import PCA
# pca = PCA(n_components=1)

start=time.time()
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_MRI_files
# Parameters
snr = C.snr
lambda2 = C.lambda2
SN_ROI = SN_semantic_ROIs()  
freq = np.arange(6, 40, 2)  # define frequencies of interest
n_cycles = freq / 3
epochs_names = C.epochs_names
inv_op_name = C.inv_op_name
X = np.zeros([len(subjects) ,2*len(SN_ROI), len(freq),600])
# Y = np.zeros([len(subjects) ,2*len(SN_ROI), len(freq),600])
n_subjects = len(subjects)
# s=time.time()
# for i in np.arange(0, len(subjects)):
    
#     meg = subjects[i]
#     sub_to = MRI_sub[i]
#     print('Participant : ' , i)

#     # morphing ROIs from fsaverage to each individual
#     morphed_labels = mne.morph_labels(SN_ROI, subject_to=data_path+sub_to,\
#                     subject_from='fsaverage',subjects_dir=data_path)
#     # Reading epochs for SD(n=0)/LD(n=1) 
#     for n in np.array([0,1]):
#         epo_name= data_path + meg + epochs_names[n]
#         epochs = mne.read_epochs(epo_name, preload=True)
#         epochs = epochs['words'].resample(500)
    
#         # Reading inverse operator
#         inv_fname = data_path + meg + inv_op_name[n]
#         inv_op = read_inverse_operator(inv_fname) 
       
#         # Computing the power and phase lock value for each ROI
#         for j in np.arange(0,len(morphed_labels)):
#             print('Participant: ' , i,'/ condition: ',n,'/ ROI: ',j)
#             power, itc = source_induced_power(epochs,inverse_operator=\
#                       inv_op, freqs=freq, label=morphed_labels[j], lambda2=\
#                       C.lambda2, method='MNE', baseline=(-.300,0), baseline_mode=\
#                       'percent', n_jobs=10, n_cycles=n_cycles,zero_mean=True)
            
                            
#             src_pow = np.sum(power.copy().reshape(power.shape[0],len(freq)*600), axis=1)
#             vertex_ind = np.argmax(src_pow)
           
#             X[i,n*len(morphed_labels)+j,:,:]= power[vertex_ind,:,:]

#             # # PCA
#             # a=pca.fit_transform(np.transpose(power.copy().reshape(power.shape[0],len(freq)*600),[1,0]))
#             # b=pca.fit_transform(np.transpose(itc.copy().reshape(itc.shape[0],len(freq)*600),[1,0]))
#             # X[i,n*len(morphed_labels)+j,:,:]= a.reshape(len(freq),600) 
#             # # Phase lock value
#             # Y[i,n*len(morphed_labels)+j,:,:]= b.reshape(len(freq),600) 
# X_file_name=os.path.expanduser('~') +'/my_semnet/source_induced_power_maxpow.json'
# # Y_file_name=os.path.expanduser('~') +'/my_semnet/source_induced_plv_pca.json'

# with open(X_file_name, "wb") as fp:   #Pickling
#     pickle.dump(X, fp)
    
# with open(Y_file_name, "wb") as fp:   #Pickling
#     pickle.dump(Y, fp)
    

# e=time.time()
# print(e-s)

X_file_name=os.path.expanduser('~') +'/my_semnet/source_induced_power_maxpow.json'
# Y_file_name=os.path.expanduser('~') +'/my_semnet/source_induced_plv_pca.json'



with open(X_file_name, "rb") as fp:   # Unpickling
    X = pickle.load(fp)

# with open(Y_file_name, "rb") as fp:   # Unpickling
#     Y = pickle.load(fp)
T=np.arange(-300,900,2)*1e-3
t_threshold = -stats.distributions.t.ppf(C.pvalue/ 2., n_subjects - 1)
a=175
b=150
times=T[a:-b]
XX=np.power(10,X.copy()/10)
##############################################################################
# ## plot each ROI across subjects: columns 0-5->SD, 6:11->LD 
# ## sequence of ROIs: 0:lATL, 1:rATL, 2:TG, 3:IFG, 4:AG, 5:PVA
# for m in np.arange(0,6):  
#     vmax=max(abs(X[:,m+6,:,a:-b].max()),\
#               abs(X[:,m,:,a:-b].max()))
#     vmin=max(abs(X[:,m+6,:,a:-b].min()),\
#               abs(X[:,m,:,a:-b].min()))
#     plt.figure()
#     plt.subplot(3,1,1)
#     plt.imshow( X[:,m,:,a:-b].copy().mean(0),
#                     extent=[times[1], times[-1], freq[0], freq[-1]],
#                     aspect='auto', origin='lower', cmap='RdBu_r') 
#     plt.title('Power of '+C.ROIs_lables[m])
#     plt.colorbar()
#     plt.ylabel('SD (HZ)')
#     plt.subplot(3,1,2)
#     plt.imshow( X[:,m+6,:,a:-b].copy().mean(0),
#                     extent=[times[1], times[-1], freq[0], freq[-1]],
#                     aspect='auto', origin='lower', cmap='RdBu_r') 
#     plt.colorbar()
#     plt.ylabel('LD (HZ)')
    
#     plt.subplot(3,1,3)
#     plt.imshow( (X[:,m,:,a:-b].copy().mean(0)-X[:,m+6,:,a:-b].copy().mean(0)),
#                     extent=[times[1], times[-1], freq[0], freq[-1]],
#                     aspect='auto', origin='lower', cmap='RdBu_r') 
#     plt.colorbar()
#     plt.ylabel('SD-LD (HZ)')
    # plt.savefig(C.pictures_path_Source_estimate+ 'TF_pca'+C.ROIs_lables[m]+'.png')
# plt.close('all')


##############################################################################
### 2-way repeated measure : F-test and cluster-based correction 
## Use to compare the SD/LD in all ROIs
tail=0
# factor_levels=[2,6]
# effects=['A','A:B']
# y_label=['SD-LD','SD-LD by ROIs']
# # effects=['A']
# # y_label=['SD-LD']
# A_all=np.arange(0,12)

# S=[]
# ## A for all ROIs and A_ATL for ATLs
# for j in A_all:
#     # S.append(X[:,j,:,175:-249])
#         S.append(X[:,j,:,a:-b]) 

    
# for e , effect in enumerate(effects):
   
#     # computing f threshold
#     f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects=effect,\
#                                     pvalue=C.pvalue )
#     p=0
#     def stat_fun(*args):
#         return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
#                           effects=effect, return_pvals=False)[0] 
   
#     T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
#         S, stat_fun=stat_fun, threshold=f_thresh, tail=tail,\
#         n_jobs=6, n_permutations=C.n_permutations, buffer_size=None,\
#         out_type='mask')
      
#     T_obs_plot = np.nan * np.ones_like(T_obs)
#     for c, p_val in zip(clusters, cluster_p_values):
#         if p_val <= 0.1:
#             T_obs_plot[c] = T_obs[c]
            
#     T_obs_ttest = np.nan * np.ones_like(T_obs)
#     for r in np.arange(0,X.shape[2]):
#         for c in np.arange(0,times.shape[0]):
#             if abs(T_obs[r,c])>f_thresh:
#                 T_obs_ttest[r,c] =  T_obs[r,c]
            
#     vmax = np.max(np.abs(T_obs))
#     vmin = 0
#     plt.figure()
#     # plotting f-values
#     plt.subplot(311)
#     plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.ylabel('Frequency (Hz)')
#     plt.title('Power ('+y_label[e]+')')

#     # Plotting the uncorrected f-test
#     plt.subplot(312)
#     plt.imshow(T_obs, cmap=plt.cm.bone,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
#     plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.ylabel('Frequency (Hz)')
    
#     # Plotting the corrected f-test
#     plt.subplot(313)
#     plt.imshow(T_obs, cmap=plt.cm.gray,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
#     plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Frequency (Hz)')

#     plt.show()      
#     plt.savefig(C.pictures_path_Source_estimate+ 'two-way_RM_pca'+y_label[e]+'.png')

##############################################################################
### t-test and cluster-based correction for each ROI
tail=0
lb=C.ROIs_lables
# difference of SD (0:6) and LD(6:12) for aech ROI and individual
Z= X[:,0:6,:,a:-b]-X[:,6:12,:,a:-b]

for k in np.arange(0,len(lb)):
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_1samp_test(Z[:,k,:,:], n_permutations=C.n_permutations,
                                        threshold=t_threshold, tail=tail,
                                        connectivity=None,out_type='mask',
                                        verbose=True)
        
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= C.pvalue:
            T_obs_plot[c] = T_obs[c]
            
    T_obs_ttest = np.nan * np.ones_like(T_obs)
    for r in np.arange(0,Z.shape[2]):
        for c in np.arange(0,Z.shape[3]):
            if abs(T_obs[r,c])>t_threshold:
                T_obs_ttest[r,c] =  T_obs[r,c]
            
    vmax = np.max(T_obs)
    vmin = np.min(T_obs)
    v=max(abs(vmax),abs(vmin))
    plt.figure()
    # plotting the t-values    
    plt.subplot(311)
    plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
                extent=[times[0], times[-1], freq[0], freq[-1]],
                aspect='auto', origin='lower', vmin=-v, vmax=v)
    plt.colorbar()
    plt.ylabel('Frequency (Hz)')
    plt.title('Power of '+lb[k])
    
    # plotting the uncorreted t-test
    plt.subplot(312)
    plt.imshow(T_obs, cmap=plt.cm.bone,
                extent=[times[0], times[-1], freq[0], freq[-1]],
                aspect='auto', origin='lower', vmin=-v, vmax=v)
    
    plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
                extent=[times[0], times[-1], freq[0], freq[-1]],
                aspect='auto', origin='lower', vmin=-v, vmax=v)
    plt.colorbar()
    plt.ylabel('Frequency (Hz)')

    # plotting the correted t-test
    plt.subplot(313)
    plt.imshow(T_obs, cmap=plt.cm.gray,label='cluster-based permutation test',
                extent=[times[0], times[-1], freq[0], freq[-1]],
                aspect='auto', origin='lower', vmin=-v, vmax=v)
    
    plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,label='cluster-based permutation test',
                extent=[times[0], times[-1], freq[0], freq[-1]],
                aspect='auto', origin='lower', vmin=-v, vmax=v)
    plt.colorbar()
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')

    plt.show()      
    # plt.savefig(C.pictures_path_Source_estimate+ 't-test_TFR_pca_'+lb[k]+'.png')

##############################################################################
### 2-way repeated measure : F-test and cluster-based correction 
## Use to compare the SD/LD in ATLs
# factor_levels=[2,2]
# effects=['A','B','A:B']
# y_label=['SD-LD','ATLs','SD-LD by ATLs']
# tail=0
# ## indices of ATLs in SD(0,1) and LD(6,7)
# A_ATL=np.array([0,1,6,7])
# S=[]
# for j in A_ATL:
#     S.append(X[:,j,:,a:-b]) 
      
# for e , effect in enumerate(effects):
   
#     # computing f threshold
#     f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects=effect,\
#                                     pvalue= C.pvalue )
#     p=0
#     def stat_fun(*args):
#         return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
#                           effects=effect, return_pvals=False)[0] 
   
#     T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
#         S, stat_fun=stat_fun, threshold=f_thresh, tail=tail,\
#         n_jobs=4, n_permutations=C.n_permutations, buffer_size=None,\
#         out_type='mask')
      
#     T_obs_plot = np.nan * np.ones_like(T_obs)
#     for c, p_val in zip(clusters, cluster_p_values):
#         if p_val <= C.pvalue:
#             T_obs_plot[c] = T_obs[c]
            
#     T_obs_ttest = np.nan * np.ones_like(T_obs)
#     for r in np.arange(0,X.shape[2]):
#         for c in np.arange(0,times.shape[0]):
#             if abs(T_obs[r,c])>f_thresh:
#                 T_obs_ttest[r,c] =  T_obs[r,c]
            
#     vmax = np.max(np.abs(T_obs))
#     vmin = 0
#     plt.figure()
#     # plotting f-values
#     plt.subplot(311)
#     plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.ylabel('Frequency (Hz)')
#     plt.title('Power ('+y_label[e]+')')

#     # Plotting the uncorrected f-test
#     plt.subplot(312)
#     plt.imshow(T_obs, cmap=plt.cm.bone,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
#     plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.ylabel('Frequency (Hz)')
    
#     # Plotting the corrected f-test
#     plt.subplot(313)
#     plt.imshow(T_obs, cmap=plt.cm.gray,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
#     plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
#                 extent=[times[0], times[-1], freq[0], freq[-1]],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Frequency (Hz)')

#     plt.show()      
#     plt.savefig(C.pictures_path_Source_estimate+ 'two-way_RM_pca_ATLs'+y_label[e]+'.png')
