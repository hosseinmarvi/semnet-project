#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:10:38 2020

@author: sr05
"""
import mne
import numpy as np
import SN_config as C
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from scipy import stats as stats
from mne.minimum_norm import apply_inverse, read_inverse_operator
from matplotlib import pyplot as plt
from mne.stats import (permutation_cluster_1samp_test,
                       summarize_clusters_stc,permutation_cluster_test,
                       f_threshold_mway_rm, f_mway_rm)

# import statsmodels.stats.multicomp as multi

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_MRI
epochs_names = C.epochs_names
inv_op_name = C.inv_op_name

# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    
X = np.zeros([len(subjects) ,2*len(SN_ROI ), 1201])



# for i in np.arange(0, len(subjects)):
#     n_subjects = len(subjects)
#     meg = subjects[i]
#     sub_to = MRI_sub[i][1:15]

#     # print('Participant : ' , i, '/ win : ',win)
#     print('Participant : ' , i)

#     morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
#                   subject_from='fsaverage',subjects_dir=data_path)
#     for n in np.array([0]):
#         # Reading epochs
#         epo_name= data_path + meg + epochs_names[n]
#         epochs = mne.read_epochs(epo_name, preload=True)
#         epochs = epochs['words']
    
#         # Reading inverse operator
#         inv_fname = data_path + meg + inv_op_name[n]
#         inv_op = read_inverse_operator(inv_fname) 
    
#         # Evoked responses 
#         evoked = epochs.average().set_eeg_reference(ref_channels = \
#                             'average',projection=True)
    
#         # Applying inverse solution to get sourse signals    
#         stc = apply_inverse(evoked, inv_op,lambda2,method ='MNE', 
#                                pick_ori=None)
        
#         morph = mne.compute_source_morph( src= inv_op['src'],subject_from\
#                    = stc.subject , subject_to = C.subject_to , spacing = \
#                    C.spacing_morph, subjects_dir = C.data_path)    

        
#         stc_fsaverage = morph.apply(stc)
#         stc_corrected = stc_baseline_correction(stc_fsaverage) 
        
#         if i==0:            
#             stc_all_fsaverage  = stc_fsaverage   
#             stc_all_corrected  = stc_corrected               
#         else:
#             stc_all_fsaverage  = stc_all_fsaverage+stc_fsaverage   
#             stc_all_corrected  = stc_all_corrected +stc_corrected       
        

        
# fig, axs = plt.subplots(2)
# fig.suptitle('Average - no correction/baseline correction')
# axs[0].plot(1e3 * stc_all_fsaverage.times, stc_all_fsaverage.data[::100, :].T.copy())
# plt.xlabel('time (ms)')
# plt.ylabel('%s value MNE' )       

# axs[1].plot(1e3 * stc_all_corrected.times, stc_all_corrected.data[::100, :].T.copy())
# plt.xlabel('time (ms)')
# plt.ylabel('%s value MNE' )  


T = np.arange(-300,901)
X=np.zeros([18,6,1201])
Y=np.zeros([18,6,1201])

for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]

    # print('Participant : ' , i, '/ win : ',win)
    print('Participant : ' , i)

    morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                  subject_from='fsaverage',subjects_dir=data_path)
    for n in np.array([0]):
        # Reading epochs
        epo_name= data_path + meg + epochs_names[n]
        epochs = mne.read_epochs(epo_name, preload=True)
        epochs = epochs['words']
    
        # Reading inverse operator
        inv_fname = data_path + meg + inv_op_name[n]
        inv_op = read_inverse_operator(inv_fname) 
    
        # Evoked responses 
        evoked = epochs.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
    
        # Applying inverse solution to get sourse signals    
        stc = apply_inverse(evoked, inv_op,lambda2,method ='MNE', 
                               pick_ori=None)
        # stc_corrected = stc_baseline_correction(stc) 
            # 
        for j in np.arange(0,len(SN_ROI)):
            morphed_labels[j].subject = sub_to   
            
            label_ =  stc.in_label(morphed_labels[j]) 
            X[i,n*len(SN_ROI)+j,:]  = abs(label_.data).copy().mean(0)
          
            # stc_corrected = stc_baseline_correction(label_) 
            Y[i,n*len(SN_ROI)+j,:]  = stc_baseline_correction(abs(label_.data).copy().mean(0),T )
          
t= np.arange(-300,901)
  
plt.figure()
plt.subplot(211)
# plt.title('t-test and cluster-based permutation on '+lb[j])
plt.plot(t, X.copy().mean(axis=0).mean(axis=0),'b')
plt.subplot(212)
plt.plot(t, Y.copy().mean(axis=0).mean(axis=0),'r')


# plt.figure()
# plt.subplot(211)
# plt.plot(T, X.copy()[0,0,:],'b')
# plt.subplot(212)
# plt.plot(T, Y.copy()[0,0,:],'r')

# 4.3798773031833986e-13
# 1.253888098327307e-12