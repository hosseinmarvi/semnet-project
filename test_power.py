#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:43:25 2020

@author: sr05
"""


import mne
import numpy as np
import SN_config as C
from scipy import stats as stats
from matplotlib import pyplot as plt
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse, read_inverse_operator,\
     source_band_induced_power,source_induced_power
from mne.stats import permutation_cluster_1samp_test,f_threshold_mway_rm,\
                      summarize_clusters_stc,permutation_cluster_test,\
                      f_mway_rm


# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_MRI
# Parameters
snr = C.snr
lambda2 = C.lambda2
SN_ROI = SN_semantic_ROIs()  
freq = np.arange(12, 30, 2)  # define frequencies of interest
n_cycles = freq / 2
epochs_names = C.epochs_names
inv_op_name = C.inv_op_name
X = np.zeros([9 ,len(freq),426])
Y = np.zeros([9 ,len(freq),426])

factor_levels = [2,6]
effects=['A','A:B']
for i in np.arange(0, len(subjects)-17):
    n_subjects = len(subjects)
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    print('Participant : ' , i)

    for n in np.array([0]):
        # Reading epochs
        epo_name= data_path + meg + epochs_names[n]
        epochs = mne.read_epochs(epo_name, preload=True)
        epochs = epochs['words'].crop(-0.300,0.550).resample(500)
    
        # Reading inverse operator
        inv_fname = data_path + meg + inv_op_name[n]
        inv_op = read_inverse_operator(inv_fname) 
       
        print('Participant: ' , i,'/ condition: ',n)
        power, itc = source_induced_power(epochs,inverse_operator=\
                 inv_op, freqs=freq, label=None, lambda2=\
                 C.lambda2, method='MNE', baseline=(-.300,0), baseline_mode=\
                 'percent', n_jobs=6, n_cycles=n_cycles)
        X[i,:,:]  = power.copy().mean(1)
        Y[i,:,:]  = itc.copy().mean(1)

        inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    
        inv_op_SD = read_inverse_operator(inv_fname_SD) 

        morph_SD = mne.compute_source_morph( src= inv_op_SD['src'],subject_from\
                    = sub_to , subject_to = 'fsaverage' , spacing = \
                    C.spacing_morph, subjects_dir = C.data_path)    
      
        
vertices_to = [np.arange(4098), np.arange(4098)]

# Visualizing
pow_stc2 = mne.SourceEstimate(power.copy().mean(1), vertices=vertices_to,
                              subject='fsaverage',tmin=-.300,tstep=0.002)

brain = pow_stc2.plot(surface='inflated', hemi='split',subject =\
          'fsaverage',  subjects_dir=data_path,size=(800,400))