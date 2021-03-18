#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:51:00 2020

@author: sr05
"""


import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import SN_config as C
from surfer import Brain
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2

def sort_matrix(X,a):    
    X_reshaped = np.reshape(X,(1,len(X)*len(X)))
    X_threshold = np.sort(np.abs(X_reshaped))[0,a-1]

    for i in np.arange(0,len(X)):
        for j in np.arange(0,len(X)):
            if np.abs(X[i,j])<X_threshold :
                X[i,j]=0
            
    return X       

       
C.ImCoh_SD_LD[win,freq,:,:]
C.ImCoh_SD_sorted[win,freq,:,:]  
C.ImCoh_LD_sorted[win,freq,:,:]   

ImCoh_SD_sort_0 = sort_matrix(C.ImCoh_SD_sorted[0,0,:,:] ,50)
ImCoh_SD_sort_1 = sort_matrix(C.ImCoh_SD_sorted[1,0,:,:] ,50)

ImCoh_LD_sort_0 = sort_matrix(C.ImCoh_LD_sorted[0,0,:,:] ,50)
ImCoh_LD_sort_1 = sort_matrix(C.ImCoh_LD_sorted[1,0,:,:] ,50)