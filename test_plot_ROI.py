#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:55:20 2020

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
import sn_config as C
from surfer import Brain
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2


mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=C.data_path,verbose=True)
        

labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',\
                                    subjects_dir=C.data_path)
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))
brain.add_annotation('HCPMMP1')
    

label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TE2a_ROI-lh','L_TE1a_ROI-lh','L_TE1m_ROI-lh']


my_ATL=[]
for j in np.arange(0,len(label_ATL )):
    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])

for m in np.arange(0,len(my_ATL)):
    if m==0:
        ATL = my_ATL[m]
    else:
        ATL = ATL + my_ATL[m]
        
 
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))

for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False)
 
brain.add_label(my_ATL, borders=False,color='blue')
