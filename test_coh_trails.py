#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 07:33:10 2020

@author: sr05
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from SN_matrix_mirror import matrix_mirror 
from mne.stats import (permutation_cluster_1samp_test,
                       summarize_clusters_stc,permutation_cluster_test)
from scipy import stats as stats
# from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import (spatio_temporal_cluster_test, f_threshold_mway_rm,
                       f_mway_rm, summarize_clusters_stc)
from matplotlib import pyplot as plt
import statsmodels.stats.multicomp as multi
import time
import pickle
import sys
import os
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr_epoch
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    
con_labels_SD=[[0]*4 for w in range(2)]
con_labels_LD=[[0]*4 for w in range(2)]
method='coh'
s=time.time()
t_min = 50
t_max = 300

nb_trails_SD=[5,10,50,100,400]  
nb_trails_LD=[5,10,50,100,200]    
con_SD_trails=np.zeros([18,4,5])
con_LD_trails=np.zeros([18,4,5])

for i in np.arange(0,len(C.subjects)):
    print('participant: ',i)
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    
    morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                  subject_from='fsaverage',subjects_dir=data_path)
        
    
    # Reading epochs
    epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
        
    epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)
    
    epochs_SD = epochs_sd['words'].copy().resample(500)
    epochs_LD = epochs_ld['words'].copy().resample(500)
    
    # Reading inverse operator
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'
    
    inv_op_SD = read_inverse_operator(inv_fname_SD) 
    inv_op_LD = read_inverse_operator(inv_fname_LD) 
                
    stc_sd = apply_inverse_epochs(epochs_SD, inv_op_SD,lambda2,method ='MNE', 
                          pick_ori="normal", return_generator=False)
    stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                            pick_ori="normal", return_generator=False)
    times=epochs_SD.times
    stc_SD_t =[]
    stc_LD_t =[]
    
    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']
 
    # for n in np.arange(0,len(stc_sd)):
    #     stc_SD_t.append(stc_baseline_correction(stc_sd[n],times))
    # for n in np.arange(0,len(stc_ld)):
    #     stc_LD_t.append(stc_baseline_correction(stc_ld[n],times))
   
    stc_SD=[]
    stc_LD=[]
    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']
   
    for k in np.arange(0,6):
        morphed_labels[k].name = C.rois_labels[k]
    if len(stc_sd)>400:
        for t, a in enumerate(nb_trails_SD):
            for n in np.arange(0,a):
                 stc_SD.append(stc_sd[n].copy().crop(t_min*1e-3,t_max*1e-3))
        
            labels_ts_sd = mne.extract_label_time_course(stc_SD, morphed_labels, \
                   src_SD, mode='mean_flip',return_generator=False)
        
            for f in np.arange(0,len(C.con_freq_band)-1):
                f_min=C.con_freq_band[f]
                f_max=C.con_freq_band[f+1]
                con_SD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                     labels_ts_sd, method=method, mode='fourier', 
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=False, n_jobs=10)
                con_SD_trails[i,f,t]= con_SD.copy().mean(2)[1,0]
  
    
    if len(stc_ld)>200:
        for t, a in enumerate(nb_trails_LD):
            for n in np.arange(0,a):
                 stc_LD.append(stc_ld[n].copy().crop(t_min*1e-3,t_max*1e-3))
            labels_ts_ld = mne.extract_label_time_course(stc_LD, morphed_labels, \
                   src_LD, mode='mean_flip',return_generator=False)
           
            for f in np.arange(0,len(C.con_freq_band)-1):
                f_min=C.con_freq_band[f]
                f_max=C.con_freq_band[f+1]   
                con_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                     labels_ts_ld, method=method, mode='fourier', 
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=False, n_jobs=10)
         
                con_LD_trails[i,f,t]= con_LD.copy().mean(2)[1,0]
con_LD_trails= np.delete(con_LD_trails,[1,2,4],0)

e=time.time()
print(e-s)
col=['r','m','b','g']
label=['theta','alpha','beta','gamma']
plt.figure()
for f in np.arange(0,4):
    means_SD= con_SD_trails[:,f,:].copy().mean(0)
    errors_SD= np.std(con_SD_trails[:,f,:],0)
   
    plt.errorbar(nb_trails_SD, means_SD,yerr=errors_SD,label=label[f])
    plt.ylabel('Coherence SD')
    plt.xlabel('Number of trials')
    plt.title('Coherence as a function of trial number')
    plt.legend(loc='upper right')

plt.figure()
for f in np.arange(0,4):
    means_LD= con_LD_trails[:,f,:].copy().mean(0)
    errors_LD= np.std(con_LD_trails[:,f,:],0)
    plt.errorbar(nb_trails_LD, means_LD,yerr=errors_LD,label=label[f])
    plt.ylabel('Coherence LD')
    plt.xlabel('Number of trials')
    plt.title('Coherence as a function of trial number')
    plt.legend(loc='upper right')

for f in np.arange(0,4):

    plt.figure()
    
    means_SD= con_SD_trails[:,f,:].copy().mean(0)
    errors_SD= np.std(con_SD_trails[:,f,:],0)
    means_LD= con_LD_trails[:,f,:].copy().mean(0)
    errors_LD= np.std(con_LD_trails[:,f,:],0)
    plt.errorbar(nb_trails_SD, means_SD,yerr=errors_SD,label=label[f]+' SD',color='b')
    plt.errorbar(nb_trails_LD, means_LD,yerr=errors_LD,label=label[f]+' LD',color='r')
    
    plt.ylabel('Coherence')
    plt.xlabel('Number of trials')
    plt.title('Coherence as a function of trial number')
    plt.legend(loc='upper right')