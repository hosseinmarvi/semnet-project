#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:59:30 2020

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
import SN_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from mne.stats import (permutation_cluster_1samp_test,spatio_temporal_cluster_test,
                       summarize_clusters_stc,permutation_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from scipy import stats as stats
from mne.epochs import equalize_epoch_counts
import time
import sys

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_MRI
# Parameters
snr = C.snr_epoch
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    
n_subjects = len(subjects)

method='coh'
f_min=C.con_freq_band[1]
f_max=C.con_freq_band[2]
t_min = C.con_time_window[0]
t_max = C.con_time_window[1]
i=0


s=time.time()
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
###
    
epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'

epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)

epochs_f= mne.epochs.combine_event_ids(epochs_fruit,['visual',
                     'hear','hand','neutral','emotional'], {'words':15})
epochs_o= mne.epochs.combine_event_ids(epochs_odour,['visual',
                     'hear','hand','neutral','emotional'], {'words':15})    
epochs_m= mne.epochs.combine_event_ids(epochs_milk,['visual',
                     'hear','hand','neutral','emotional'], {'words':15})

epochs_f=epochs_f['words']
epochs_o=epochs_o['words']
epochs_m=epochs_m['words']


# equalize_epoch_counts(epochs_SD,epochs_LD)
# Reading inverse operator
inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

inv_op_SD = read_inverse_operator(inv_fname_SD) 
inv_op_LD = read_inverse_operator(inv_fname_LD) 
            
stc_sd = apply_inverse_epochs(epochs_SD, inv_op_SD,lambda2,method ='MNE', 
                      pick_ori="normal", return_generator=False)
stc_f = apply_inverse_epochs(epochs_f, inv_op_SD,lambda2,method ='MNE', 
                      pick_ori="normal", return_generator=False)
stc_o = apply_inverse_epochs(epochs_o, inv_op_SD,lambda2,method ='MNE', 
                      pick_ori="normal", return_generator=False)
stc_m = apply_inverse_epochs(epochs_m, inv_op_SD,lambda2,method ='MNE', 
                      pick_ori="normal", return_generator=False)






# stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
#                         pick_ori="normal", return_generator=False)
src_SD = inv_op_SD['src']
src_LD = inv_op_LD['src']
# Construct indices to estimate connectivity between the label time course
# and all source space time courses
vertices_SD = [src_SD[j]['vertno'] for j in range(2)]
n_signals_tot = 1 + len(vertices_SD[0]) + len(vertices_SD[1])
indices = seed_target_indices([0], np.arange(1, n_signals_tot))

morph_SD = mne.compute_source_morph(src=inv_op_SD['src'],\
                subject_from=sub_to, subject_to=C.subject_to,\
                spacing=C.spacing_morph, subjects_dir=C.data_path) 
        
# morph_LD = mne.compute_source_morph(src= inv_op_LD['src'],\
#                 subject_from=sub_to, subject_to=C.subject_to,\
#                 spacing=C.spacing_morph, subjects_dir=C.data_path) 
   
stc_SD=[]
stc_LD=[]
stc_F=[]
stc_O=[]
stc_M=[]

 

for n in np.arange(0,len(stc_sd)):
      stc_SD.append(stc_sd[n].copy().crop(t_min*1e-3,t_max*1e-3))
for n in np.arange(0,len(stc_f)):
      stc_F.append(stc_f[n].copy().crop(t_min*1e-3,t_max*1e-3))
for n in np.arange(0,len(stc_o)):
      stc_O.append(stc_o[n].copy().crop(t_min*1e-3,t_max*1e-3))
for n in np.arange(0,len(stc_m)):
      stc_M.append(stc_m[n].copy().crop(t_min*1e-3,t_max*1e-3))
      
# for n in np.arange(0,len(stc_ld)):
#       stc_LD.append(stc_ld[n].copy().crop(t_min*1e-3,t_max*1e-3))

  
morphed_labels[0].name = C.ROIs_lables[0] 

seed_ts_sd = mne.extract_label_time_course(stc_SD, morphed_labels[0], \
           src_SD, mode='mean_flip',return_generator=False)
seed_ts_f = mne.extract_label_time_course(stc_F, morphed_labels[0], \
           src_SD, mode='mean_flip',return_generator=False)
seed_ts_o = mne.extract_label_time_course(stc_O, morphed_labels[0], \
           src_SD, mode='mean_flip',return_generator=False)
seed_ts_m = mne.extract_label_time_course(stc_M, morphed_labels[0], \
           src_SD, mode='mean_flip',return_generator=False)
        

# seed_ts_ld = mne.extract_label_time_course(stc_LD, morphed_labels[0], \
#            src_LD, mode='mean_flip',return_generator=False)
 

comb_ts_sd = zip(seed_ts_sd, stc_SD)
comb_ts_f = zip(seed_ts_f, stc_F)
comb_ts_o = zip(seed_ts_o, stc_O)
comb_ts_m = zip(seed_ts_m, stc_M)


# comb_ts_ld = zip(seed_ts_ld, stc_LD)  

con_SD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    comb_ts_sd, method=method, mode='fourier', indices=indices,
    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)

con_F, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    comb_ts_f, method=method, mode='fourier', indices=indices,
    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)

con_O, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    comb_ts_o, method=method, mode='fourier', indices=indices,
    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)

con_M, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    comb_ts_m, method=method, mode='fourier', indices=indices,
    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)

                          
# con_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
#     comb_ts_ld, method=method, mode='fourier', indices=indices,
#     sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
    
con_stc_SD = mne.SourceEstimate(con_SD, vertices=vertices_SD,\
             tmin=t_min*1e-3, tstep=2e-3,subject=sub_to)

con_stc_F = mne.SourceEstimate(con_F, vertices=vertices_SD,\
             tmin=t_min*1e-3, tstep=2e-3,subject=sub_to)

con_stc_O = mne.SourceEstimate(con_O, vertices=vertices_SD,\
             tmin=t_min*1e-3, tstep=2e-3,subject=sub_to)
con_stc_M = mne.SourceEstimate(con_M, vertices=vertices_SD,\
             tmin=t_min*1e-3, tstep=2e-3,subject=sub_to)


    
# con_stc_LD = mne.SourceEstimate(con_LD, vertices=vertices_SD,\
#              tmin=t_min*1e-3, tstep=2e-3,subject=sub_to)    


stc_total_SD= morph_SD.apply(con_stc_SD)
stc_total_F= morph_SD.apply(con_stc_F)
stc_total_O= morph_SD.apply(con_stc_O)
stc_total_M= morph_SD.apply(con_stc_M)

Con_all=(stc_total_F + stc_total_O + stc_total_M)/3

con_sub= stc_total_SD -Con_all
# stc_total_LD= morph_LD.apply(con_stc_LD)

# brain = stc_total_SD.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[0],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='percent', lims=(50,75,100)))

# brain = stc_total_LD.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[0],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='percent', lims=(50,75,100)))
           
# sub=stc_total_SD-stc_total_LD
# brain = sub.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[0],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='percent', pos_lims=(50,75,100)),colormap='mne')
           
 