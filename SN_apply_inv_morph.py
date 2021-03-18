#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:59:42 2020

@author: sr05
"""
import numpy as np
import mne
import SN_config as C
from mne.minimum_norm import (read_inverse_operator , apply_inverse)

# path to raw data
data_path = C.data_path

# Parameters
snr = C.snr
lambda2 = C.lambda2

# subjects' directories
subjects =  C.subjects 

for i in np.arange(0,len(subjects)):
    print('Participant : ', i)
    meg = subjects[i]
    
    
    for m in np.arange(0,len(C.signal_mode)):
        inv_fname_SD = data_path + meg + 'InvOp_SD_'+C.signal_mode[m]+'-inv.fif'
        inv_op = read_inverse_operator(inv_fname_SD)   
        C.inv_op_SD[C.signal_mode[m]]=inv_op
        
        inv_fname_LD  = data_path + meg + 'InvOp_LD_'+C.signal_mode[m]+'-inv.fif'
        inv_op = read_inverse_operator(inv_fname_LD)   
        C.inv_op_LD[C.signal_mode[m]]=inv_op
        
## SD Source Estimation
    for keys in C.fname_SD.keys():
       epoch_fname = data_path + meg + C.fname_SD[keys]
       epochs = mne.read_epochs(epoch_fname, preload=True)

       for n in np.arange(0,len(C.signal_mode)):
           if keys == 'SD_words':
               evoked = epochs.average().set_eeg_reference(ref_channels = \
                        'average',projection=True)
               stc = apply_inverse(evoked, C.inv_op_SD[C.signal_mode[n]],lambda2,
                      method ='MNE', pick_ori=None)
                 
               stc_fname = C.data_path + meg + 'block_'+keys+'_'+ \
                   C.signal_mode[n]
               stc.save(stc_fname)
                        
               # setup source morph         
               morph = mne.compute_source_morph( src=C.inv_op_SD\
                  [C.signal_mode[n]]['src'], subject_from = stc.subject, 
                  subject_to = C.subject_to , spacing = C.spacing_morph, 
                  subjects_dir = C.data_path)
               stc_fsaverage = morph.apply(stc)
               stc_fname_fsaverage = C.data_path + meg + 'block_'+keys+'_'+\
                                     C.signal_mode[n]+'_fsaverage'
               stc_fsaverage.save(stc_fname_fsaverage)
           else:
               
               for s in np.arange(0,len(C.SD_categories)):
                   evoked = epochs[C.SD_categories[s]].average().\
                            set_eeg_reference(ref_channels = 'average',
                            projection=True)
                   stc = apply_inverse(evoked, C.inv_op_SD[C.signal_mode[n]],
                         lambda2,method ='MNE', pick_ori=None)
                    
                   stc_fname = C.data_path + meg + 'block_'+keys+'_'+\
                   C.SD_categories[s]+'_'+ C.signal_mode[n]
                   stc.save(stc_fname)
                   
                   morph = mne.compute_source_morph( src=C.inv_op_SD\
                        [C.signal_mode[n]]['src'], subject_from = stc.subject, 
                        subject_to = C.subject_to , spacing = C.spacing_morph, 
                        subjects_dir = C.data_path)
                   stc_fsaverage = morph.apply(stc)
                   stc_fname_fsaverage = C.data_path + meg + 'block_'+keys+'_'+\
                       C.SD_categories[s]+'_'+C.signal_mode[n]+'_fsaverage'
                   stc_fsaverage.save(stc_fname_fsaverage)
## LD Source Estimation
    for keys in C.fname_LD.keys():
       epoch_fname = data_path + meg + C.fname_LD[keys]
       epochs = mne.read_epochs(epoch_fname, preload=True)

       for n in np.arange(0,len(C.signal_mode)):
           if keys == 'LD_words':
               evoked = epochs.average().set_eeg_reference(ref_channels = \
                        'average',projection=True)
               stc = apply_inverse(evoked, C.inv_op_LD[C.signal_mode[n]],
                     lambda2,method ='MNE', pick_ori=None)
                 
               stc_fname = C.data_path + meg + 'block_'+keys+'_'+\
                   C.signal_mode[n]
               stc.save(stc_fname)  
               
               morph = mne.compute_source_morph( src=C.inv_op_LD\
                        [C.signal_mode[n]]['src'], subject_from = stc.subject, 
                        subject_to = C.subject_to , spacing = C.spacing_morph, 
                        subjects_dir = C.data_path)
               stc_fsaverage = morph.apply(stc)
               stc_fname_fsaverage = C.data_path + meg + 'block_'+keys+'_'+\
                       C.signal_mode[n]+'_fsaverage'
               stc_fsaverage.save(stc_fname_fsaverage)       
           else:
               
                for s in np.arange(0,len(C.LD_categories)):
                    evoked = epochs[C.LD_categories[s]].average().\
                        set_eeg_reference(ref_channels = 'average',
                        projection=True)
                    stc = apply_inverse(evoked, C.inv_op_LD[C.signal_mode[n]],
                        lambda2, method ='MNE', pick_ori=None)
                    
                    stc_fname = C.data_path + meg + 'block' +keys+'_'+\
                                C.LD_categories[s]+'_'+ C.signal_mode[n]
                    stc.save(stc_fname) 
                    
                    morph = mne.compute_source_morph( src=C.inv_op_LD\
                        [C.signal_mode[n]]['src'], subject_from = stc.subject, 
                        subject_to = C.subject_to , spacing = C.spacing_morph, 
                        subjects_dir = C.data_path)
                    stc_fsaverage = morph.apply(stc)
                    stc_fname_fsaverage = C.data_path + meg + 'block_'+keys+'_'+\
                           C.LD_categories[s]+'_'+C.signal_mode[n]+'_fsaverage'
                
  