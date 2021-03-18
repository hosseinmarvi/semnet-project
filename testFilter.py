
"""
Created on Thu Mar  5 19:16:48 2020

@author: sr05
"""

import mne
import os
import numpy as np
from matplotlib import pyplot as plt
import mne
import os
import numpy as np
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs

#import scipy.io as scio
"""
#****************************************************************************#
#                                 Filtering Data                               #
#****************************************************************************#
"""


main_path = '/imaging/rf02/Semnet/'
data_path = '/imaging/rf02/Semnet/'	# where subdirs for MEG data are
new_path = '/imaging/sr05/SemNet/SemNetData'

n_components = .99  
method = 'fastica'
decim = 3  
n_max_eog = 2 #EOG061/EOG062

list_all =  ['/meg16_0030/160216/']

bad_channels_fruit = {'meg16_0030': ['EEG034']}
                      

bad_channels_milk = {'meg16_0030': ['EEG034']}


bad_channels_odour = {'meg16_0030': ['EEG008', 'EEG028', 'EEG034']}


lfreq=1
h_freq=48

for i in np.arange(0, len(list_all)):
    print('Participant : ', i+1)
    meg = list_all[i]

    raw_fname_fruit = main_path + meg + 'block_fruit_tsss_raw.fif'
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)#, preload=True
    raw_fname_milk = main_path + meg + 'block_milk_tsss_raw.fif'
    raw_milk = mne.io.Raw(raw_fname_milk , preload=True)#, preload=True
    raw_fname_odour = main_path + meg + 'block_odour_tsss_raw.fif'
    raw_odour  = mne.io.Raw(raw_fname_odour, preload=True)#, preload=True
    

    
     
    raw_fruit.info['bads'] = bad_channels_fruit[meg[1:11]]
    raw_milk.info['bads']  = bad_channels_milk[meg[1:11]]
    raw_odour.info['bads'] = bad_channels_odour[meg[1:11]]

  
    raw_fruit.interpolate_bads(reset_bads = True , mode = 'accurate')
    raw_milk.interpolate_bads(reset_bads = True , mode = 'accurate')
    raw_odour.interpolate_bads(reset_bads = True , mode = 'accurate')

    
    raw_fruit.set_eeg_reference( ref_channels = 'average')
    raw_milk.set_eeg_reference( ref_channels = 'average')
    raw_odour.set_eeg_reference( ref_channels = 'average')
    
    
    picks_fruit = mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_milk = mne.pick_types(raw_milk.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_odour = mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=False,
            stim=False )
#
#
#
    raw_fruit_notch = raw_fruit.copy().notch_filter(freqs=50 , picks = picks_fruit)
    raw_milk_notch  = raw_milk.copy().notch_filter(freqs=50 , picks = picks_milk)
    raw_odour_notch = raw_odour.copy().notch_filter(freqs=50 , picks = picks_odour)



    raw_fruit_notch_BPF1 = raw_fruit_notch.copy().filter(l_freq=lfreq, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_fruit)
    raw_milk_notch_BPF1 =  raw_milk_notch.copy().filter(l_freq=lfreq, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_milk)
    raw_odour_notch_BPF1 = raw_odour_notch.copy().filter(l_freq=lfreq, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_odour)
    
    raw_fruit_notch_BPF01 = raw_fruit_notch.copy().filter(l_freq=0.1, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_fruit)
    raw_milk_notch_BPF01 =  raw_milk_notch.copy().filter(l_freq=0.1, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_milk)
    raw_odour_notch_BPF01 = raw_odour_notch.copy().filter(l_freq=0.1, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_odour)
    
#...................................ICA......................................#

    picks_fruit1 = mne.pick_types(raw_fruit_notch_BPF1 .info, meg=True,
                 eeg=True, eog=True, stim=False)
   
    
     
    ica_fruit = ICA(n_components=n_components, method=method)
    ica_milk  = ICA(n_components=n_components, method=method)
    ica_odour = ICA(n_components=n_components, method=method)
#    reject = dict(eeg=120e-6, grad=200e-12, mag=4e-12, eog=150e-6)
    reject = dict(grad=200e-12, mag=4e-12)


#    reject = dict(mag=4e-12, grad=4000e-13)
    ica_fruit.fit(raw_fruit, picks=picks_fruit, decim=decim, reject=reject)
    ica_odour.fit(raw_odour, picks=picks_odour, decim=decim, reject=reject)
    ica_milk.fit(raw_milk, picks=picks_milk, decim=decim, reject=reject)

    
    eog_epochs_fruit = create_eog_epochs(raw_fruit, reject=reject) 
    eog_epochs_milk  = create_eog_epochs(raw_milk, reject=reject) 
    eog_epochs_odour = create_eog_epochs(raw_odour, reject=reject) 

    eog_inds_fruit, scores_fruit = ica_fruit.find_bads_eog(eog_epochs_fruit)
    eog_inds_milk, scores_milk   = ica_milk.find_bads_eog(eog_epochs_milk)
    eog_inds_odour, scores_odour = ica_odour.find_bads_eog(eog_epochs_odour)

    eog_inds_fruit = eog_inds_fruit [:n_max_eog]
    eog_inds_milk = eog_inds_milk [:n_max_eog]
    eog_inds_odour = eog_inds_odour[:n_max_eog]

    ica_fruit.exclude += eog_inds_fruit    
    ica_milk.exclude  += eog_inds_milk
    ica_odour.exclude += eog_inds_odour
    
    ica_fruit.apply(inst=raw_fruit, exclude=eog_inds_fruit)
    ica_milk.apply(inst=raw_milk, exclude=eog_inds_milk)
    ica_odour.apply(inst=raw_odour, exclude=eog_inds_odour)




