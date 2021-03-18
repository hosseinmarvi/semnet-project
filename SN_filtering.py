
"""
Created on Thu Mar  5 19:16:48 2020

@author: sr05
"""

import mne
import os
import numpy as np
import SN_config as C

# path to maxfiltered raw data
main_path = C.main_path
data_path = C.data_path

# filter parameters
l_freq , h_freq = C.l_freq , C.h_freq

# subjects' directories
subjects = C.subjects

# EEG bad channels
EEG_bad_channels_fruit = C.EEG_bad_channels_fruit
EEG_bad_channels_odour = C.EEG_bad_channels_odour
EEG_bad_channels_milk  = C.EEG_bad_channels_milk
EEG_bad_channels_LD    = C.EEG_bad_channels_LD

# MEG bad channels
MEG_bad_channels_fruit = C.MEG_bad_channels_fruit         
MEG_bad_channels_odour = C.MEG_bad_channels_odour
MEG_bad_channels_milk  = C.MEG_bad_channels_milk
MEG_bad_channels_LD    = C.MEG_bad_channels_LD

for i in np.arange(0,len(subjects)):
    print('Participants : ',i)
    meg = subjects[i]
    
    # complete path to raw data 
    raw_fname_fruit = main_path + meg + 'block_fruit_tsss_raw.fif'
    raw_fname_odour = main_path + meg + 'block_odour_tsss_raw.fif'
    raw_fname_milk  = main_path + meg + 'block_milk_tsss_raw.fif'
    raw_fname_LD    = main_path + meg + 'block_LD_tsss_raw.fif'

    
    # loading raw data
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk  = mne.io.Raw(raw_fname_milk , preload=True)
    raw_LD    = mne.io.Raw(raw_fname_LD   , preload=True)
    
    # Adding MEG bad channels to raw.info 
    raw_fruit.info['bads'].extend(MEG_bad_channels_fruit[i] + EEG_bad_channels_fruit[i])
    raw_odour.info['bads'].extend(MEG_bad_channels_odour[i] + EEG_bad_channels_odour[i])
    raw_milk.info['bads'].extend(MEG_bad_channels_milk[i] + EEG_bad_channels_milk[i])
    raw_LD.info['bads'].extend(MEG_bad_channels_LD[i] + EEG_bad_channels_LD[i])


    #Interpolateing MEG bad channels
    raw_fruit.interpolate_bads(reset_bads = True , mode = 'fast')
    raw_odour.interpolate_bads(reset_bads = True , mode = 'fast')
    raw_milk.interpolate_bads(reset_bads = True  , mode = 'fast')
    raw_LD.interpolate_bads(reset_bads = True    , mode = 'fast')
  

    #re-referencing the data according to the desired reference
    raw_fruit.set_eeg_reference( ref_channels = 'average')
    raw_odour.set_eeg_reference( ref_channels = 'average')
    raw_milk.set_eeg_reference(  ref_channels = 'average')
    raw_LD.set_eeg_reference(    ref_channels = 'average')

 
    picks_fruit = mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_odour = mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_milk = mne.pick_types(raw_milk.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_LD   = mne.pick_types(raw_LD.info, meg=True, eeg=True, eog=False,
            stim=False )

    # Notch filter for the raw data
    raw_fruit_notch = raw_fruit.copy().notch_filter(freqs=50,picks=picks_fruit)
    raw_odour_notch = raw_odour.copy().notch_filter(freqs=50,picks=picks_odour)
    raw_milk_notch  = raw_milk.copy().notch_filter(freqs=50,picks=picks_milk)
    raw_LD_notch    = raw_LD.copy().notch_filter(freqs=50,picks=picks_LD)


    # Band-pass filtering
    raw_fruit_notch_BPF = raw_fruit_notch.copy().filter(l_freq=l_freq, 
                  h_freq=h_freq, fir_design='firwin' , picks = picks_fruit)
    raw_odour_notch_BPF = raw_odour_notch.copy().filter(l_freq=l_freq, 
                  h_freq=h_freq, fir_design='firwin' , picks = picks_odour)
    raw_milk_notch_BPF =  raw_milk_notch.copy().filter(l_freq=l_freq, 
                  h_freq=h_freq, fir_design='firwin' , picks = picks_milk)
    raw_LD_notch_BPF =  raw_LD_notch.copy().filter(l_freq=l_freq, 
                  h_freq=h_freq, fir_design='firwin' , picks = picks_LD)
    
   
    
    # checking for the desired directory to save the data
    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    
    out_name_fruit = data_path + meg + 'block_fruit_tsss_notch_BPF0.1_45_raw.fif'
    out_name_odour = data_path + meg + 'block_odour_tsss_notch_BPF0.1_45_raw.fif'
    out_name_milk  = data_path + meg + 'block_milk_tsss_notch_BPF0.1_45_raw.fif'
    out_name_LD    = data_path + meg + 'block_LD_tsss_notch_BPF0.1_45_raw.fif'

	
    # saving filtered data
         
    raw_fruit_notch_BPF.save(out_name_fruit, overwrite=True)
    raw_odour_notch_BPF.save(out_name_odour, overwrite=True)
    raw_milk_notch_BPF.save(out_name_milk, overwrite=True)
    raw_LD_notch_BPF.save(out_name_LD, overwrite=True)



