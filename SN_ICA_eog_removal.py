
"""
Created on Thu Mar  5 22:45:41 2020

@author: sr05
"""


import mne
import os
import numpy as np
import SN_config as C
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs


# path to filtered raw data
data_path = C.data_path

# ICA parameters
n_components = C.n_components 
n_max_eog = C.n_max_eog
method = C.ICA_method
decim = C.decim  


# subjects' directories
subjects =  C.subjects

 
for i in np.arange(0, len(subjects)):

    meg = subjects[i]    
    # complete path to raw data 
    raw_fname_fruit = data_path+meg+'block_fruit_tsss_notch_BPF0.1_45_raw.fif'
    raw_fname_odour = data_path+meg+'block_odour_tsss_notch_BPF0.1_45_raw.fif'
    raw_fname_milk  = data_path+meg+'block_milk_tsss_notch_BPF0.1_45_raw.fif'   
    raw_fname_LD    = data_path+meg+'block_LD_tsss_notch_BPF0.1_45_raw.fif'   

    # loading raw data
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk  = mne.io.Raw(raw_fname_milk , preload=True)
    raw_LD    = mne.io.Raw(raw_fname_LD   , preload=True)
    
    picks_fruit= mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=True,
                 stim=False)
    picks_odour= mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=True,
                 stim=False)
    picks_milk = mne.pick_types(raw_milk.info , meg=True, eeg=True, eog=True,
                 stim=False)
    picks_LD   = mne.pick_types(raw_LD  .info , meg=True, eeg=True, eog=True,
                 stim=False)
    
    # M/EEG signal decomposition using Independent Component Analysis (ICA)
    ica_fruit = ICA(n_components=n_components, method=method)
    ica_odour = ICA(n_components=n_components, method=method)
    ica_milk  = ICA(n_components=n_components, method=method)
    ica_LD    = ICA(n_components=n_components, method=method)

    
    reject = dict(grad=200e-12, mag=4e-12, eeg=150e-6)

    # # Run the ICA decomposition on raw data
    # ica_fruit.fit(raw_fruit, picks=picks_fruit, decim=decim, reject=reject)
    # ica_odour.fit(raw_odour, picks=picks_odour, decim=decim, reject=reject)
    # ica_milk.fit(raw_milk  , picks=picks_milk , decim=decim, reject=reject)
    # ica_LD.fit(raw_LD      , picks=picks_LD   , decim=decim, reject=reject)
    
    # # Conveniently generates epochs around EOG artifact events
    # eog_epochs_fruit = create_eog_epochs(raw_fruit, reject=reject) 
    # eog_epochs_odour = create_eog_epochs(raw_odour, reject=reject) 
    # eog_epochs_milk  = create_eog_epochs(raw_milk , reject=reject) 
    # eog_epochs_LD    = create_eog_epochs(raw_LD   , reject=reject) 
 
    
    ica_fruit.fit(raw_fruit, picks=picks_fruit, decim=decim)
    ica_odour.fit(raw_odour, picks=picks_odour, decim=decim)
    ica_milk.fit(raw_milk  , picks=picks_milk , decim=decim)
    ica_LD.fit(raw_LD      , picks=picks_LD   , decim=decim) 
    
    eog_epochs_fruit = create_eog_epochs(raw_fruit) 
    eog_epochs_odour = create_eog_epochs(raw_odour) 
    eog_epochs_milk  = create_eog_epochs(raw_milk ) 
    eog_epochs_LD    = create_eog_epochs(raw_LD   ) 


    # Detecting EOG related components using correlation
    eog_inds_fruit, scores_fruit = ica_fruit.find_bads_eog(eog_epochs_fruit)
    eog_inds_odour, scores_odour = ica_odour.find_bads_eog(eog_epochs_odour)
    eog_inds_milk , scores_milk  = ica_milk.find_bads_eog(eog_epochs_milk)
    eog_inds_LD   , scores_LD    = ica_LD.find_bads_eog(eog_epochs_LD)


    eog_inds_fruit = eog_inds_fruit [:n_max_eog]
    eog_inds_odour = eog_inds_odour[:n_max_eog]
    eog_inds_milk  = eog_inds_milk [:n_max_eog]
    eog_inds_LD    = eog_inds_LD [:n_max_eog]

    
    # Excluding the list of sources indices obtained in the previous line
    ica_fruit.exclude += eog_inds_fruit
    ica_odour.exclude += eog_inds_odour    
    ica_milk.exclude  += eog_inds_milk
    ica_LD.exclude    += eog_inds_LD

    
    # Removing the selected components from the data
    ica_fruit.apply(inst=raw_fruit, exclude=eog_inds_fruit)
    ica_odour.apply(inst=raw_odour, exclude=eog_inds_odour)
    ica_milk.apply(inst=raw_milk  , exclude=eog_inds_milk)
    ica_LD.apply(inst=raw_LD      , exclude=eog_inds_LD)


  
    # checking for the desired directory to save the data
    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    out_fname_fruit = data_path + meg + 'block_fruit_tsss_notch_BPF0.1_45_ICAeog_raw.fif'
    out_fname_odour = data_path + meg + 'block_odour_tsss_notch_BPF0.1_45_ICAeog_raw.fif'
    out_fname_milk  = data_path + meg + 'block_milk_tsss_notch_BPF0.1_45_ICAeog_raw.fif'
    out_fname_LD    = data_path + meg + 'block_LD_tsss_notch_BPF0.1_45_ICAeog_raw.fif'

    # saving cleaned data
    raw_fruit.save(out_fname_fruit, overwrite=True)
    raw_odour.save(out_fname_odour, overwrite=True)
    raw_milk.save(out_fname_milk  , overwrite=True)
    raw_LD.save(out_fname_LD      , overwrite=True)

    


