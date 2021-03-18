
"""
Created on Fri Mar  6 14:29:15 2020

@author: sr05
"""

import mne
import os
import numpy as np
import sn_config as C


# path to filtered raw data
data_path = C.data_path

# Parameters
tmin, tmax = C.tmin , C.tmax
stim_delay = C.stim_delay
category_code = C.category_code

# Events info
event_id_SD = C.event_id_sd
event_id_LD = C.event_id_ld
 
reject = C.reject

# subjects' directories
subjects =  C.subjects
 
for i in np.arange(0, len(subjects)):
    print('participant : ' , i)
    meg = subjects[i]
    
    # complete path to raw data 
    raw_fname_fruit = data_path+meg+'block_fruit_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
    raw_fname_odour = data_path+meg+'block_odour_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
    raw_fname_milk  = data_path+meg+'block_milk_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'   
    raw_fname_LD    = data_path+meg+'block_LD_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'   


    # loading raw data
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk  = mne.io.Raw(raw_fname_milk , preload=True)
    raw_LD    = mne.io.Raw(raw_fname_LD   , preload=True)


    picks_fruit = mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=False,
                  stim=False)
    picks_odour = mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=False,
                  stim=False) 
    picks_milk  = mne.pick_types(raw_milk.info , meg=True, eeg=True, eog=False,
                  stim=False)
    picks_LD    = mne.pick_types(raw_LD.info   , meg=True, eeg=True, eog=False,
                  stim=False)
    
    # Find events from raw file.  
    events_fruit = mne.find_events(raw_fruit, stim_channel='STI101',
                   min_duration=0.001 , shortest_event=1)
    events_odour = mne.find_events(raw_odour, stim_channel='STI101',
                   min_duration=0.001 , shortest_event=1) 
    events_milk = mne.find_events(raw_milk, stim_channel='STI101',
                  min_duration=0.001 , shortest_event=1) 
    events_LD   = mne.find_events(raw_LD, stim_channel='STI101',
                  min_duration=0.001 , shortest_event=1) 
        

    # Considering the device(!) delay
    events_fruit[:,0] = events_fruit[:,0] + np.round( raw_fruit.info['sfreq']*stim_delay )
    events_odour[:,0] = events_odour[:,0] + np.round( raw_odour.info['sfreq']*stim_delay )
    events_milk[:,0]  = events_milk[:,0]  + np.round( raw_milk.info['sfreq']*stim_delay )
    events_LD[:,0]    = events_LD[:,0]    + np.round( raw_LD.info['sfreq']*stim_delay )

    # Finding events with false responses
    for evcnt in range(events_fruit.shape[0]-2):    
        if (events_fruit[evcnt,2]in category_code and  events_fruit[evcnt+2,2]>100):
            events_fruit[evcnt,2] = 7777   
        elif(events_fruit[evcnt,2] == 8 and events_fruit[evcnt+2,2]<100):
            events_fruit[evcnt,2] = 8888  
            
    for evcnt in range(events_odour.shape[0]-2):    
        if (events_odour[evcnt,2]in category_code and  events_odour[evcnt+2,2]>100):
           events_odour[evcnt,2] = 7777
        elif(events_odour[evcnt,2] == 8 and events_odour[evcnt+2,2]<100):
           events_odour[evcnt,2] = 8888 

    for evcnt in range(events_milk.shape[0]-2):    
        if (events_milk[evcnt,2] in category_code and events_milk[evcnt+2,2]>100):
            events_milk[evcnt,2] = 7777
        elif(events_milk[evcnt,2] == 8 and events_milk[evcnt+2,2]<100):
            events_milk[evcnt,2] = 8888 
            
    for evcnt in np.arange(0,events_LD.shape[0]-2):    
        if (events_LD[evcnt,2] in category_code  and  events_LD[evcnt+2,2]!=16384):
            events_LD[evcnt,2] = 7777          
        elif(events_LD[evcnt,2] in np.array([6,7,9]) and events_LD[evcnt+2,2]!=4096):
            events_LD[evcnt,2] = 8888
         



    # Extracting epochs from a raw instance
    epochs_fruit = mne.Epochs(raw_fruit, events_fruit, event_id_SD, tmin, tmax, 
            picks=picks_fruit, proj=True, baseline=(tmin, 0), reject=reject)
    epochs_odour = mne.Epochs(raw_odour, events_odour, event_id_SD, tmin, tmax, 
            picks=picks_odour, proj=True, baseline=(tmin, 0), reject=reject)
    epochs_milk  = mne.Epochs(raw_milk, events_milk, event_id_SD, tmin, tmax, 
            picks=picks_milk, proj=True, baseline=(tmin, 0), reject=reject)
    epochs_LD    = mne.Epochs(raw_LD, events_LD, event_id_LD, tmin, tmax, 
            picks=picks_LD, proj=True, baseline=(tmin, 0), reject=reject)


    # checking for the existance of desired directory to save the data
    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    
    out_name_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    out_name_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    out_name_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    out_name_LD    = data_path + meg + 'block_LD_epochs-epo.fif'

     
    # saving epochs       
    epochs_fruit.save(out_name_fruit, overwrite=True)
    epochs_odour.save(out_name_odour, overwrite=True)
    epochs_milk.save(out_name_milk, overwrite=True)
    epochs_LD.save(out_name_LD, overwrite=True)

    
    # Saving events 
    mne.write_events(data_path + meg + 'block_fruit-eve.fif', events_fruit)
    mne.write_events(data_path + meg + 'block_odour-eve.fif', events_odour)
    mne.write_events(data_path + meg + 'block_milk-eve.fif' , events_milk)
    mne.write_events(data_path + meg + 'block_LD-eve.fif'   , events_LD)