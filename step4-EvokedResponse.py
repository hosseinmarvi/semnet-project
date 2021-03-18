
"""
Created on Mon Mar  9 08:03:09 2020

@author: sr05
"""


"""
Created on Fri Mar  6 14:29:15 2020

@author: sr05
"""



import mne
import os
import numpy as np
import numpy as np
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
"""
#****************************************************************************#
#                                 Filtering Data                               #
#****************************************************************************#
"""

data_path = '/imaging/sr05/SemNet/SemNetData'



list_all =  ['/meg16_0030/160216/',#0
            '/meg16_0032/160218/', #1
            '/meg16_0034/160219/', #2
            '/meg16_0035/160222/', #3
            '/meg16_0042/160229/', #4
            '/meg16_0045/160303/', #5
            '/meg16_0052/160310/', #6
            '/meg16_0056/160314/', #7
            '/meg16_0069/160405/', #8 
            '/meg16_0070/160407/', #9
            '/meg16_0072/160408/', #10
            '/meg16_0073/160411/', #11
            '/meg16_0075/160411/', #12
            '/meg16_0078/160414/', #13
            '/meg16_0082/160418/', #14
            '/meg16_0086/160422/', #15
            '/meg16_0097/160512/', #16 
            '/meg16_0122/160707/', #17
            '/meg16_0125/160712/', #18
            ]



tmin, tmax = -0.3, 0.6
stim_delay = 0.034 # delay in s
nt_event = list(range(1,6))


#
# for i in np.arange(0,len(list_all)-18):
for i in [13]:

    print('**************Participant : ', i)
    meg=list_all[i]
    

    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epochs_fruit       = mne.read_epochs(epoch_fname_fruit, preload=True)#, preload=True
    epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    epochs_milk         = mne.read_epochs(epoch_fname_milk, preload=True)
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    

    # raw_fname_fruit = data_path + meg + 'block_fruit_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
    # raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)#, preload=True
    # raw_fname_milk = data_path + meg + 'block_milk_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
    # raw_milk = mne.io.Raw(raw_fname_milk , preload=True)#, preload=True
    # raw_fname_odour = data_path + meg + 'block_odour_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
    # raw_odour  = mne.io.Raw(raw_fname_odour, preload=True)#, preload=True
    
         
    print('**************Participant : ', i)

    # events_fruit = mne.find_events(raw_fruit, stim_channel='STI101',
    #                                min_duration=0.001 , shortest_event=1)
    # events_milk = mne.find_events(raw_milk, stim_channel='STI101',
    #                                    min_duration=0.001 , shortest_event=1) 
    # events_odour = mne.find_events(raw_odour, stim_channel='STI101',
    #                                min_duration=0.001 , shortest_event=1)     
    
        
    # event_id = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
    #             'pwordc': 6, 'target': 8}
    
    # color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 6: 'red' ,
    #          8: 'blue'}    
    
    # events_fruit[:,0] = events_fruit[:,0] + np.round( raw_fruit.info['sfreq']*stim_delay )
    # events_milk[:,0] = events_milk[:,0] + np.round( raw_milk.info['sfreq']*stim_delay )
    # events_odour[:,0] = events_odour[:,0] + np.round( raw_odour.info['sfreq']*stim_delay )
    
    
    
    # FP_fruit = 0
    # FN_fruit = 0
    # for evcnt in range(events_fruit.shape[0]-2):    
    #     if (events_fruit[evcnt,2]in nt_event and  events_fruit[evcnt+2,2]>100):
    #        events_fruit[evcnt,2]=7777
    #        FP_fruit = FP_fruit + 1
           
    #     elif(events_fruit[evcnt,2] == 8 and events_fruit[evcnt+2,2]<100):
    #         FN_fruit = FN_fruit + 1
           
           
    # FP_milk = 0
    # FN_milk = 0
    # for evcnt in range(events_milk.shape[0]-2):    
    #     if (events_milk[evcnt,2] in nt_event and events_milk[evcnt+2,2]>100):
    #         events_milk[evcnt,2] = 7777
    #         FP_milk = FP_milk + 1
           
    #     elif(events_milk[evcnt,2] == 8 and events_milk[evcnt+2,2]<100):
    #         FN_milk = FN_milk + 1
           
#     FP_odour = 0
#     FN_odour = 0
#     for evcnt in range(events_odour.shape[0]-2):    
#         if (events_odour[evcnt,2]in nt_event and  events_odour[evcnt+2,2]>100):
#            events_odour[evcnt,2]=7777
#            FP_odour = FP_odour + 1
           
#         elif(events_odour[evcnt,2] == 8 and events_odour[evcnt+2,2]<100):
#            FN_odour = FN_odour + 1
# #           
# #       
# #           
    
#     reject = dict(eeg=120e-6, grad=200e-12, mag=4e-12)#, eog=150e-6)
# #    event_ids = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4,
# #                     'emotional': 5,'pwordc': 6}
    
    
    print('**************Participant : ', i)


    picks_fruit = mne.pick_types(epochs_fruit.info, meg=True, eeg=True)
    picks_milk  = mne.pick_types(epochs_milk.info, meg=True, eeg=True)
    picks_odour = mne.pick_types(epochs_odour.info, meg=True, eeg=True)


    evoked_fruit_visual   = epochs_fruit['visual'].average(picks=picks_fruit)
    evoked_fruit_hear     = epochs_fruit['hear'].average(picks=picks_fruit)
    evoked_fruit_hand     = epochs_fruit['hand'].average(picks=picks_fruit)
    evoked_fruit_neutral  = epochs_fruit['neutral'].average(picks=picks_fruit)
    evoked_fruit_emotional= epochs_fruit['emotional'].average(picks=picks_fruit)
    evoked_fruit_pwordc   = epochs_fruit['pwordc'].average(picks=picks_fruit)
    evoked_fruit_target   = epochs_fruit['target'].average(picks=picks_fruit)
    
    evoked_milk_visual   = epochs_milk['visual'].average(picks=picks_milk)
    evoked_milk_hear     = epochs_milk['hear'].average(picks=picks_milk)
    evoked_milk_hand     = epochs_milk['hand'].average(picks=picks_milk)
    evoked_milk_neutral  = epochs_milk['neutral'].average(picks=picks_milk)
    evoked_milk_emotional= epochs_milk['emotional'].average(picks=picks_milk)
    evoked_milk_pwordc   = epochs_milk['pwordc'].average(picks=picks_milk)
    evoked_milk_target   = epochs_milk['target'].average(picks=picks_milk)
    
    evoked_odour_visual   = epochs_odour['visual'].average(picks=picks_odour)
    evoked_odour_hear     = epochs_odour['hear'].average(picks=picks_odour)
    evoked_odour_hand     = epochs_odour['hand'].average(picks=picks_odour)
    evoked_odour_neutral  = epochs_odour['neutral'].average(picks=picks_odour)
    evoked_odour_emotional= epochs_odour['emotional'].average(picks=picks_odour)
    evoked_odour_pwordc   = epochs_odour['pwordc'].average(picks=picks_odour)
    evoked_odour_target   = epochs_odour['target'].average(picks=picks_odour)
    
    evoked_fruit={'visual':evoked_fruit_visual , 'hear':evoked_fruit_hear,
            'hand':evoked_fruit_hand, 'neutral':evoked_fruit_neutral,
            'emotional':evoked_fruit_emotional, 'pwordc':evoked_fruit_pwordc,
            'target':evoked_fruit_target}
    
    evoked_milk={'visual':evoked_milk_visual , 'hear':evoked_milk_hear,
            'hand':evoked_milk_hand, 'neutral':evoked_milk_neutral,
            'emotional':evoked_milk_emotional, 'pwordc':evoked_milk_pwordc,
            'target':evoked_milk_target}
    
    evoked_odour={'visual':evoked_odour_visual , 'hear':evoked_odour_hear,
            'hand':evoked_odour_hand, 'neutral':evoked_odour_neutral,
            'emotional':evoked_odour_emotional, 'pwordc':evoked_odour_pwordc,
            'target':evoked_odour_target}

    print('**************Participant : ', i)

    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

#    
    out_name_fruit = data_path + meg + 'block_fruit_evoked.npy'
    out_name_milk  = data_path + meg + 'block_milk_evoked.npy'
    out_name_odour = data_path + meg + 'block_odour_evoked.npy'
    
     
    np.save(out_name_fruit, evoked_fruit)
    np.save(out_name_milk, evoked_milk)
    np.save(out_name_odour, evoked_odour)
    

    
    




