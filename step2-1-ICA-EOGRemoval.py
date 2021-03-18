
"""
Created on Thu Mar  5 22:45:41 2020

@author: sr05
"""


import mne
import os
import numpy as np
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
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

n_components = .95  
method = 'fastica'
decim = 3  
n_max_eog = 2 #EOG061/EOG062

#for i in np.arange(10, 11):

for i in np.arange(0, 1):
    print('*************************participants : ' , i+1)
    meg = list_all[i]
    

    raw_fname_fruit = data_path + meg + 'block_fruit_tsss_notch_BPF0.1_45_raw.fif'
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)#, preload=True
    raw_fname_milk = data_path + meg + 'block_milk_tsss_notch_BPF0.1_45_raw.fif'
    raw_milk = mne.io.Raw(raw_fname_milk , preload=True)#, preload=True
    raw_fname_odour = data_path + meg + 'block_odour_tsss_notch_BPF0.1_45_raw.fif'
    raw_odour  = mne.io.Raw(raw_fname_odour, preload=True)#, preload=True
    

    picks_fruit= mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=True,
                              stim=False)
    picks_milk= mne.pick_types(raw_milk.info, meg=True, eeg=True, eog=True,
                              stim=False)
    picks_odour= mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=True,
                              stim=False)
    
    
    
    
    print('*************************participants : ' , i+1)
   
    ica_fruit = ICA(n_components=n_components, method=method)
    ica_milk  = ICA(n_components=n_components, method=method)
    ica_odour = ICA(n_components=n_components, method=method)
#    reject = dict(eeg=120e-6, grad=200e-12, mag=4e-12, eog=150e-6)
    reject = dict(grad=200e-12, mag=4e-12)

    print('*************************participants : ' , i+1)

    ica_fruit.fit(raw_fruit, picks=picks_fruit, decim=decim, reject=reject)
    ica_odour.fit(raw_odour, picks=picks_odour, decim=decim, reject=reject)
    ica_milk.fit(raw_milk, picks=picks_milk, decim=decim, reject=reject)


    eog_epochs_fruit = create_eog_epochs(raw_fruit, reject=reject) 
    eog_epochs_milk  = create_eog_epochs(raw_milk, reject=reject) 
    eog_epochs_odour = create_eog_epochs(raw_odour, reject=reject) 


    eog_epochs_fruit = create_eog_epochs(raw_fruit) 
    eog_epochs_milk  = create_eog_epochs(raw_milk) 
    eog_epochs_odour = create_eog_epochs(raw_odour) 

    eog_inds_fruit, scores_fruit = ica_fruit.find_bads_eog(eog_epochs_fruit)
    eog_inds_milk, scores_milk   = ica_milk.find_bads_eog(eog_epochs_milk)
    eog_inds_odour, scores_odour = ica_odour.find_bads_eog(eog_epochs_odour)

    eog_inds_fruit = eog_inds_fruit [:n_max_eog]
    eog_inds_milk = eog_inds_milk [:n_max_eog]
    eog_inds_odour = eog_inds_odour[:n_max_eog]
    
    print('*************************participants : ' , i+1)

    eog_inds_fruit = [0,8,15,18,22]
    eog_inds_milk = [0]
    eog_inds_odour = [1,13,22,30]

    ica_fruit.exclude += eog_inds_fruit    
    ica_milk.exclude  += eog_inds_milk
    ica_odour.exclude += eog_inds_odour
    
    ica_fruit.apply(inst=raw_fruit, exclude=eog_inds_fruit)
    ica_milk.apply(inst=raw_milk, exclude=eog_inds_milk)
    ica_odour.apply(inst=raw_odour, exclude=eog_inds_odour)

    print('*************************participants : ' , i+1)


    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    
    out_name_fruit = data_path + meg + 'block_fruit_tsss_notch_BPF0.1_45_ICAeog_raw.fif'
    out_name_milk  = data_path + meg + 'block_milk_tsss_notch_BPF0.1_45_ICAeog_raw.fif'
    out_name_odour = data_path + meg + 'block_odour_tsss_notch_BPF0.1_45_ICAeog_raw.fif'

	
             
    raw_fruit.save(out_name_fruit, overwrite=True)
    raw_milk.save(out_name_milk, overwrite=True)
    raw_odour.save(out_name_odour, overwrite=True)
    


