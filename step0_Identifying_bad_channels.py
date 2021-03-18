

"""
Created on Wed Mar 11 15:42:25 2020

@author: sr05
"""


import mne
import os
import numpy as np
from matplotlib import pyplot as plt
#import configparser
import csv


#import scipy.io as scio
"""
#****************************************************************************#
#                                 Filtering Data                             #
#****************************************************************************#
"""


main_path = '/imaging/rf02/Semnet/'
data_path = '/imaging/rf02/Semnet/'	# where subdirs for MEG data are
new_path = '/imaging/sr05/SemNet/SemNetData'


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

bad_channels=(['EEG008','EEG028'],#0 
              ['EEG067'],#1
              ['EEG027', 'EEG028'],#2
              ['EEG003', 'EEG007', 'EEG008', 'EEG027', 'EEG045', 'EEG057','EEG070'],#3
              ['EEG007', 'EEG008', 'EEG027', 'EEG070'],#4
              ['EEG039','EEG043'],              #5
              ['EEG023', 'EEG034','EEG039', 'EEG041','EEG047'], #6
              ['EEG003', 'EEG007', 'EEG008', 'EEG027','EEG046', 'EEG067','EEG070'],#7
              ['EEG020', 'EEG055'],   #8
              ['EEG044', 'EEG045','EEG055', 'EEG057', 'EEG059', 'EEG060'],#9
              ['EEG038', 'EEG039','EEG073'],    #10
              ['EEG044','EEG045'],    #11
              ['EEG002', 'EEG045','EEG046'],    #12
              ['EEG029','EEG039','EEG067'],    #13
              ['EEG033','EEG034', 'EEG044','EEG045','EEG046'],    #14
              ['EEG039','EEG045'],    #15
              [],    #16
              [],    #17
              ['EEG033'] #18
              )

bad_channels_fruit = dict()
bad_channels_milk = dict()
bad_channels_odour = dict()



for i in np.arange(0, len(list_all)):
    print('Participant : ', i+1)
    meg = list_all[i]

    raw_fname_fruit = main_path + meg + 'block_fruit_tsss_raw.fif'
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)#, preload=True
    raw_fname_milk = main_path + meg + 'block_milk_tsss_raw.fif'
    raw_milk = mne.io.Raw(raw_fname_milk , preload=True)#, preload=True
    raw_fname_odour = main_path + meg + 'block_odour_tsss_raw.fif'
    raw_odour  = mne.io.Raw(raw_fname_odour, preload=True)#, preload=True
    

    picks_fruit = mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_milk = mne.pick_types(raw_milk.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_odour = mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=False,
            stim=False )

#    raw_fruit_notch = raw_fruit.copy().notch_filter(freqs=50 , picks = picks_fruit)
#    raw_milk_notch  = raw_milk.copy().notch_filter(freqs=50 , picks = picks_milk)
#    raw_odour_notch = raw_odour.copy().notch_filter(freqs=50 , picks = picks_odour)

    raw_fruit.plot_sensors(ch_type='eeg')
    raw_fruit.plot(duration=1, n_channels=15, 
       title='{0} -Participant : {1}'.format(' (Fruit)',meg[1:11]))
    
    raw_milk.plot(duration=1, n_channels=15, 
       title='{0} -Participant : {1}'.format(' (Milk)', meg[1:11]))
    
    raw_odour.plot(duration=1, n_channels=15, 
       title='{0} -Participant : {1}'.format(' (Odour)', meg[1:11]))
    

    
    bad_channels_fruit[meg[1:11]] = raw_fruit.info['bads']
    bad_channels_milk[meg[1:11]] = raw_milk.info['bads'] 
    bad_channels_odour[meg[1:11]] = raw_odour.info['bads'] 
    


    
    with open('bad_channels_fruit.csv', 'a') as f:
        f.write('{},{}\n'.format(meg[1:11], raw_fruit.info['bads']))
        f.close()
       
    with open('bad_channels_milk.csv', 'a') as f:
        f.write('{},{}\n'.format(meg[1:11], raw_fruit.info['bads']))
        f.close()
        
    with open('bad_channels_odour.csv', 'a') as f:
        f.write('{},{}\n'.format(meg[1:11], raw_fruit.info['bads']))
        f.close()
