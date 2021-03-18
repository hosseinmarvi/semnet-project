#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:32:39 2020

@author: sr05
"""

import mne
import numpy as np
from mne.preprocessing import maxwell_filter


# path to unmaxfiltered raw data
main_path = '/megdata/cbu/semnet/'

# Path to the FIF file with cross-talk correction information.
cross_talk = '/neuro/databases_vectorview/ctc/ct_sparse.fif'

# Path to the '.dat' file with fine calibration coefficients
calibration = '/neuro/databases_vectorview/sss/sss_cal.dat'


# subjects' directories
subjects =  ['/meg16_0030/160216/',#0
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

for i in np.arange(0, len(subjects)):
    meg = subjects[i]

    # complete path to raw data 
    raw_fname_fruit = main_path + meg + 'block4_fruit.fif'  
    raw_fname_odour = main_path + meg + 'block3_odour.fif'
    raw_fname_milk  = main_path + meg + 'block2_milk.fif'
    
    # loading raw data
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk  = mne.io.Raw(raw_fname_milk , preload=True)
    

    # It is critical to mark MEG bad channels in raw.info['bads'] prior to 
    # processing in order to prevent artifact spreading
    raw_fruit.info['bads'] = []
    raw_odour.info['bads'] = []
    raw_milk.info['bads']  = []
    

    raw_sss = maxwell_filter(raw_fruit, calibration = calibration, 
              cross_talk=cross_talk , destination = raw_fname_fruit)
    

