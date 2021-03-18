#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:09:28 2020

@author: sr05
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:34:31 2020

@author: sr05
"""

import mne
import numpy as np
import sn_config as C
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

# path to filtered raw data
main_path = C.main_path
data_path = C.data_path
pictures_path = C.pictures_path_Source_estimate

# subjects' directories
subjects =  C.subjects

for n in np.arange(0,len(C.signal_mode)):
    # stc_SD_words_all = C.stc_SD_words_all
    # stc_LD_words_all = C.stc_LD_words_all
    # stc_SD_LD_words_all = C.stc_SD_LD_words_all
    
    for i in np.arange(0, len(subjects)):

    
        print('participant : ' , i , C.signal_mode[n])
        subject_from = subjects[i]
        meg = subjects[i]

    
        fname_SD_words_fsaverage = C.data_path + meg + 'block_SD_words_'+\
                                   C.signal_mode[n]+'_fsaverage'
        fname_LD_words_fsaverage = C.data_path + meg + 'block_LD_words_'+\
                                   C.signal_mode[n]+'_fsaverage'    
        stc_SD_words = mne.read_source_estimate(fname_SD_words_fsaverage )
        stc_LD_words = mne.read_source_estimate(fname_LD_words_fsaverage )
        if (i==0):
            stc_SD_all_words = stc_SD_words
            stc_LD_all_words = stc_LD_words
        else: 
            stc_SD_all_words = stc_SD_all_words + stc_SD_words
            stc_LD_all_words = stc_LD_all_words + stc_LD_words

# a=stc_SD_all_words.copy().crop(0.100,0.500).mean().data.squeeze()
# brain_SD = a.plot(surface='inflated', hemi='lh',subject = 'fsaverage', 
#           subjects_dir=data_path,time_viewer =False,title=
#           'SD_words_all',colorbar=False, size=(500, 400))

stc_SD_all_words = stc_SD_all_words/len(subjects)  
stc_LD_all_words = stc_LD_all_words/len(subjects)     
   
stc_SD_LD =  stc_SD_words - stc_LD_words 

brain_SD = stc_SD_all_words.plot(surface='inflated', hemi='split',subject = 'fsaverage', 
          subjects_dir=data_path,initial_time= 0.200,time_viewer =False,title=
          'SD_words',colorbar=True, size=(800, 400))

brain_LD= stc_LD_all_words.plot(surface='inflated', hemi='split',subject = 'fsaverage', 
          subjects_dir=data_path,initial_time= 0.600,time_viewer =False,title=
          'LD_words',colorbar=True, size=(800, 400))


brain_SL= stc_SD_LD.plot(surface='inflated', hemi='split',subject = 'fsaverage', 
          subjects_dir=data_path,initial_time= 0.500,time_viewer =False,title=
          'SD-LD',colorbar=True, size=(800, 400))


stc_SD_100_300 = stc_SD_all_words.copy().crop(0.200,0.250).mean() 
stc_LD_100_300 = stc_LD_all_words.copy().crop(0.100,0.300).mean()    
   
stc_SD_LD_100_300 =  stc_SD_LD.copy().crop(0.100,0.300).mean() 

