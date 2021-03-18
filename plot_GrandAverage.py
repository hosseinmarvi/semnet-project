#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 03:09:24 2020

@author: sr05
"""


import mne
import numpy as np
import os
from matplotlib import pyplot as plt

all_evokeds_SD_words=list()
all_evokeds_LD_word = list()

Nb_Ave_SD = 0
nb_ave_SD_array=list()
Nb_Ave = 0
nb_ave_array = list()

data_path = '/imaging/sr05/SemNet/SemNetData'
# pictures_path = os.path.expanduser('~') + '/Python/pictures/SD-LD_task_pictures/SD_LD/'

pictures_path = os.path.expanduser('~') + '/Python/pictures/evoked_white/'


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

# list_a=[1,3,4,6,8,9,13,14,15,16,18]
for i in np.arange(0,len(list_all)-18):
# for i in list_a:

# for i in np.array([0,1,2,4,5,6,7,8,9,10,11,12,13,15,16,17,18]):


    print('Participant : ', i)
    meg=list_all[i]
    print(meg)
    # all_evokeds_SD_words=list()
    # all_evokeds_LD_word=list()


    fname_fruit = data_path + meg + 'block_fruit_evoked_categories.npy'
    fname_odour = data_path + meg + 'block_odour_evoked_categories.npy'
    fname_milk  = data_path + meg + 'block_milk_evoked_categories.npy'
    fname_LD    = data_path + meg + 'block_LD_evoked_categories.npy'

    
    evoked_fruit = np.load( fname_fruit , allow_pickle='True').item()
    evoked_odour = np.load( fname_odour , allow_pickle='TRUE').item()
    evoked_milk  = np.load( fname_milk  , allow_pickle='TRUE').item()
    evoked_LD    = np.load( fname_LD    , allow_pickle='TRUE').item()



    nb_ave_SD=0   
    evoked_fruit_visual = evoked_fruit['visual']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_visual.nave
    
    evoked_fruit_hear = evoked_fruit['hear']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_hear.nave
    
    evoked_fruit_hand = evoked_fruit['hand']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_hand.nave
     
    evoked_fruit_neutral = evoked_fruit['neutral']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_neutral.nave

    evoked_fruit_emotional = evoked_fruit['emotional']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_emotional.nave

  

    evoked_milk_visual = evoked_milk['visual']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_visual.nave

    evoked_milk_hear = evoked_milk['hear']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_hear.nave

    evoked_milk_hand = evoked_milk['hand']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_hand.nave

    evoked_milk_neutral = evoked_milk['neutral']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_neutral.nave

    evoked_milk_emotional= evoked_milk['emotional']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_emotional.nave


 
    evoked_odour_visual = evoked_odour['visual']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_visual.nave

    evoked_odour_hear = evoked_odour['hear']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_hear.nave

    evoked_odour_hand = evoked_odour['hand']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_hand.nave

    evoked_odour_neutral = evoked_odour['neutral']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_neutral.nave

    evoked_odour_emotional= evoked_odour['emotional']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_emotional.nave

    Nb_Ave_SD =  Nb_Ave_SD  + nb_ave_SD 
    nb_ave_SD_array.append(nb_ave_SD )

   
    nb_ave = 0
    evoked_LD_visual = evoked_LD['visual']
    nb_ave =  nb_ave  + evoked_LD_visual.nave

    evoked_LD_hear = evoked_LD['hear']
    nb_ave =  nb_ave  + evoked_LD_hear.nave

    evoked_LD_hand = evoked_LD['hand']
    nb_ave =  nb_ave  + evoked_LD_hand.nave

    evoked_LD_neutral = evoked_LD['neutral']
    nb_ave =  nb_ave  + evoked_LD_neutral.nave

    evoked_LD_emotional = evoked_LD['emotional']
    nb_ave =  nb_ave  + evoked_LD_emotional.nave

    Nb_Ave =  Nb_Ave  + nb_ave 
    nb_ave_array.append(nb_ave )

    
    all_evokeds_SD_words.append(evoked_fruit_visual)
    all_evokeds_SD_words.append(evoked_fruit_hand)
    all_evokeds_SD_words.append(evoked_fruit_hear)
    all_evokeds_SD_words.append(evoked_fruit_neutral)
    all_evokeds_SD_words.append(evoked_fruit_emotional)
    
    all_evokeds_SD_words.append(evoked_milk_visual)
    all_evokeds_SD_words.append(evoked_milk_hand)
    all_evokeds_SD_words.append(evoked_milk_hear)
    all_evokeds_SD_words.append(evoked_milk_neutral)
    all_evokeds_SD_words.append(evoked_milk_emotional)
    
    all_evokeds_SD_words.append(evoked_odour_visual)
    all_evokeds_SD_words.append(evoked_odour_hand)
    all_evokeds_SD_words.append(evoked_odour_hear)
    all_evokeds_SD_words.append(evoked_odour_neutral)
    all_evokeds_SD_words.append(evoked_odour_emotional)


    all_evokeds_LD_word.append( evoked_LD_visual )
    all_evokeds_LD_word.append( evoked_LD_hear) 
    all_evokeds_LD_word.append( evoked_LD_hand )
    all_evokeds_LD_word.append( evoked_LD_neutral )
    all_evokeds_LD_word.append( evoked_LD_emotional )
    

    
    
    # # ********************************SD*************************************#
    Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
    
    # fig = Grand_Average_SD_words.plot_white(noise_cov=cov_SD, show=True)
    # fig.suptitle('Participant : ' +meg[1:11]+' - SD Task')
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'evoked_white_SD')   

    
    # ts_args = dict(gfp=True)
    # topomap_args = dict(sensors=True) 
    
    # # EEG Grand Average for each individual 
    # Grand_Average_SD_words.plot_joint( times='peaks',
    #           picks = 'eeg', title='*Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - SD_Words (EEG)', '-Nave: ' + str(nb_ave_SD )),ts_args=ts_args,
    #           topomap_args=topomap_args)
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'SD_Words_EEG')   
    
    # # mag Grand Average for each individual 
    # Grand_Average_SD_words.plot_joint( times='peaks',
    #           picks = 'mag', title='*Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - SD_Words (Mag)', '-Nave: ' + str(nb_ave_SD )),ts_args=ts_args,
    #           topomap_args=topomap_args)
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'SD_Words_Mag') 
    
    # # grad Grand Average for each individual 
    # Grand_Average_SD_words.plot_joint( times='peaks', 
    #           picks = 'grad', title='*Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - SD_Words (Grad)', '-Nave: ' + str(nb_ave_SD )),ts_args=ts_args,
    #           topomap_args=topomap_args)
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'SD_Words_Grad') 
    
    
    # # # ********************************LD*************************************#

    Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
    # fig = Grand_Average_LD_words.plot_white(noise_cov=cov_LD, show=True)
    # fig.suptitle('Participant : ' +meg[1:11]+' - LD Task')
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'evoked_white_LD')   

    # ts_args = dict(gfp=True)
    # topomap_args = dict(sensors=True) 
    
    # # EEG Grand Average for each individual   
    # Grand_Average_LD_words.plot_joint( times='peaks', 
    #           picks = 'eeg', title='*Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - LD_Words (EEG)', '-Nave: ' + str(nb_ave )),ts_args=ts_args,
    #           topomap_args=topomap_args)
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11]+'LD_Words_EEG')
    
    # # mag Grand Average for each individual 
    # Grand_Average_LD_words.plot_joint( times='peaks',
    #           picks = 'mag',title='*Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - LD_Words (Mag)', '-Nave: ' + str(nb_ave )),ts_args=ts_args,
    #           topomap_args=topomap_args)
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11]+'LD_Words_Mag')
    
    
    # # grad Grand Average for each individual      
    # Grand_Average_LD_words.plot_joint( times='peaks', 
    #           picks = 'grad', title='*Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - LD_Words (Grad)', '-Nave: ' + str(nb_ave )),ts_args=ts_args,
    #           topomap_args=topomap_args)
    # plt.savefig(pictures_path + 'Participant : ' +meg[1:11]+'LD_Words_Grad')
    

 
    
    

# Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 

# # EEG Grand Average across all individuals    
# Grand_Average_SD_words.plot_joint( times='peaks', picks = 'eeg', 
#           title='*Grand Average (EEG)- SD_Words -Nave: '+str(Nb_Ave_SD ),ts_args=ts_args,
#           topomap_args=topomap_args)
# # plt.savefig(pictures_path + 'Grand Average-SD_Words_EEG')   

# # mag Grand Average across all individuals     
# Grand_Average_SD_words.plot_joint( times='peaks', picks = 'mag', 
#           title='*Grand Average (Mag) - SD_Words-Nave: '+str(Nb_Ave_SD ),ts_args=ts_args,
#           topomap_args=topomap_args)
# # plt.savefig(pictures_path + 'Grand Average-SD_Words_Mag') 

# # grad Grand Average across all individuals     
# Grand_Average_SD_words.plot_joint( times='peaks', picks = 'grad', 
#           title='*Grand Average (Grad) - SD_Words-Nave: '+str(Nb_Ave_SD ),ts_args=ts_args,
#           topomap_args=topomap_args)
# # plt.savefig(pictures_path + 'Grand Average-SD_Words_Grad') 

# # # #*****************************************************************************#
# Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 

# # EEG Grand Average across all individuals      
# Grand_Average_LD_words.plot_joint( times='peaks', picks = 'eeg', 
#           title='*Grand Average (EEG) - LD_Words-Nave: '+ str(Nb_Ave ),ts_args=ts_args,
#           topomap_args=topomap_args)
# # plt.savefig(pictures_path + 'Grand Average-LD_Words_EEG')

# # mag Grand Average across all individuals    

# Grand_Average_LD_words.plot_joint( times='peaks', picks = 'mag', 
#           title='*Grand Average (Mag) - LD_Words-Nave: '+str(Nb_Ave ),ts_args=ts_args,
#           topomap_args=topomap_args)
# # plt.savefig(pictures_path + 'Grand Average-LD_Words_Mag')

# # grad Grand Average across all individuals    
# Grand_Average_LD_words.plot_joint( times='peaks', picks = 'grad', 
#           title='*Grand Average (Grad) - LD_Words-Nave: '+str(Nb_Ave ),ts_args=ts_args,
#           topomap_args=topomap_args)
# # plt.savefig(pictures_path + 'Grand Average-LD_Words_Grad')




