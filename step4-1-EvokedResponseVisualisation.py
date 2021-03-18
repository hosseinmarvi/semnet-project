
"""
Created on Mon Mar  9 09:23:48 2020

@author: sr05
"""



import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from PIL import Image

#from fpdf import FPDF

"""
#****************************************************************************#
#                                 Filtering Data                               #
#****************************************************************************#
"""



data_path = '/imaging/sr05/SemNet/SemNetData'
pictures_path = os.path.expanduser('~') + '/Python/pictures/SD_task_pictures/'

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

all_evokeds_visual=list()
all_evokeds_hear=list()
all_evokeds_hand=list()
all_evokeds_neutral=list()
all_evokeds_emotional=list()


all_evokeds_SD_words=list()
Nb_Ave_SD = 0
nb_ave_SD_array=list()
Nb_Ave = 0
nb_ave_array = list()
all_evokeds_LD_word = list()

nb_F_v_SD=list()
nb_F_h_SD=list()
nb_F_ha_SD=list()
nb_F_n_SD=list()
nb_F_e_SD=list()

nb_M_v_SD=list()
nb_M_h_SD=list()
nb_M_ha_SD=list()
nb_M_n_SD=list()
nb_M_e_SD=list()

nb_O_v_SD=list()
nb_O_h_SD=list()
nb_O_ha_SD=list()
nb_O_n_SD=list()
nb_O_e_SD=list()

nb_v_LD=list()
nb_h_LD=list()
nb_ha_LD=list()
nb_n_LD=list()
nb_e_LD=list()
for i in np.arange(0,len(list_all)):
# for i in [13]:

    print('Participant : ', i)
    meg=list_all[i]
# #    
    all_evokeds_visual=list()
    all_evokeds_hear=list()
    all_evokeds_hand=list()
    all_evokeds_neutral=list()
    all_evokeds_emotional=list()
    all_evokeds_SD_words=list()
    all_evokeds_LD_word=list()

##    

    evoked_fruit = data_path + meg + 'block_fruit_evoked.npy'
    evoked_milk  = data_path + meg + 'block_milk_evoked.npy'
    evoked_odour = data_path + meg + 'block_odour_evoked.npy'
    evoked_LD    = data_path + meg + 'block_LD_evoked.npy'
    
    read_evoked_fruit = np.load( evoked_fruit , allow_pickle='True').item()
    read_evoked_milk  = np.load( evoked_milk , allow_pickle='TRUE').item()
    read_evoked_odour = np.load( evoked_odour , allow_pickle='TRUE').item()
    read_evoked_LD    = np.load( evoked_LD , allow_pickle='TRUE').item()
    
#    All_list_fruit_evoked = [All_list_fruit_evoked , read_evoked_fruit]



    nb_ave_SD=0
   
    evoked_fruit_visual = read_evoked_fruit['visual']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_visual.nave
    nb_F_v_SD.append(evoked_fruit_visual.nave)
    
    evoked_fruit_hear     = read_evoked_fruit['hear']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_hear.nave
    nb_F_h_SD.append(evoked_fruit_hear.nave)
    
    evoked_fruit_hand = read_evoked_fruit['hand']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_hand.nave
    nb_F_ha_SD.append(evoked_fruit_hand.nave)
     
    evoked_fruit_neutral = read_evoked_fruit['neutral']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_neutral.nave
    nb_F_n_SD.append(evoked_fruit_neutral.nave)

    evoked_fruit_emotional = read_evoked_fruit['emotional']
    nb_ave_SD =  nb_ave_SD  + evoked_fruit_emotional.nave
    nb_F_e_SD.append(evoked_fruit_emotional.nave)

  

    evoked_milk_visual = read_evoked_milk['visual']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_visual.nave
    nb_M_v_SD.append(evoked_milk_visual.nave)

    evoked_milk_hear = read_evoked_milk['hear']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_hear.nave
    nb_M_h_SD.append(evoked_milk_hear.nave)


    evoked_milk_hand = read_evoked_milk['hand']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_hand.nave
    nb_M_ha_SD.append(evoked_milk_hand.nave)


    evoked_milk_neutral = read_evoked_milk['neutral']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_neutral.nave
    nb_M_n_SD.append(evoked_milk_neutral.nave)


    evoked_milk_emotional= read_evoked_milk['emotional']
    nb_ave_SD =  nb_ave_SD  + evoked_milk_emotional.nave
    nb_M_e_SD.append(evoked_milk_emotional.nave)


 
    evoked_odour_visual = read_evoked_odour['visual']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_visual.nave
    nb_O_v_SD.append(evoked_odour_visual.nave)


    evoked_odour_hear = read_evoked_odour['hear']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_hear.nave
    nb_O_h_SD.append(evoked_odour_hear.nave)


    evoked_odour_hand = read_evoked_odour['hand']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_hand.nave
    nb_O_ha_SD.append(evoked_odour_hand.nave)


    evoked_odour_neutral = read_evoked_odour['neutral']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_neutral.nave
    nb_O_n_SD.append(evoked_odour_neutral.nave)


    evoked_odour_emotional= read_evoked_odour['emotional']
    nb_ave_SD =  nb_ave_SD  + evoked_odour_emotional.nave
    nb_O_e_SD.append(evoked_odour_emotional.nave)

    Nb_Ave_SD =  Nb_Ave_SD  + nb_ave_SD 
    nb_ave_SD_array.append(nb_ave_SD )

   
    nb_ave = 0
    evoked_LD_visual   = read_evoked_LD['visual']
    nb_ave =  nb_ave  + evoked_LD_visual.nave
    nb_v_LD.append(evoked_LD_visual.nave)


    evoked_LD_hear     = read_evoked_LD['hear']
    nb_ave =  nb_ave  + evoked_LD_hear.nave
    nb_h_LD.append(evoked_LD_hear.nave)


    evoked_LD_hand     = read_evoked_LD['hand']
    nb_ave =  nb_ave  + evoked_LD_hand.nave
    nb_ha_LD.append(evoked_LD_hand.nave)


    evoked_LD_neutral  = read_evoked_LD['neutral']
    nb_ave =  nb_ave  + evoked_LD_neutral.nave
    nb_n_LD.append(evoked_LD_neutral.nave)


    evoked_LD_emotional= read_evoked_LD['emotional']
    nb_ave =  nb_ave  + evoked_LD_emotional.nave
    nb_e_LD.append(evoked_LD_emotional.nave)

    Nb_Ave =  Nb_Ave  + nb_ave 
    nb_ave_array.append(nb_ave )


#    all_evokeds_visual.append( evoked_fruit_visual) 
#    all_evokeds_visual.append( evoked_milk_visual) 
#    all_evokeds_visual.append( evoked_odour_visual) 
#    
#    all_evokeds_hear.append( evoked_fruit_hear) 
#    all_evokeds_hear.append( evoked_milk_hear) 
#    all_evokeds_hear.append( evoked_odour_hear) 
#    
#    all_evokeds_hand.append( evoked_fruit_hand) 
#    all_evokeds_hand.append( evoked_milk_hand) 
#    all_evokeds_hand.append( evoked_odour_hand) 
#    
#    all_evokeds_neutral.append( evoked_fruit_neutral) 
#    all_evokeds_neutral.append( evoked_milk_neutral) 
#    all_evokeds_neutral.append( evoked_odour_neutral) 
#    
#    all_evokeds_emotional.append( evoked_fruit_emotional) 
#    all_evokeds_emotional.append( evoked_milk_emotional) 
#    all_evokeds_emotional.append( evoked_odour_emotional) 
#    
#    all_evokeds_pwordc.append( evoked_fruit_pwordc ) 
#    all_evokeds_pwordc.append( evoked_milk_pwordc) 
#    all_evokeds_pwordc.append( evoked_odour_pwordc) 
#    
#    all_evokeds_target.append( evoked_fruit_target) 
#    all_evokeds_target.append( evoked_milk_target) 
#    all_evokeds_target.append( evoked_odour_target) 
#    
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
    
    
    
    Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=True) 
      
    Grand_Average_SD_words.plot_joint( times=[0.089,0.135,0.200,0.260,0.298],
              picks = 'eeg', title='Participant : {0}{1}{2}'.format(meg[1:11],
              ' - SD_Words (EEG)', '-Nave: ' + str(nb_ave_SD )),ts_args=ts_args,
              topomap_args=topomap_args)
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'SD_Words_EEG')   
    
    
    Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=True) 
      
    Grand_Average_SD_words.plot_joint( times=[0.083,0.127,0.200,0.258,0.357,0.525],
              picks = 'mag', title='Participant : {0}{1}{2}'.format(meg[1:11],
              ' - SD_Words (Mag)', '-Nave: ' + str(nb_ave_SD )),ts_args=ts_args,
              topomap_args=topomap_args)
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'SD_Words_Mag') 
    
    
    Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=True) 
      
    Grand_Average_SD_words.plot_joint( times=[0.071,0.120,0.198,0.353], 
              picks = 'grad', title='Participant : {0}{1}{2}'.format(meg[1:11],
              ' - SD_Words (Grad)', '-Nave: ' + str(nb_ave_SD )),ts_args=ts_args,
              topomap_args=topomap_args)
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'SD_Words_Grad') 

    # #*****************************************************************************#
    Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=True) 
      
    Grand_Average_LD_words.plot_joint( times=[0.090, 0.135, 0.196, 0.292, 0.513], 
              picks = 'eeg', title='Participant : {0}{1}{2}'.format(meg[1:11],
              ' - LD_Words (EEG)', '-Nave: ' + str(nb_ave )),ts_args=ts_args,
              topomap_args=topomap_args)
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11]+'LD_Words_EEG')
    
    
    Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=True) 
      
    Grand_Average_LD_words.plot_joint( times=[0.083,0.125,0.197,0.266,0.346],
              picks = 'mag',title='Participant : {0}{1}{2}'.format(meg[1:11],
              ' - LD_Words (Mag)', '-Nave: ' + str(nb_ave )),ts_args=ts_args,
              topomap_args=topomap_args)
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11]+'LD_Words_Mag')
    
    
    
    Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=True) 
      
    Grand_Average_LD_words.plot_joint( times=[0.075,0.122,0.196,0.288,0.454], 
              picks = 'grad', title='Participant : {0}{1}{2}'.format(meg[1:11],
              ' - LD_Words (Grad)', '-Nave: ' + str(nb_ave )),ts_args=ts_args,
              topomap_args=topomap_args)
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11]+'LD_Words_Grad')
    

 
    
    

# Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 
  
# Grand_Average_SD_words.plot_joint( times=[0.089,0.135,0.200,0.260,0.298], picks = 'eeg', 
#           title='Grand Average (EEG)- SD_Words -Nave: '+str(Nb_Ave_SD ),ts_args=ts_args,
#           topomap_args=topomap_args)
# plt.savefig(pictures_path + 'Grand Average-SD_Words_EEG')   

# Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 
  
# Grand_Average_SD_words.plot_joint( times=[0.083,0.127,0.200,0.258,0.357,0.525], picks = 'mag', 
#           title='Grand Average (Mag) - SD_Words-Nave: '+str(Nb_Ave_SD ),ts_args=ts_args,
#           topomap_args=topomap_args)
# plt.savefig(pictures_path + 'Grand Average-SD_Words_Mag') 

# Grand_Average_SD_words = mne.grand_average(all_evokeds_SD_words)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 
  
# Grand_Average_SD_words.plot_joint( times=[0.071,0.120,0.198,0.353], picks = 'grad', 
#           title='Grand Average (Grad) - SD_Words-Nave: '+str(Nb_Ave_SD ),ts_args=ts_args,
#           topomap_args=topomap_args)
# plt.savefig(pictures_path + 'Grand Average-SD_Words_Grad') 

# # # #*****************************************************************************#
# Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 
  
# Grand_Average_LD_words.plot_joint( times=[0.090, 0.135, 0.196, 0.292, 0.513], picks = 'eeg', 
#           title='Grand Average (EEG) - LD_Words-Nave: '+ str(Nb_Ave ),ts_args=ts_args,
#           topomap_args=topomap_args)
# plt.savefig(pictures_path + 'Grand Average-LD_Words_EEG')
# # 
# Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 
  
# Grand_Average_LD_words.plot_joint( times=[0.083,0.125,0.197,0.266,0.346], picks = 'mag', 
#           title='Grand Average (Mag) - LD_Words-Nave: '+str(Nb_Ave ),ts_args=ts_args,
#           topomap_args=topomap_args)
# plt.savefig(pictures_path + 'Grand Average-LD_Words_Mag')

# Grand_Average_LD_words = mne.grand_average(all_evokeds_LD_word)
# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True) 
  
# Grand_Average_LD_words.plot_joint( times=[0.075,0.122,0.196,0.288,0.454], picks = 'grad', 
#           title='Grand Average (Grad) - LD_Words-Nave: '+str(Nb_Ave ),ts_args=ts_args,
#           topomap_args=topomap_args)
# plt.savefig(pictures_path + 'Grand Average-LD_Words_Grad')



image_list = []
path_list = []

for file in os.listdir(pictures_path):
    if file.endswith('.png'):
        path_list.append(file)

path_list.sort()

for image in path_list[1:]:
    image_list.append(Image.open(pictures_path + image).convert('RGB'))

img = Image.open(pictures_path + path_list[0]).convert('RGB')
img.save(pictures_path + 'LD-SD-Words.pdf', save_all=True, append_images=image_list)



#
#
#    
# image_list = []

# for image in os.listdir(pictures_path)[1:]:
#     if image.endswith('.png'):
#         image_list.append(Image.open(pictures_path + image).convert('RGB'))
# img = Image.open(pictures_path + 'Participant-meg16_0030-SD_Words.png').convert('RGB')
# img.save(pictures_path + 'SD_all_participants.pdf', save_all=True, append_images=image_list)
#

#   
#    All_evoked_fruit = [evoked_fruit_visual , evoked_fruit_hear , 
#        evoked_fruit_hand , evoked_fruit_neutral , evoked_fruit_emotional , 
#        evoked_fruit_pwordc , evoked_fruit_target]
#    
#    All_evoked_milk = [evoked_milk_visual , evoked_milk_hear , 
#        evoked_milk_hand , evoked_milk_neutral , evoked_milk_emotional , 
#        evoked_milk_pwordc , evoked_milk_target]
##    
#    All_evoked_odour= [evoked_odour_visual , evoked_odour_hear , 
#        evoked_odour_hand , evoked_odour_neutral , evoked_odour_emotional , 
#        evoked_odour_pwordc , evoked_odour_target]
#    
##    
##    evoked_fruit_visual.plot(titles=None)
##    evoked_fruit_hear.plot(titles=None)
##    evoked_fruit_hand.plot(titles=None)
##
##
##    
##    for j in evoked_fruit_visual.ch_names[::-1][:5]:
##        evoked_fruit_visual.plot(picks=j) 
##
##    evoked_fruit_visual.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='eeg',
##           title='Participant : {0} - (Visual)'.format(meg[1:11]))
##    evoked_fruit_hear.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='eeg',
##           title='Participant : {0} - (Hear)'.format(meg[1:11]))        
##    evoked_fruit_hand.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='eeg', 
##           title='Participant : {0} - (Hand)'.format(meg[1:11]))        
##    evoked_fruit_neutral.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='eeg',
##           title='Participant : {0} - (Neutral)'.format(meg[1:11]))        
##    evoked_fruit_emotional.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='eeg',
##           title='Participant : {0} - (Emotional)'.format(meg[1:11]))        
##    evoked_fruit_pwordc.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='eeg', 
##           title='Participant : {0} - (Pwordc)'.format(meg[1:11]))        
##    evoked_fruit_target.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='eeg', 
##           title='Participant : {0} - (Target)'.format(meg[1:11]))        
##
###    mne.viz.plot_evoked_topo(evoked_fruit_visual,title='visual')
##    
##
###***************************************************************************#
##    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20,20))
##    ax = axes[0]
##    mne.viz.plot_compare_evokeds(read_evoked_fruit , picks='mag',
##           title='{0} -Participant : {1}'.format('Mag/Grad/EEG (Fruit)',meg[1:11]),
##           axes=ax)
##    
##    ax = axes[1]
##    mne.viz.plot_compare_evokeds(read_evoked_fruit , picks='grad', title = None, 
##           axes=ax , show_legend= None)
##    
##    ax = axes[2]
##    mne.viz.plot_compare_evokeds(read_evoked_fruit , picks='eeg', title = None,
##            axes=ax , show_legend= None)
###****************************************************************************#
##    evoked_fruit_visual.plot_joint( picks = 'eeg', 
##            title='Participant : {0} {1}'.format(meg[1:11], ' - Fruit/Visual'))
##    evoked_fruit_hear.plot_joint( picks = 'eeg', 
##            title='Participant : {0} {1}'.format(meg[1:11], ' - Fruit/Hear'))
##    evoked_fruit_hand.plot_joint( picks = 'eeg', 
##            title='Participant : {0} {1}'.format(meg[1:11], ' - Fruit/Hand'))
#    
#****************************************************************************#
#grand_average_visual = mne.grand_average(all_evokeds_visual)
#grand_average_hear   = mne.grand_average(all_evokeds_hear)
#grand_average_hand   = mne.grand_average(all_evokeds_hand)
#grand_average_neutral = mne.grand_average(all_evokeds_neutral)
#grand_average_emotional = mne.grand_average(all_evokeds_emotional)
#grand_average_pwordc = mne.grand_average(all_evokeds_pwordc)
#grand_average_target = mne.grand_average(all_evokeds_target)

#Grand_Average_NonTarget = mne.grand_average(all_evokeds_NonTarget)


#grand_average_visual.plot_joint( picks = 'eeg', title='Grand Average :  Visual')
#grand_average_hear.plot_joint( picks = 'eeg', title='Grand Average :  Hear')
#grand_average_hand.plot_joint( picks = 'eeg', title='Grand Average :  Hand')
#grand_average_neutral.plot_joint( picks = 'eeg', title='Grand Average :  Neutral')
#grand_average_emotional.plot_joint( picks = 'eeg', title='Grand Average :  Emotional')
#grand_average_pwordc.plot_joint( picks = 'eeg', title='Grand Average :  Pword')
#grand_average_target.plot_joint( picks = 'eeg', title='Grand Average :  Target')
#Grand_Average_NonTarget.plot_joint( times=[0,0.09,0.135,0.220,0.300],
    #picks = 'eeg', title='Grand Average :  NonTarget')    
