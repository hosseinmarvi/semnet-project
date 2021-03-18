
"""
Created on Fri Mar 20 01:05:42 2020

@author: sr05
"""
import mne
import os
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
from mne.preprocessing import create_ecg_epochs
from PIL import Image



"""
#****************************************************************************#
#                                 Filtering Data                             #
#****************************************************************************#
"""

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


bad_channels = { 'meg16_0030': ['EEG034', 'EEG053', 'EEG046', 'EEG067', 'EEG070', 'EEG071'],
                 'meg16_0032': ['EEG048', 'EEG045', 'EEG027'],
                 'meg16_0034': ['EEG010', 'EEG072', 'EEG071', 'EEG069', 'EEG073', 'EEG074', 
                                'EEG060', 'EEG059','EEG066'],
                 'meg16_0035': ['EEG071', 'EEG069', 'EEG068', 'EEG070', 'EEG073'],
                 'meg16_0042': ['EEG072', 'EEG071', 'EEG074', 'EEG073', 'EEG070'],
                 'meg16_0045': ['EEG034', 'EEG058', 'EEG071', 'EEG066'],
                 'meg16_0052': ['EEG067'],
                 'meg16_0056': ['EEG069', 'EEG074'],
                 'meg16_0069': ['EEG043', 'EEG057', 'EEG047', 'EEG046'],
                 'meg16_0070': ['EEG019', 'EEG053', 'EEG072', 'EEG071'],
                 'meg16_0072': ['EEG066'],
                 'meg16_0073': ['EEG072', 'EEG071'],
                 'meg16_0075': ['EEG073', 'EEG071', 'EEG068', 'EEG072'],
                 'meg16_0078': ['EEG027', 'EEG016', 'EEG074', 'EEG073'],
                 'meg16_0082': ['EEG047'],
                 'meg16_0086': ['EEG034', 'EEG071'],
                 'meg16_0097': ['EEG013'],
                 'meg16_0122': ['EEG067'],
                 'meg16_0125': ['EEG035', 'EEG032', 'EEG069', 'EEG071', 'EEG067']}


lfreq=0.1
h_freq=45
n_components = .95  
method = 'fastica'
decim = 3  
n_max_eog = 2 #EOG061/EOG062
n_max_ecg = 1
tmin, tmax = -0.3, 0.6
stim_delay = 0.034 # delay in s

a = 18
for i in np.arange(a, a+1):
    print('***********************Participant : ', i+1)
    meg = list_all[i]

    raw_fname = data_path + meg + 'block_LD_tsss_raw.fif'
    raw = mne.io.Raw(raw_fname, preload=True)#, preload=True

#***************************Indicating bad channels***************************#    
#    picks_fruit = mne.pick_types(raw.info, meg=True, eeg=True, eog=False,
#        stim=False )
#    raw.plot(duration=1, n_channels=15, 
#       title='{0} -Participant : {1}'.format(' (Raw)',meg[1:11]))
#    
#    
#    
#    bad_channels[meg[1:11]] = raw.info['bads']
#
#    with open('bad_channels_LD.csv', 'a') as f:
#        f.write('{},{}\n'.format(meg[1:11], raw.info['bads']))
#        f.close()

#******************************* Filtering *******************************#    
  
    raw.info['bads'] = bad_channels[meg[1:11]]
    raw.interpolate_bads(reset_bads = True , mode = 'accurate')
    raw.set_eeg_reference( ref_channels = 'average')
    
    
    picks= mne.pick_types(raw.info, meg=True, eeg=True, eog=False,
            stim=False )

    raw_notch = raw.copy().notch_filter(freqs=50 , picks = picks)
 

    raw_notch_BPF = raw_notch.copy().filter(l_freq=lfreq, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks)


#    if not os.path.isdir(new_path + meg):
#        os.makedirs(new_path + meg)
#
#    
#    out_name= new_path + meg + 'block_LD_tsss_notch_BPF0.1_45_raw.fif'
#    raw_notch_BPF.save(out_name, overwrite=True)
#    
#******************************* EOG/ECG*******************************#    




    raw = raw_notch_BPF
    
    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,
          stim=False)
    
    ica = ICA(n_components=n_components, method=method)
   
    reject = dict(grad=200e-12, mag=4e-12)

    ica.fit(raw, picks=picks, decim=decim, reject=reject)
   
    
    eog_epochs = create_eog_epochs(raw) 
    ecg_epochs = create_ecg_epochs(raw , reject=reject) 

    eog_inds , scores_eog = ica.find_bads_eog(eog_epochs)
    ecg_inds , scores_ecg = ica.find_bads_ecg(ecg_epochs)

    eog_inds = eog_inds[:n_max_eog]
    ecg_inds = ecg_inds[:n_max_ecg]
    
#    eog_inds = [0,33]


    ica.exclude += eog_inds  
    ica.exclude += ecg_inds   

    ica.apply(inst=raw , exclude=eog_inds)
    ica.apply(inst=raw , exclude=ecg_inds)
        



#pick_av = np.abs((scores_eog[0]+scores_eog[1])/2).argsort()[::-1][:5]
#pick_0 = np.abs(scores_eog[0]).argsort()[::-1][:5]
#pick_1 = np.abs(scores_eog[1]).argsort()[::-1][:5]
#
#pick_t=[0,2,31,35,33,11,25,30,13,23]
#
##
##ica_odour.plot_components(picks=[0,43,30,1])
##ica_fruit.plot_components()
ica.plot_components([16,10,35,22,50,18])
##
##ica.plot_properties(raw)
#ica.plot_properties(raw, picks=pick_t, psd_args={'fmax': 45.})
ica.plot_sources(raw,picks=[50,20,47,22,28,33,38,35,4,8,11,10,16,18])

  

#    if not os.path.isdir(new_path + meg):
#        os.makedirs(new_path + meg)
#
#    
#    out_name = new_path + meg + 'block_LD_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
#    raw.save(out_name , overwrite=True)



#******************************* Epoching ******************************#    



    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,
                              stim=False)
  
    
    
    events = mne.find_events(raw , stim_channel='STI101',
                                   min_duration=0.001 , shortest_event=1)
   
    event_id = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
                'pwordc': 6 , 'pworda': 7}
    color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 6: 'red' ,
             7 : 'blue'}
                
    
    events[:,0] = events[:,0] + np.round( raw.info['sfreq']*stim_delay )
 
    reject = dict(eeg=120e-6, grad=200e-12, mag=4e-12)#, eog=150e-6)

    epochs  = mne.Epochs(raw , events, event_id, tmin, tmax, 
            picks=picks , proj=True, baseline=(tmin, 0), reject=reject)
 

#    if not os.path.isdir(new_path + meg):
#        os.makedirs(data_path + meg)
#   
#    out_name = new_path + meg + 'block_LD_epochs-epo.fif'     
#    epochs.save(out_name , overwrite=True)
 
#******************************* Evoked ******************************#    

#for i in np.arange(0, len(list_all)):
#    print('***********************Participant : ', i+1)
#    meg = list_all[i]
#    
#    epoch_fname = new_path + meg + 'block_LD_epochs-epo.fif'
#    epochs      = mne.read_epochs(epoch_fname , preload=True)#, preload=True
#    
#
#    raw_fname  = new_path + meg + 'block_LD_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
#    raw        = mne.io.Raw(raw_fname , preload=True)#, preload=True
#   
         
    


    picks = mne.pick_types(epochs.info, meg=True, eeg=True)
#

    evoked_visual   = epochs['visual'].average(picks=picks)
    evoked_hear     = epochs['hear'].average(picks=picks)
    evoked_hand     = epochs['hand'].average(picks=picks)
    evoked_neutral  = epochs['neutral'].average(picks=picks)
    evoked_emotional= epochs['emotional'].average(picks=picks)
## 
    
#    evoked_LD={'visual':evoked_visual , 'hear':evoked_hear,
#            'hand':evoked_hand, 'neutral':evoked_neutral,
#            'emotional':evoked_emotional, 'pwordc':evoked_pwordc,
#            'pworda':evoked_pworda}

#
#    if not os.path.isdir(new_path + meg):
#        os.makedirs(new_path + meg)
#
##    
#    out_name = new_path + meg + 'block_LD_evoked.npy'
#    np.save(out_name, evoked_LD)

#******************************* Visualization ******************************#    

#data_path = '/imaging/sr05/SemNet/SemNetData/'
#for i in np.arange(2, len(list_all)):
#    meg = list_all[i]
# 
#    for file in os.listdir(data_path + meg):
#        if file.endswith('epo.fif') or file.endswith('evoked.npy'):
#            os.remove(data_path + meg + file)


#******************************* Visualization ******************************#    
    

#
data_path = '/imaging/sr05/SemNet/SemNetData'
pictures_path = os.path.expanduser('~') + '/Python/pictures/'

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
Nb_Ave = 0
nb_ave_array = list()
all_evokeds_word = list()



for i in np.arange(0,len(list_all)):
    print('Participant : ', i)
    meg=list_all[i]
    
    all_evokeds_word=list()
    all_evokeds_pword=list()
    all_evokeds=list()


    evoked_LD = data_path + meg + 'block_LD_evoked.npy'
    read_evoked_LD = np.load( evoked_LD , allow_pickle='TRUE').item()

    nb_ave = 0
    evoked_LD_visual   = read_evoked_LD['visual']
    nb_ave =  nb_ave  + evoked_LD_visual.nave

    evoked_LD_hear     = read_evoked_LD['hear']
    nb_ave =  nb_ave  + evoked_LD_hear.nave

    evoked_LD_hand     = read_evoked_LD['hand']
    nb_ave =  nb_ave  + evoked_LD_hand.nave

    evoked_LD_neutral  = read_evoked_LD['neutral']
    nb_ave =  nb_ave  + evoked_LD_neutral.nave

    evoked_LD_emotional= read_evoked_LD['emotional']
    nb_ave =  nb_ave  + evoked_LD_emotional.nave
    Nb_Ave =  Nb_Ave  + nb_ave 
    nb_ave_array.append(nb_ave )


    
    all_evokeds_word.append( evoked_LD_visual )
    all_evokeds_word.append( evoked_LD_hear) 
    all_evokeds_word.append( evoked_LD_hand )
    all_evokeds_word.append( evoked_LD_neutral )
    all_evokeds_word.append( evoked_LD_emotional )

    

    
    
    grand_average_word  = mne.grand_average(all_evokeds_word)
    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=True)
    
    grand_average_word.plot_joint( times='peaks', picks = 'eeg', 
            title='Participant : {0}{1}{2}'.format(meg[1:11],
              ' - LD_Words', '-Nave: ' +str(nb_ave )), ts_args=ts_args,
              topomap_args=topomap_args)
#             
#    plt.savefig(pictures_path + 'Participant-'+ meg[1:11]+ '-SD_Words')   

#
#
#
#
##grand_average_word  = mne.grand_average(all_evokeds_word)
##
##ts_args = dict(gfp=True)
##topomap_args = dict(sensors=True)
##    
##grand_average_word.plot_joint( times='peaks', picks = 'eeg', 
##          title='Grand Average - LD_Words',ts_args=ts_args, 
##          topomap_args=topomap_args)
##plt.savefig(pictures_path + 'Grand Average -LD_Words')   
##
##
##    
##image_list = []
##
##for image in os.listdir(pictures_path)[1:]:
##    if image.endswith('.png'):
##        image_list.append(Image.open(pictures_path + image).convert('RGB'))
##img = Image.open(pictures_path + 'Participant-meg16_0030-Words1.png').convert('RGB')
##img.save(pictures_path + 'all_participants.pdf', save_all=True, append_images=image_list)
