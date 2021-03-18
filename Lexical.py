
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


#
#"""
##****************************************************************************#
##                                 Filtering Data                             #
##****************************************************************************#
#"""
#
main_path = '/imaging/rf02/Semnet/'	# where subdirs for MEG data are
data_path = '/imaging/sr05/SemNet/SemNetData'
#
#
#
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
#
#
bad_channels = { 'meg16_0030': ['EEG034', 'EEG053', 'EEG046', 'EEG067', 
                                'EEG070', 'EEG071', 'EEG045', 'EEG028', 
                                'EEG037', 'EEG038', 'EEG048', 'EEG049',
                                'EEG031'],
                'meg16_0032': ['EEG048', 'EEG045', 'EEG027'],
                'meg16_0034': ['EEG010', 'EEG072', 'EEG071', 'EEG069',
                                'EEG073', 'EEG074', 'EEG009', 'EEG050',
                                'EEG060', 'EEG059', 'EEG066', 'EEG002', 
                                'EEG004', 'EEG005', 'EEG006', 'EEG007',
                                'EEG008', 'EEG016', 'EEG025', 'EEG003',
                                'EEG009', 'EEG067', 'EEG071', 'MEG1421',
                                'MEG1431','MEG1531','MEG1541','MEG1711', 
                                'MEG1741','MEG1421','EEG028', 'EEG015',
                                'EEG031', 'EEG039', 'EEG026', 'MEG0141',
                                'MEG2621','MEG2631','MEG0111','MEG0131', 
                                'MEG0141','MEG1411','MEG1421','MEG1431',
                                'MEG1441','MEG1511','MEG1521','MEG1531', 
                                'MEG1541','MEG1711','MEG1721','MEG2531', 
                                'MEG2611','MEG2621','MEG2631','MEG2641'],
                'meg16_0035': [ 'EEG071', 'EEG069', 'EEG068', 'EEG070',
                                'EEG073', 'EEG002', 'EEG008', 'EEG016',
                                'EEG017', 'EEG005', 'EEG049', 'EEG004', 
                                'EEG005', 'EEG065', 'EEG015', 'EEG067',
                                'EEG072', 'EEG054', 'EEG011'],
                'meg16_0042': [ 'EEG072', 'EEG071', 'EEG074', 'EEG073', 
                                'EEG070', 'EEG006', 'EEG002', 'EEG067',
                                'EEG016', 'EEG017', 'EEG026', 'EEG057',
                                'EEG010', 'EEG015', 'EEG004', 'EEG005',
                                'EEG007', 'EEG008', 'EEG009'],
                'meg16_0045': [ 'EEG034', 'EEG058', 'EEG071', 'EEG066',
                                'EEG039', 'EEG047', 'EEG069', 'EEG042',
                                'EEG005', 'EEG011', 'EEG012', 'EEG021', 
                                'EEG022', 'EEG023', 'EEG010', 'EEG001', 
                                'EEG010', 'EEG068', 'EEG072', 'EEG059',
                                'EEG019', 'EEG030', 'EEG070', 'EEG001', 
                                'EEG004', 'EEG008', 'EEG014', 'EEG067',
                                'EEG073', 'EEG002', 'EEG003', 'EEG005', 
                                'EEG009', 'EEG010', 'EEG012', 'EEG013',
                                'EEG014', 'EEG020', 'EEG021', 'EEG022', 
                                'EEG023', 'EEG027', 'EEG031', 'EEG040', 
                                'EEG041', 'EEG043', 'EEG049', 'EEG050',
                                'EEG051', 'EEG052', 'EEG053', 'EEG054', 
                                'EEG065', 'EEG066', 'EEG067', 'EEG070',
                                'EEG039', 'EEG027', 'EEG024', 'EEG048', 
                                'MEG0231','EEG027', 'EEG038'],
                'meg16_0052': ['EEG067', 'EEG001', 'EEG005', 'EEG022',
                                'EEG024', 'EEG035', 'EEG046', 'EEG002',
                                'EEG008', 'EEG039', 'EEG014'],
                'meg16_0056': ['EEG069', 'EEG074', 'EEG002', 'EEG005',
                                'EEG004', 'EEG073', 'EEG008', 'EEG016',
                                'EEG026', 'EEG004', 'EEG010', 'EEG011', 
                                'EEG016', 'EEG017', 'EEG018', 'EEG019', 
                                'EEG021', 'EEG026'],
                'meg16_0069': ['EEG043', 'EEG057', 'EEG047', 'EEG046',
                                'EEG002', 'EEG004', 'EEG005', 'EEG006', 
                                'EEG007', 'EEG008', 'EEG010', 'MEG0522',
                                'MEG0912','MEG0943'],
                'meg16_0070': ['EEG019', 'EEG053', 'EEG072', 'EEG071',
                                'EEG002', 'EEG004', 'EEG005', 'EEG006',
                                'EEG007', 'EEG008', 'EEG043', 'EEG034',
                                'EEG015', 'EEG029', 'EEG039', 'EEG046', 
                                'EEG047', 'EEG010', 'EEG015', 'EEG068',
                                'EEG047'],
                'meg16_0072': ['EEG066', 'EEG070', 'EEG039', 'EEG069',
                                'EEG002', 'EEG003', 'EEG008', 'EEG039', 
                                'EEG041', 'EEG065', 'EEG070', 'EEG002', 
                                'EEG003', 'EEG008', 'EEG039', 'EEG041', 
                                'EEG065', 'EEG070', 'EEG039', 'EEG043',
                                'EEG054', 'EEG056', 'EEG068', 'EEG069'],
                'meg16_0073': ['EEG072', 'EEG071', 'EEG045', 'EEG070',
                                'EEG008', 'EEG055', 'EEG011', 'EEG004'],
                'meg16_0075': ['EEG073', 'EEG071', 'EEG068', 'EEG072',
                                'EEG002', 'EEG004', 'EEG008', 'EEG007',
                                'EEG017', 'EEG074', 'EEG009', 'EEG017', 
                                'EEG019', 'EEG020', 'EEG021', 'EEG030',
                                'EEG042', 'EEG001', 'EEG002', 'EEG004',
                                'EEG005', 'EEG008', 'EEG029', 'EEG039', 
                                'EEG042', 'EEG050', 'EEG051', 'EEG052',
                                'EEG066', 'EEG068', 'EEG069', 'EEG070', 
                                'EEG071', 'EEG072', 'EEG073', 'EEG074',
                                'EEG010', 'EEG011', 'EEG015', 'EEG016', 
                                'EEG018', 'EEG019', 'EEG020', 'EEG021', 
                                'EEG022', 'EEG025', 'EEG026', 'EEG027', 
                                'EEG028'],
                'meg16_0078': [ 'EEG027', 'EEG016', 'EEG074', 'EEG073',
                                'EEG002', 'EEG004', 'EEG005', 'EEG006',
                                'EEG007', 'EEG008', 'EEG016', 'EEG029',
                                'EEG049', 'MEG1421','MEG1431','EEG003',
                                'EEG015', 'EEG017', 'EEG026', 'EEG038', 
                                'EEG028', 'EEG060', 'EEG059', 'EEG039',
                                'EEG011', 'EEG019', 'EEG050', 'EEG009',
                                'EEG010', 'EEG030', 'EEG021', 'EEG040',
                                'EEG042', 'EEG036', 'EEG037', 'EEG051',
                                'EEG048', 'EEG058', 'EEG072', 'EEG071',
                                'EEG068'],
                'meg16_0082': [ 'EEG047', 'EEG002', 'EEG004', 'EEG005', 
                                'EEG006', 'EEG007', 'EEG008', 'EEG009', 
                                'EEG010', 'EEG011', 'EEG015', 'EEG016',
                                'EEG017', 'EEG019', 'EEG020', 'EEG021',
                                'EEG027', 'EEG029', 'EEG030', 'EEG037', 
                                'EEG038', 'EEG039', 'EEG012', 'EEG013',
                                'EEG014'],
                'meg16_0086': [ 'EEG034', 'EEG071', 'EEG070', 'EEG039', 
                                'EEG041', 'EEG015', 'EEG038', 'EEG025',
                                'EEG070', 'EEG052'],
                'meg16_0097': [ 'EEG013', 'EEG002', 'EEG004', 'EEG008',
                                'EEG051', 'EEG006', 'EEG019', 'EEG029',
                                'EEG033', 'EEG034', 'EEG039', 'EEG044', 
                                'EEG045', 'EEG046', 'EEG050', 'EEG068',
                                'EEG009', 'EEG045', 'EEG009', 'EEG045',
                                'EEG007', 'EEG010', 'EEG011', 'EEG001',],
                'meg16_0122': [ 'EEG067', 'MEG0912', 'EEG004','EEG047',
                                'EEG034', 'EEG045'],
                'meg16_0125': [ 'EEG035', 'EEG032', 'EEG069', 'EEG071', 
                                'EEG067', 'EEG049','MEG0441']}
#
#
lfreq=0.1
h_freq=45

n_components = .95  
method = 'fastica'
decim = 3  
n_max_eog = 2 #EOG061/EOG062
n_max_ecg = 1
tmin, tmax = -0.3, 0.6
stim_delay = 0.034 # delay in s
participants=list(range(0,19))
# for i in [4,13,14,16]:
#     participants.remove(i)
    

# for i in participants:
# for i in [0]:
for i in np.arange(14,len(list_all)):

    print('***********************Participant : ', i)
    meg = list_all[i]

    raw_fname = main_path + meg + 'block_LD_tsss_raw.fif'
    raw = mne.io.Raw(raw_fname, preload=True)#, preload=True

#***************************Indicating bad channels***************************#    
    # picks_fruit = mne.pick_types(raw.info, meg=True, eeg=True, eog=False,
    #     stim=False )
    
    # raw.plot_sensors(ch_type = 'eeg')
    # raw.plot(duration=1, n_channels=17, 
    #   title='{0} -Participant : {1}'.format(' (Raw)',meg[1:11]))
    
##    
##    
    # bad_channels[meg[1:11]] = raw.info['bads']
##
##    with open('bad_channels_LD.csv', 'a') as f:
##        f.write('{},{}\n'.format(meg[1:11], raw.info['bads']))
##        f.close()
#
##******************************* Filtering *******************************#    
    print('***********************Participant : ', i)

    raw.info['bads'] = bad_channels[meg[1:11]]
    raw.interpolate_bads(reset_bads = True , mode = 'accurate')
    raw.set_eeg_reference( ref_channels = 'average')
    
    
    picks= mne.pick_types(raw.info, meg=True, eeg=True, eog=False,
            stim=False )

    raw_notch = raw.copy().notch_filter(freqs=50 , picks = picks)


    raw_notch_BPF = raw_notch.copy().filter(l_freq=lfreq, 
                h_freq=h_freq, fir_design='firwin' , picks = picks)


    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    
    out_name= data_path + meg + 'block_LD_tsss_notch_BPF0.1_45_raw.fif'
    raw_notch_BPF.save(out_name, overwrite=True)
    
#******************************* EOG/ECG*******************************#    


    print('***********************Participant : ', i)

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
    
    # eog_inds = [0,40]


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
#ica.plot_components()
##
##ica.plot_properties(raw)
#ica.plot_properties(raw, picks=pick_t, psd_args={'fmax': 45.})
#ica.plot_sources(raw,picks=[0,1,31,29,2,19])

  

    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    
    out_name = data_path + meg + 'block_LD_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
    raw.save(out_name , overwrite=True)



#******************************* Epoching ******************************#    


    print('***********************Participant : ', i)
    



    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,
                              stim=False)
  
    
    
    events = mne.find_events(raw , stim_channel='STI101',
                                  min_duration=0.001 , shortest_event=1)
  
    event_id = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
                'pwordc': 6 , 'pworda': 7 , 'filler' : 9}
    color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 6: 'red' ,
            7 : 'blue'}
                
    category_code = np.arange(1,6)
    events[:,0] = events[:,0] + np.round( raw.info['sfreq']*stim_delay )
    for evcnt in np.arange(0,events.shape[0]-2):    
      if (events[evcnt,2] in category_code  and  events[evcnt+2,2]!=16384):
          events[evcnt,2] = 7777
      elif(events[evcnt,2] in np.array([6,7,9]) and events[evcnt+2,2]!=4096):
          events[evcnt,2] = 8888
          

    reject = dict(eeg=120e-6, grad=200e-12, mag=4e-12)#, eog=150e-6)

    epochs  = mne.Epochs(raw , events, event_id, tmin, tmax, 
            picks=picks , proj=True, baseline=(tmin, 0), reject=reject)


    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)
  
    out_name = data_path + meg + 'block_LD_epochs-epo.fif'     
    epochs.save(out_name , overwrite=True)

#******************************* Evoked ******************************#    


    print('***********************Participant : ', i)
  
    
#    epoch_fname = new_path + meg + 'block_LD_epochs-epo.fif'
#    epochs      = mne.read_epochs(epoch_fname , preload=True)#, preload=True
#    
#
#    raw_fname  = new_path + meg + 'block_LD_tsss_notch_BPF0.1_45_ICAeog_ecg_raw.fif'
#    raw        = mne.io.Raw(raw_fname , preload=True)#, preload=True
#   
#         
#    
#    events = mne.find_events(raw , stim_channel='STI101',
#                                   min_duration=0.001 , shortest_event=1)
#    
#        
#    event_id = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
#                'pwordc': 6, 'pworda': 7 ,'filler' :9}
#    
#    color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 6: 'red' ,
#             7: 'blue' , 9:'blue'}    
#    
#    events[:,0] = events[:,0] + np.round( raw.info['sfreq']*stim_delay )
#    
#    reject = dict(eeg=120e-6, grad=200e-12, mag=4e-12)#, eog=150e-6)
#
    picks = mne.pick_types(epochs.info, meg=True, eeg=True)
#
#
    evoked_visual   = epochs['visual'].average(picks=picks)
    evoked_hear     = epochs['hear'].average(picks=picks)
    evoked_hand     = epochs['hand'].average(picks=picks)
    evoked_neutral  = epochs['neutral'].average(picks=picks)
    evoked_emotional= epochs['emotional'].average(picks=picks)
    evoked_pwordc   = epochs['pwordc'].average(picks=picks)
    evoked_pworda   = epochs['pworda'].average(picks=picks)
    
    evoked_LD={'visual':evoked_visual , 'hear':evoked_hear,
            'hand':evoked_hand, 'neutral':evoked_neutral,
            'emotional':evoked_emotional, 'pwordc':evoked_pwordc,
            'pworda':evoked_pworda}


    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

#    
    out_name = data_path + meg + 'block_LD_evoked.npy'
    np.save(out_name, evoked_LD)
#
##******************************* Visualization ******************************#    
#
##data_path = '/imaging/sr05/SemNet/SemNetData/'
##for i in np.arange(2, len(list_all)):
##    meg = list_all[i]
## 
##    for file in os.listdir(data_path + meg):
##        if file.endswith('epo.fif') or file.endswith('evoked.npy'):
##            os.remove(data_path + meg + file)
#
#
##******************************* Visualization ******************************#    
#    

# #
# data_path = '/imaging/sr05/SemNet/SemNetData'
# pictures_path = '/home/sr05/Python/pictures/SD_task_pictures/'
# #
#list_all =  ['/meg16_0030/160216/',#0
#            '/meg16_0032/160218/', #1'
#            '/meg16_0034/160219/', #2
#            '/meg16_0035/160222/', #3
#            '/meg16_0042/160229/', #4
#            '/meg16_0045/160303/', #5
#            '/meg16_0052/160310/', #6
#            '/meg16_0056/160314/', #7
#            '/meg16_0069/160405/', #8 
#            '/meg16_0070/160407/', #9
#            '/meg16_0072/160408/', #10
#            '/meg16_0073/160411/', #11
#            '/meg16_0075/160411/', #12
#            '/meg16_0078/160414/', #13
#            '/meg16_0082/160418/', #14
#            '/meg16_0086/160422/', #15
#            '/meg16_0097/160512/', #16 
#            '/meg16_0122/160707/', #17
#            '/meg16_0125/160712/', #18
#            ]




#tmin, tmax = -0.3, 0.6
# #stim_delay = 0.034 # delay in s
# Nb_Ave = 0
# nb_ave_array = list()
# all_evokeds_LD_word = list()
# #
# #
# #
# for i in np.arange(0,len(list_all)):
#     print('Participant : ', i)
#     meg=list_all[i]
    
# #    all_evokeds_LD_word=list()


#     evoked_LD = data_path + meg + 'block_LD_evoked.npy'
#     read_evoked_LD = np.load( evoked_LD , allow_pickle='TRUE').item()

#     nb_ave = 0
#     evoked_LD_visual   = read_evoked_LD['visual']
#     nb_ave =  nb_ave  + evoked_LD_visual.nave

#     evoked_LD_hear     = read_evoked_LD['hear']
#     nb_ave =  nb_ave  + evoked_LD_hear.nave

#     evoked_LD_hand     = read_evoked_LD['hand']
    # nb_ave =  nb_ave  + evoked_LD_hand.nave

    # evoked_LD_neutral  = read_evoked_LD['neutral']
    # nb_ave =  nb_ave  + evoked_LD_neutral.nave

    # evoked_LD_emotional= read_evoked_LD['emotional']
    # nb_ave =  nb_ave  + evoked_LD_emotional.nave
    # Nb_Ave =  Nb_Ave  + nb_ave 
    # nb_ave_array.append(nb_ave )

 
    
    # all_evokeds_LD_word.append( evoked_LD_visual )
    # all_evokeds_LD_word.append( evoked_LD_hear) 
    # all_evokeds_LD_word.append( evoked_LD_hand )
    # all_evokeds_LD_word.append( evoked_LD_neutral )
    # all_evokeds_LD_word.append( evoked_LD_emotional )

     

    
#    
#    grand_average_word  = mne.grand_average(all_evokeds_word)
#    ts_args = dict(gfp=True)
#    topomap_args = dict(sensors=True)
#     
#    grand_average_word.plot_joint( times=[0.196 , 0.295, 0.514], picks = 'eeg', 
#             title='Participant : {0}{1}{2}'.format(meg[1:11],
#              ' - LD_Words', '-Nave: ' +str(nb_ave )), ts_args=ts_args,
#              topomap_args=topomap_args)
#             
#    plt.savefig(pictures_path + 'Participant-'+ meg[1:11]+ '-LD_Words')   




# #
# grand_average_word  = mne.grand_average(all_evokeds_LD_word)

# ts_args = dict(gfp=True)
# topomap_args = dict(sensors=True)
    
# grand_average_word.plot_joint( times=[0.105,0.130,0.190,0.320,0.406],  
#                               picks = 'grad',
#           title='Grand Average(Grad) - LD_Words- Nave: '+str(Nb_Ave ),ts_args=ts_args, 
#           topomap_args=topomap_args)

# grand_average_word.plot_joint( times=[0.099,0.130,0.186,0.219,0.320,0.445], 
#                               picks = 'mag',
#           title='Grand Average (Mag) - LD_Words- Nave: '+str(Nb_Ave ),ts_args=ts_args, 
#           topomap_args=topomap_args)

# grand_average_word.plot_joint( times=[0.086,0.130,0.217,0.257,0.328,0.437,0.490],  
#                               picks = 'eeg',
#           title='Grand Average (EEG) - LD_Words- Nave: '+str(Nb_Ave ),ts_args=ts_args, 
#           topomap_args=topomap_args)
# plt.savefig(pictures_path + 'Grand Average -LD_Words')   

#
    
#image_list = []
#
#for image in os.listdir(pictures_path)[1:]:
#    if image.endswith('.png'):
#        image_list.append(Image.open(pictures_path + image).convert('RGB'))
#img = Image.open(pictures_path + 'Grand Average -LD_Words.png').convert('RGB')
#img.save(pictures_path + 'all_participants.pdf', save_all=True, append_images=image_list)




#main_path = '/imaging/rf02/Semnet/'	# where subdirs for MEG data are
#data_path = '/imaging/sr05/SemNet/SemNetData'
##


#image_list = []
#
#for image in os.listdir(pictures_path)[1:]:
#    if image.endswith('.png'):
#        image_list.append(Image.open(pictures_path + image).convert('RGB'))
#img = Image.open(pictures_path + 'Grand Average -LD_Words.png').convert('RGB')
#img.save(pictures_path + 'all_participants.pdf', save_all=True, append_images=image_list)




#image_list = []
#path_list = []
#
#for file in os.listdir(pictures_path):
#    if file.endswith('.png'):
#        path_list.append(file)
#
#path_list.sort()
#
#for image in path_list[1:]:
#    image_list.append(Image.open(pictures_path + image).convert('RGB'))
#
#img = Image.open(pictures_path + path_list[0]).convert('RGB')
#img.save(pictures_path + 'all_participants.pdf', save_all=True, append_images=image_list)
