"""
Created on Tue Dec 17 10:47:56 2019

@author: Setare
"""
"""
#****************************************************************************#
#                                Initializing                                #
#****************************************************************************#
"""
import os
import mne
import numpy as np
from mne.datasets import sample
from os import path as op
from mne.preprocessing import maxwell_filter
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
import sys
#sys.path.insert(1,'/imaging/local/software/mne_python/latest_v0.9full')

"""
#****************************************************************************#
#                                 Loading Data                               #
#****************************************************************************#
"""
#data_path = sample.data_path()
#raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

data_path = '/megdata/cbu/semnet/'#mne.datasets.sample.data_path()
raw_fname = data_path + '/meg16_0122/160707/block_odour_raw.fif'

data_path_tsss= '/imaging/rf02/Semnet/'
raw_fname_tsss = data_path_tsss + '/meg16_0122/160707/block_odour_tsss_raw.fif'


raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw_tsss = mne.io.read_raw_fif(raw_fname_tsss, preload=True)
raw.ch_names

#ctc_fname = data_path + '/SSS/ct_sparse_mgh.fif'
#fine_cal_fname = data_path + '/SSS/sss_cal_mgh.dat'
#proj_fname = data_path + '/MEG/sample/sample_audvis_eog_proj.fif'
#raw.info['bads'] = ['MEG 2443', 'EEG 053', 'MEG 1032', 'MEG 2313']  # set bads

##..........................................................................##
##                               Ploting Data                               ##
#raw.plot_psd(fmax=100)
raw_intepolate = raw.copy().interpolate_bads(mode='accurate')
raw.plot_psd(tmin=1.0, tmax=None, fmin=1, fmax=80)
raw_tsss.plot_psd(tmin=1.0, tmax=None, fmin=1, fmax=80)

raw.plot(duration=1, n_channels=3)
raw_intepolate.plot(duration=1, n_channels=3)
raw_tsss.plot(duration=1, n_channels=3)
#raw.plot( duration=10, start=50, n_channels=5, title='MEG', show=True)

"""
#****************************************************************************#
#                                Preprocessing                               #
#****************************************************************************#
"""

#Warning
#Automatic bad channel detection is not currently implemented. It is critical 
#to mark bad channels before running Maxwell filtering, so data should be 
#inspected and marked accordingly prior to running this algorithm.
##..........................................................................##
##                              Maxwell Filter                              ##

#raw_sss = maxwell_filter(raw, cross_talk=ctc_fname, calibration=fine_cal_fname)
raw_sss = maxwell_filter(raw, calibration=None, cross_talk=None)

#raw.plot_psd(fmax=100)
raw_sss.plot_psd(fmax=100)
raw_tsss.plot_psd(fmax=100)

##..........................................................................##
##                              Band-pass Filter                            ##

#picks = mne.pick_types(raw_sss.info, meg=True, eeg=True, eog=True,
#                       stim=False, exclude='bads')
raw_sss_notch = raw_sss.copy().notch_filter(freqs=50)
raw_sss_notch.plot_psd(fmax=100)
raw_sss.plot_psd(fmax=100)

raw_sss_notch_BPF=raw_sss_notch.copy().filter(1, 48., fir_design='firwin')
#raw_sss.filter(None, 48., fir_design='firwin', picks=picks)
raw_sss_notch_BPF.plot_psd(fmax=100)
raw_sss_BPF=raw_sss.copy().filter(1, 48., fir_design='firwin')
raw_sss_BPF.plot_psd(fmax=100)
raw.interpolate_bads()

#raw_sss.filter(1., None, fir_design='firwin')
#raw_sss.filter(1., None, fir_design='firwin', picks=picks)
#raw_sss.plot_psd(fmax=100)
##..........................................................................##
##                                Down Sampling                             ##

#raw_sss.resample(100, npad="auto")  # set sampling frequency to 100Hz
#raw_sss.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=True)


##..........................................................................##
##                            ICA Artifact Removal                          ##
n_components = .99  
method = 'fastica'
decim = 3  

picks_all= mne.pick_types(raw_sss.info, meg=True, eeg=True, eog=True,
                          stim=False)
ica = ICA(n_components=n_components, method=method)
reject = dict(mag=4e-12, grad=4000e-13)
ica.fit(raw_sss, picks=picks_all, decim=decim, reject=reject)
print(ica)

ica.plot_components()  
#ica.plot_properties(raw_sss, picks=0)


##................................EOG Removal............................... ##

title = 'Sources related to %s artifacts (red)'
n_max_ecg, n_max_eog = 1, 2 #EOG061/EOG062/ECG063

#eog_average = create_eog_epochs(raw_sss, reject=reject, 
#                                picks=picks_all).average()
eog_epochs = create_eog_epochs(raw_sss, reject=reject)  
eog_inds, scores = ica.find_bads_eog(eog_epochs)
eog_inds = eog_inds[:n_max_eog]
#eog_inds1, scores1 = ica.find_bads_eog(raw_sss)
ica.exclude += eog_inds

ica.plot_properties(raw_sss, picks=eog_inds, psd_args={'fmax': 45.})
ica.plot_scores(scores, exclude=eog_inds, title=title % 'eog', labels='eog')

show_picks = np.abs(scores[1]).argsort()[::-1][:5]
ica.plot_sources(raw_sss, show_picks, exclude=eog_inds, title=title % 'eog')
#ica.plot_components(eog_inds, title=title % 'EOG', colorbar=True)
#ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.})


ica.apply(inst=raw_sss, exclude=eog_inds)
print(ica)



##................................ECG Removal............................... ##

ecg_epochs = create_ecg_epochs(raw_sss, reject=reject, picks=picks_all)
ecg_inds, scores_ecg = ica.find_bads_ecg(ecg_epochs, method='ctps')
ecg_inds = ecg_inds[:n_max_ecg]
ica.exclude += ecg_inds

#ecg_inds1, scores1 = ica.find_bads_ecg(raw_sss, method='ctps')
ica.plot_components(ecg_inds, title=title % 'ECG', colorbar=True)
ica.plot_scores(scores_ecg , exclude=ecg_inds, title=title % 'ecg', 
                labels='ecg')

show_picks = np.abs(scores_ecg ).argsort()[::-1][:5]
ica.plot_sources(raw_sss, show_picks, exclude=ecg_inds, title=title % 'ecg')
ica.plot_components(ecg_inds, title=title % 'ecg', colorbar=True)
ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': 35.})


ica.apply(raw_sss, exclude=ecg_inds )
print(ica)
ica.plot_sources(raw_sss)



#ecg_evoked = create_ecg_epochs(raw_sss, reject=reject, picks=picks_all)
#
#ica.plot_sources(ecg_evoked, exclude=ecg_inds)  # plot ECG sources + selection
#ica.plot_overlay(ecg_evoked.average(), exclude=ecg_inds)  # plot ECG cleaning
#
#eog_evoked = create_eog_epochs(raw_sss, picks=picks_all)
#ica.plot_sources(eog_evoked, exclude=eog_inds)  # plot EOG sources + selection
#ica.plot_overlay(eog_evoked, exclude=eog_inds)  # plot EOG cleaning
#
## check the amplitudes do not change
#ica.plot_overlay(raw_sss) 

"""
#****************************************************************************#
#                                  Processing                                #
#****************************************************************************#
"""
#************************************ ERP ************************************#
tmin, tmax = -0.3, 0.6
stim_delay = 0.034 # delay in s

include = []
exclude = []
picks = mne.pick_types(raw_sss.info, meg=True, eeg=True, eog=False,
                stim=False, include=include, exclude=exclude)
events = mne.find_events(raw_sss, stim_channel='STI101',min_duration=0.001,
                         shortest_event=1)    

    
event_id = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
            'pwordc': 6, 'target': 8}
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 6: 'red' ,
         8: 'blue'}    

events[:,0] = events[:,0]+np.round( raw.info['sfreq']*stim_delay )

reject = dict(eeg=120e-6, grad=200e-12, mag=4e-12)#, eog=150e-6)
event_ids = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4,
                 'emotional': 5,'pwordc': 6}
epochs = mne.Epochs(raw_sss, events, event_id, tmin, tmax, picks=picks, 
                        proj=True, baseline=(tmin, 0), reject=reject)

##................................Visualizing.............................. ##

##................................ERP/Visual............................... ##
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_visual = epochs['visual'].average(picks=picks)
evoked_visual.plot()
for i in evoked_visual.ch_names[::-1][:5]:
    evoked_visual.plot(picks=i)
    
#evoked_visual.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='mag')
# mne.viz.plot_evoked_topo(evoked_visual,title='Visual')    
  
##.................................ERP/Hear................................ ##
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_hear = epochs['hear'].average(picks=picks)
evoked_hear.plot()
for i in evoked_hear.ch_names[::-1][:5]:
    evoked_hear.plot(picks=i) 

evoked_hear.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='mag')        
mne.viz.plot_evoked_topo(evoked_hear,title='Hear')    

##.................................ERP/Hand................................ ##
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_hand = epochs['hand'].average(picks=picks)
evoked_hand.plot()
for i in evoked_hand.ch_names[::-1][:5]:
    evoked_hand.plot(picks=i)  

evoked_hand.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='mag')
mne.viz.plot_evoked_topo(evoked_hand,title='Hand')    


##................................ERP/Neutral.............................. ##
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_neutral = epochs['neutral'].average(picks=picks)
evoked_neutral.plot()
for i in evoked_neutral.ch_names[::-1][:5]:
    evoked_neutral.plot(picks=i)

evoked_neutral.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='mag')
mne.viz.plot_evoked_topo(evoked_neutral,title='Neutral')    
    
##...............................ERP/Emotional............................. ##
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_emotional = epochs['emotional'].average(picks=picks)
evoked_emotional.plot()
for i in evoked_emotional.ch_names[::-1][:5]:
    evoked_emotional.plot(picks=i)  

evoked_emotional.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='mag')
mne.viz.plot_evoked_topo(evoked_emotional,title='Emotional')    

##................................ERP/Pwords............................... ##
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_pwordc = epochs['pwordc'].average(picks=picks)
evoked_pwordc.plot()
for i in evoked_pwordc.ch_names[::-1][:5]:
    evoked_pwordc.plot(picks=i)  

evoked_pwordc.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='mag')
mne.viz.plot_evoked_topo(evoked_pwordc,title='Pwords')    

##................................ERP/Target............................... ##
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_target = epochs['target'].average(picks=picks)
evoked_target.plot()
for i in evoked_target.ch_names[::-1][:5]:
    evoked_target.plot(picks=i)  

evoked_target.plot_topomap(times=[-0.2,0,0.2,0.4,0.6],ch_type='mag')
mne.viz.plot_evoked_topo(evoked_target,title='Target')    

##.............................All condistions............................. ##


conditions = ["visual","hear","hand","neutral","emotional","pwordc","target"]
evoked_dict = dict()
evoked_dict={'visual':evoked_visual,'hear':evoked_hear,'hand':evoked_hand,
             'neutral':evoked_neutral,'emotional':evoked_emotional,
             'pwordc':evoked_pwordc, 'target':evoked_target}
print(evoked_dict)
All_evoked=[evoked_visual , evoked_hear , evoked_hand , evoked_neutral , 
            evoked_emotional , evoked_pwordc , evoked_target]


mne.combine_evoked([evoked_visual , evoked_hear)
aud_epochs.plot_image(picks=['MEG 1332', 'EEG 021'])
mne.viz.plot_compare_evokeds(dict(auditory=aud_evoked, visual=vis_evoked),
                             legend='upper left', show_sensors='upper right')
aud_evoked.plot_joint(picks='eeg')
aud_evoked.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2], ch_type='eeg')


#............................................................................#
raw.plot_sensors(ch_type='eeg', show_names=True)

mne.viz.plot_compare_evokeds(evoked_dict)

pick = evoked_dict["visual"].ch_names.index('EEG001')
mne.viz.plot_compare_evokeds(evoked_dict,picks =pick )
    


#ts_args = dict(gfp=True)
#topomap_args = dict(sensors=False)
#evoked_visual.plot_joint( times=[0,0.4],
#                        ts_args=ts_args, topomap_args=topomap_args)

