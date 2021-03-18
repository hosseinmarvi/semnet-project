# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:47:56 2019

@author: Setare
"""
#****************************************************************************#
#                                  Initiating                                #
#****************************************************************************#

import os
import mne
import numpy as np
from mne.datasets import sample
from os import path as op
from mne.preprocessing import maxwell_filter
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs


#****************************************************************************#
#                                 Loading Data                               #
#****************************************************************************#
data_path = '/megdata/cbu/semnet/'#mne.datasets.sample.data_path()
raw_fname = data_path + '/meg16_0122/160707/block_odour_raw.fif'
#data_path = mne.datasets.sample.data_path()
#raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
#raw_fnamw = os.path.join(data_path, 'MEG', 'sample','sample_audvis_raw.fif')
ctc_fname = data_path + '/SSS/ct_sparse_mgh.fif'
fine_cal_fname = data_path + '/SSS/sss_cal_mgh.dat'
proj_fname = data_path + '/MEG/sample/sample_audvis_eog_proj.fif'
raw = mne.io.read_raw_fif(raw_fname)
raw.info['bads'] = ['MEG 2443', 'EEG 053', 'MEG 1032', 'MEG 2313']  # set bads

##..........................................................................##
##                               Ploting Data                               ##
raw.plot_psd(fmax=100)
raw.plot(duration=5, n_channels=30)


#****************************************************************************#
#                                Preprocessing                               #
#****************************************************************************#


##..........................................................................##
##                              Maxwell Filter                              ##

# Here we don't use tSSS (set st_duration) because MGH data is very clean
#raw_sss = maxwell_filter(raw, cross_talk=ctc_fname, calibration=fine_cal_fname)
raw_sss = maxwell_filter(raw, calibration=None, cross_talk=None)


raw_sss.plot_psd(fmax=100)

##..........................................................................##
##                              Band-pass Filter                            ##

raw_sss.filter(None, 40., fir_design='firwin')
raw_sss.plot_psd(fmax=100)


raw_sss.filter(1., None, fir_design='firwin')
raw_sss.plot_psd(fmax=100)

##..........................................................................##
##                                Down Sampling                             ##

raw_sss.resample(100, npad="auto")  # set sampling frequency to 100Hz
raw_sss.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=True)


##..........................................................................##
##                            ICA Artifact Removal                          ##

picks_meg = mne.pick_types(raw_sss.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
n_components = 25  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time
random_state = 23

ica = ICA(n_components=n_components, method=method, random_state=random_state, 
          max_iter=200)
print(ica)

reject = dict(grad=4000e-13, # T / m (gradiometers)
              mag=4e-12) # T (magnetometers)
ica.fit(raw_sss, picks=picks_meg, decim=decim, reject=reject)
print(ica)
raw_sss.plot_psd(fmax=100)

eog_average = create_eog_epochs(raw_sss, reject=dict(mag=5e-12, grad=4000e-13),
                                picks=picks_meg).average()

n_max_eog = 1  # here we bet on finding the vertical EOG components
eog_epochs = create_eog_epochs(raw_sss, reject=reject)  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
print(ica)
raw_sss.plot_psd(fmax=100)


#ica.plot_scores(scores, exclude=eog_inds)  
#ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course

ica.apply(raw_sss, exclude=eog_inds )
print(ica)
raw_sss.plot_psd(fmax=100)


ecg_epochs = create_ecg_epochs(raw_sss, tmin=-.5, tmax=.5)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')

ica.apply(raw_sss, exclude=ecg_inds )
raw_sss.plot_psd(fmax=100)


ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': 35.})



#****************************************************************************#
#                                     ERP                                    #
#****************************************************************************#
tmin, tmax = -0.5, 0.7
event_id = {'Auditory/Left': 1}
events = mne.find_events(raw, 'STI 014')
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                       include=[], exclude='bads')
for r, kind in zip((raw, raw_sss), ('Raw data', 'Maxwell filtered data')):
    epochs = mne.Epochs(r, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=dict(eog=150e-6),
                        preload=False)
    evoked = epochs.average()
    evoked.plot(window_title=kind, ylim=dict(grad=(-200, 250),
                                             mag=(-600, 700)))


