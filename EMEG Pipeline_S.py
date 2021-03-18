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

data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
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
raw_sss = maxwell_filter(raw, cross_talk=ctc_fname, calibration=fine_cal_fname)

raw.plot_psd(fmax=100)
raw_sss.plot_psd(fmax=100)

##..........................................................................##
##                              Band-pass Filter                            ##


tmin, tmax = 0, 20  # use the first 20s of data

# Setup for reading the raw data (save memory by cropping the raw data
# before loading it)
raw_sss.crop(tmin, tmax).load_data()
raw_sss.info['bads'] = ['MEG 2443', 'EEG 053']  # bads + 2 more

n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2

# Pick a subset of channels (here for speed reasons)
selection = mne.read_selection('Left-temporal')
picks = mne.pick_types(raw_sss.info, meg=True, eeg=True, eog=False,
                       stim=False, exclude='bads', selection=selection)
raw_sss.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=False)

# Low-pass Filter

raw_sss.filter(None, 58., fir_design='firwin')
raw_sss.plot_psd(fmax=100, area_mode='range', tmax=10.0, average=False)

#raw_sss.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=False)
# High-pass Filter
#f_l=0.1
raw_sss.filter(1., None, fir_design='firwin')
raw_sss.plot_psd(fmax=100, area_mode='range', tmax=10.0,  average=False)
#raw_sss.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=False)

##..........................................................................##
##                                Down Sampling                             ##

raw_sss.resample(100, npad="auto")  # set sampling frequency to 100Hz
raw_sss.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=True)


##..........................................................................##
##                            ICA Artifact Removal                          ##

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw_sss)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)



picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
##
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40, n_jobs=2)  # 1Hz high pass is often helpful for fitting ICA

picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

n_components = 25  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23

ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)
reject = dict(mag=5e-12, grad=4000e-13)
ica.fit(raw, picks=picks_meg, decim=decim, reject=reject)
print(ica)

ica.plot_components()  # can you spot some potential bad guys?

# first, component 0:
ica.plot_properties(raw, picks=0)

ica.plot_properties(raw, picks=0, psd_args={'fmax': 35.})

ica.plot_properties(raw, picks=[1, 2], psd_args={'fmax': 35.})

# uncomment the code below to test the inteactive mode of plot_components:
# ica.plot_components(picks=range(10), inst=raw)
eog_average = create_eog_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                picks=picks_meg).average()

# We simplify things by setting the maximum number of components to reject
n_max_eog = 1  # here we bet on finding the vertical EOG components
eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).

ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course

ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})
print(ica.labels_)

ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
# red -> before, black -> after. Yes! We remove quite a lot!

# to definitely register this component as a bad one to be removed
# there is the ``ica.exclude`` attribute, a simple Python list
ica.exclude.extend(eog_inds)

# from now on the ICA will reject this component even if no exclude
# parameter is passed, and this information will be stored to disk
# on saving

# uncomment this for reading and writing
# ica.save('my-ica.fif')
# ica = read_ica('my-ica.fif')
ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': 35.})



#************************************ ERP ************************************#
tmin, tmax = -0.2, 0.5
event_id = {'Auditory/Left': 1}
events = mne.find_events(raw, 'STI 014')
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       include=[], exclude='bads')
for r, kind in zip((raw, raw_sss), ('Raw data', 'Maxwell filtered data')):
    epochs = mne.Epochs(r, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=dict(eog=150e-6),
                        preload=False)
    evoked = epochs.average()
    evoked.plot(window_title=kind, ylim=dict(grad=(-200, 250),
                                             mag=(-600, 700)))


