#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:09:46 2020

@author: sr05
"""

import numpy as np
import mne

from mne.minimum_norm import ( make_inverse_operator, write_inverse_operator, read_inverse_operator)
import os
import matplotlib.pyplot as plt

list_all =  ['/MRI_meg16_0030/160216/',#0
            '/MRI_meg16_0032/160218/', #1
            '/MRI_meg16_0034/160219/', #2
            '/MRI_meg16_0035/160222/', #3
            '/MRI_meg16_0042/160229/', #4
            '/MRI_meg16_0045/160303/', #5
            '/MRI_meg16_0052/160310/', #6
            '/MRI_meg16_0056/160314/', #7
            '/MRI_meg16_0069/160405/', #8 
            '/MRI_meg16_0070/160407/', #9
            '/MRI_meg16_0072/160408/', #10
            '/MRI_meg16_0073/160411/', #11
            '/MRI_meg16_0075/160411/', #12
            '/MRI_meg16_0078/160414/', #13
            '/MRI_meg16_0082/160418/', #14
            '/MRI_meg16_0086/160422/', #15
            '/MRI_meg16_0097/160512/', #16 
            '/MRI_meg16_0122/160707/', #17
            '/MRI_meg16_0125/160712/', #18
            ]



  

     
# data_path = '/imaging/rf02/Semnet/' # root directory for your MEG data
new_path = '/imaging/sr05/SemNet/SemNetData/'
bem_ico = 4

bem_conductivity_1 = (0.3,)  # for single layer
bem_conductivity_3 = (0.3, 0.006, 0.3)  # for three layers

for i in np.arange(1,len(list_all)):
    print('Participant : ', i)
    meg = '/'+list_all[i][5:]
    mri = list_all[i][1:15]
    
    # fwd_fname = data_path + meg + 'ico5_forward_5-3L-EMEG-fwd.fif'
    # fwd = mne.read_forward_solution(fwd_fname)
    # leadfield = fwd['sol']['data']
    # print("Leadfield size : %d x %d" % leadfield.shape)
    
    # Freesurfer and MNE environment variables
    filename = "/imaging/local/software/mne_python/set_MNE_2.7.3_FS_6.0.0_environ.py"
    # for Python 3 instead of execfile
    exec(compile(open(filename, "rb").read(), filename, 'exec'))
     
     
    # where MRIs are
    # os.environ['SUBJECTS_DIR'] = config.subjects_dir
    
    # my_subject =  data_path + mri
    # mne.bem.make_watershed_bem(subject = my_subject)
    # mne.bem.make_watershed_bem(subject = mri , subjects_dir = data_path )
    mne.bem.make_watershed_bem(subject = mri , subjects_dir = new_path , overwrite= True)

    # mne.bem.make_watershed_bem(subject = mri + '/bem/', subjects_dir = data_path)
    # mne.bem.make_watershed_bem(subject = mri + '/mri/', subjects_dir = data_path)
  model = mne.make_bem_model(subject=subject, ico=C.bem_ico,
                                conductivity=conductivity_1,
                                subjects_dir=subjects_dir)
    
    bem = mne.make_bem_solution(model)

    bem_fname = op.join(C.bem_path, subject, 'bem', subject + '_MEG-bem.fif')

    print "###\nWriting BEM solution to " + bem_fname + "\n###"
    mne.bem.write_bem_solution(bem_fname, bem)

    ### three-shell BEM for EEG+MEG
    model = mne.make_bem_model(subject=subject, ico=C.bem_ico,
                                conductivity=conductivity_3,
                                subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    bem_fname = op.join(C.bem_path, subject, 'bem', subject + '_EEGMEG-bem.fif')

    print "###\nWriting BEM solution to " + bem_fname + "\n###"
    mne.bem.write_bem_solution(bem_fname, bem)
    
    
    