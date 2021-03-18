#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:45:41 2020

@author: sr05
"""


import mne
import numpy as np
import sn_config as C
from mne.minimum_norm import read_inverse_operator 


# path to filtered raw data
main_path = C.main_path
data_path = C.data_path
# subjects' directories
subjects =  C.subjects_mri
 
for i in np.arange(0, len(subjects)):
    print('participant : ' , i , ' / EEG + MEG')
    subject_from = subjects[i]
    meg = subjects[i][5:]
    # fname_morph = C.fname_STC(C, 'SensitivityMaps', subject, 'SensMap_' + modality + '_' + metric + '_mph')

     ##................................EEG + MEG............................##

    inv_fname_EMEG_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_EMEG_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'


    inv_op_SD = read_inverse_operator(inv_fname_EMEG_SD)   
    inv_op_LD = read_inverse_operator(inv_fname_EMEG_LD)    
    
    stc_fname_SD_words  = data_path + meg + 'block_SD_words_EMEG'
    stc_fname_LD_words  = data_path + meg + 'block_LD_words_EMEG'
    # stc_fname_fruit_visual    = data_path + meg + 'block_fruit_visual_EMEG'
    # stc_fname_fruit_hear      = data_path + meg + 'block_fruit_hear_EMEG'
    # stc_fname_fruit_hand      = data_path + meg + 'block_fruit_hand_EMEG'
    # stc_fname_fruit_emotional = data_path + meg + 'block_fruit_emotional_EMEG'
    # stc_fname_fruit_neutral   = data_path + meg + 'block_fruit_neutral_EMEG'
    # stc_fname_fruit_pwordc    = data_path + meg + 'block_fruit_pwordc_EMEG'
    # stc_fname_fruit_target    = data_path + meg + 'block_fruit_target_EMEG'

    # stc_fname_odour_visual    = data_path + meg + 'block_odour_visual_EMEG'
    # stc_fname_odour_hear      = data_path + meg + 'block_odour_hear_EMEG'
    # stc_fname_odour_hand      = data_path + meg + 'block_odour_hand_EMEG'
    # stc_fname_odour_emotional = data_path + meg + 'block_odour_emotional_EMEG'
    # stc_fname_odour_neutral   = data_path + meg + 'block_odour_neutral_EMEG'
    # stc_fname_odour_pwordc    = data_path + meg + 'block_odour_pwordc_EMEG'
    # stc_fname_odour_target    = data_path + meg + 'block_odour_target_EMEG'
    
    # stc_fname_milk_visual     = data_path + meg + 'block_milk_visual_EMEG'
    # stc_fname_milk_hear       = data_path + meg + 'block_milk_hear_EMEG'
    # stc_fname_milk_hand       = data_path + meg + 'block_milk_hand_EMEG'
    # stc_fname_milk_emotional  = data_path + meg + 'block_milk_emotional_EMEG'
    # stc_fname_milk_neutral    = data_path + meg + 'block_milk_neutral_EMEG'
    # stc_fname_milk_pwordc     = data_path + meg + 'block_milk_pwordc_EMEG'
    # stc_fname_milk_target     = data_path + meg + 'block_milk_target_EMEG'
    
    # stc_fname_LD_visual       = data_path + meg + 'block_LD_visual_EMEG'
    # stc_fname_LD_hear         = data_path + meg + 'block_LD_hear_EMEG'
    # stc_fname_LD_hand         = data_path + meg + 'block_LD_hand_EMEG'
    # stc_fname_LD_emotional    = data_path + meg + 'block_LD_emotional_EMEG'
    # stc_fname_LD_neutral      = data_path + meg + 'block_LD_neutral_EMEG'
    # stc_fname_LD_pwordc       = data_path + meg + 'block_LD_pwordc_EMEG'
    # stc_fname_LD_pworda       = data_path + meg + 'block_LD_pworda_EMEG'
    # stc_fname_LD_filler       = data_path + meg + 'block_LD_filler_EMEG'

    stc_SD_words = mne.read_source_estimate(stc_fname_SD_words)
    stc_LD_words = mne.read_source_estimate(stc_fname_LD_words)
    
    # stc_fruit_visual   = mne.read_source_estimate(stc_fname_fruit_visual)
    # stc_fruit_hear     = mne.read_source_estimate(stc_fname_fruit_hear)
    # stc_fruit_hand     = mne.read_source_estimate(stc_fname_fruit_hand)
    # stc_fruit_emotional= mne.read_source_estimate(stc_fname_fruit_emotional)
    # stc_fruit_neutral  = mne.read_source_estimate(stc_fname_fruit_neutral)
    # stc_fruit_pwordc   = mne.read_source_estimate(stc_fname_fruit_pwordc)
    # stc_fruit_target   = mne.read_source_estimate(stc_fname_fruit_target)

    # stc_odour_visual   = mne.read_source_estimate(stc_fname_odour_visual)
    # stc_odour_hear     = mne.read_source_estimate(stc_fname_odour_hear)
    # stc_odour_hand     = mne.read_source_estimate(stc_fname_odour_hand)
    # stc_odour_emotional= mne.read_source_estimate(stc_fname_odour_emotional)
    # stc_odour_neutral  = mne.read_source_estimate(stc_fname_odour_neutral)
    # stc_odour_pwordc   = mne.read_source_estimate(stc_fname_odour_pwordc)
    # stc_odour_target   = mne.read_source_estimate(stc_fname_odour_target)


    # stc_milk_visual   = mne.read_source_estimate(stc_fname_milk_visual)
    # stc_milk_hear     = mne.read_source_estimate(stc_fname_milk_hear)
    # stc_milk_hand     = mne.read_source_estimate(stc_fname_milk_hand)
    # stc_milk_emotional= mne.read_source_estimate(stc_fname_milk_emotional)
    # stc_milk_neutral  = mne.read_source_estimate(stc_fname_milk_neutral)
    # stc_milk_pwordc   = mne.read_source_estimate(stc_fname_milk_pwordc)
    # stc_milk_target   = mne.read_source_estimate(stc_fname_milk_target)

    # stc_LD_visual   = mne.read_source_estimate(stc_fname_LD_visual)
    # stc_LD_hear     = mne.read_source_estimate(stc_fname_LD_hear)
    # stc_LD_hand     = mne.read_source_estimate(stc_fname_LD_hand)
    # stc_LD_emotional= mne.read_source_estimate(stc_fname_LD_emotional)
    # stc_LD_neutral  = mne.read_source_estimate(stc_fname_LD_neutral)
    # stc_LD_pwordc   = mne.read_source_estimate(stc_fname_LD_pwordc)
    # stc_LD_pworda   = mne.read_source_estimate(stc_fname_LD_pworda)
    # stc_LD_filler   = mne.read_source_estimate(stc_fname_LD_filler )

    # # setup source morph
    morph_SD_words = mne.compute_source_morph( src=inv_op_SD['src'], 
                  subject_from = stc_SD_words.subject, 
                  subject_to = C.subject_to , spacing = C.spacing_morph, 
                  subjects_dir = C.data_path)
    morph_LD_words = mne.compute_source_morph( src=inv_op_LD['src'], 
                  subject_from = stc_LD_words.subject, 
                  subject_to = C.subject_to , spacing = C.spacing_morph, 
                  subjects_dir = C.data_path)
    
    
    # morph_fruit_visual = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_fruit_visual.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_fruit_hear = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_fruit_hear.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_fruit_hand = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_fruit_hand.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_fruit_emotional = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_fruit_emotional.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_fruit_neutral = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_fruit_neutral.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_fruit_pwordc = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_fruit_pwordc.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_fruit_target = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_fruit_target.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)    

    # morph_odour_visual = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_odour_visual.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_odour_hear = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_odour_hear.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_odour_hand = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_odour_hand.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_odour_emotional = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_odour_emotional.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_odour_neutral = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_odour_neutral.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_odour_pwordc = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_odour_pwordc.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_odour_target = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_odour_target.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path) 


    # morph_milk_visual = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_milk_visual.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_milk_hear = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_milk_hear.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_milk_hand = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_milk_hand.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_milk_emotional = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_milk_emotional.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_milk_neutral = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_milk_neutral.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_milk_pwordc = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_milk_pwordc.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_milk_target = mne.compute_source_morph( src=inv_op_SD['src'], 
    #               subject_from = stc_milk_target.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path) 

    # morph_LD_visual = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_visual.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_LD_hear = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_hear.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_LD_hand = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_hand.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_LD_emotional = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_emotional.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_LD_neutral = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_neutral.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_LD_pwordc = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_pwordc.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # morph_LD_pworda = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_pworda.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path) 
    # morph_LD_filler = mne.compute_source_morph( src=inv_op_LD['src'], 
    #               subject_from = stc_LD_filler.subject, 
    #               subject_to = C.subject_to , spacing = C.spacing_morph, 
    #               subjects_dir = C.data_path)
    # # morph data
    stc_SD_words_fsaverage = morph_SD_words.apply(stc_SD_words)
    stc_LD_words_fsaverage = morph_LD_words.apply(stc_LD_words)
    
    # stc_fruit_visual_fsaverage    = morph_fruit_visual.apply(stc_fruit_visual)
    # stc_fruit_hear_fsaverage      = morph_fruit_hear.apply(stc_fruit_hear)
    # stc_fruit_hand_fsaverage      = morph_fruit_hand.apply(stc_fruit_hand)
    # stc_fruit_emotional_fsaverage = morph_fruit_emotional.apply(stc_fruit_emotional)
    # stc_fruit_neutral_fsaverage   = morph_fruit_neutral.apply(stc_fruit_neutral)
    # stc_fruit_pwordc_fsaverage    = morph_fruit_pwordc.apply(stc_fruit_pwordc)
    # stc_fruit_target_fsaverage    = morph_fruit_target.apply(stc_fruit_target)

    # stc_odour_visual_fsaverage    = morph_odour_visual.apply(stc_odour_visual)
    # stc_odour_hear_fsaverage      = morph_odour_hear.apply(stc_odour_hear)
    # stc_odour_hand_fsaverage      = morph_odour_hand.apply(stc_odour_hand)
    # stc_odour_emotional_fsaverage = morph_odour_emotional.apply(stc_odour_emotional)
    # stc_odour_neutral_fsaverage   = morph_odour_neutral.apply(stc_odour_neutral)
    # stc_odour_pwordc_fsaverage    = morph_odour_pwordc.apply(stc_odour_pwordc)
    # stc_odour_target_fsaverage    = morph_odour_target.apply(stc_odour_target)


    # stc_milk_visual_fsaverage    = morph_milk_visual.apply(stc_milk_visual)
    # stc_milk_hear_fsaverage      = morph_milk_hear.apply(stc_milk_hear)
    # stc_milk_hand_fsaverage      = morph_milk_hand.apply(stc_milk_hand)
    # stc_milk_emotional_fsaverage = morph_milk_emotional.apply(stc_milk_emotional)
    # stc_milk_neutral_fsaverage   = morph_milk_neutral.apply(stc_milk_neutral)
    # stc_milk_pwordc_fsaverage    = morph_fruit_pwordc.apply(stc_milk_pwordc)
    # stc_milk_target_fsaverage    = morph_milk_target.apply(stc_milk_target)

    # stc_LD_visual_fsaverage    = morph_LD_visual.apply(stc_LD_visual)
    # stc_LD_hear_fsaverage      = morph_LD_hear.apply(stc_LD_hear)
    # stc_LD_hand_fsaverage      = morph_LD_hand.apply(stc_LD_hand)
    # stc_LD_emotional_fsaverage = morph_LD_emotional.apply(stc_LD_emotional)
    # stc_LD_neutral_fsaverage   = morph_LD_neutral.apply(stc_LD_neutral)
    # stc_LD_pwordc_fsaverage    = morph_LD_pwordc.apply(stc_LD_pwordc)
    # stc_LD_pworda_fsaverage    = morph_LD_pworda.apply(stc_LD_pworda)
    # stc_LD_filler_fsaverage    = morph_LD_filler.apply(stc_LD_filler)
 
    
  
    fname_SD_words_fsaverage = C.data_path + meg + 'block_SD_words_EMEG_fsaverage'
    fname_LD_words_fsaverage = C.data_path + meg + 'block_LD_words_EMEG_fsaverage'
    
    # fname_fruit_visual_fsaverage    = C.data_path + meg + 'block_fruit_visual_EMEG_fsaverage'
    # fname_fruit_hear_fsaverage      = C.data_path + meg + 'block_fruit_hear_EMEG_fsaverage'
    # fname_fruit_hand_fsaverage      = C.data_path + meg + 'block_fruit_hand_EMEG_fsaverage'
    # fname_fruit_emotional_fsaverage = C.data_path + meg + 'block_fruit_emotional_EMEG_fsaverage'
    # fname_fruit_neutral_fsaverage   = C.data_path + meg + 'block_fruit_neutral_EMEG_fsaverage'
    # fname_fruit_pwordc_fsaverage    = C.data_path + meg + 'block_fruit_pwordc_EMEG_fsaverage'
    # fname_fruit_target_fsaverage    = C.data_path + meg + 'block_fruit_target_EMEG_fsaverage'

    # fname_odour_visual_fsaverage    = C.data_path + meg + 'block_odour_visual_EMEG_fsaverage'
    # fname_odour_hear_fsaverage      = C.data_path + meg + 'block_odour_hear_EMEG_fsaverage'
    # fname_odour_hand_fsaverage      = C.data_path + meg + 'block_odour_hand_EMEG_fsaverage'
    # fname_odour_emotional_fsaverage = C.data_path + meg + 'block_odour_emotional_EMEG_fsaverage'
    # fname_odour_neutral_fsaverage   = C.data_path + meg + 'block_odour_neutral_EMEG_fsaverage'
    # fname_odour_pwordc_fsaverage    = C.data_path + meg + 'block_odour_pwordc_EMEG_fsaverage'
    # fname_odour_target_fsaverage    = C.data_path + meg + 'block_odour_target_EMEG_fsaverage'

    # fname_milk_visual_fsaverage    = C.data_path + meg + 'block_milk_visual_EMEG_fsaverage'
    # fname_milk_hear_fsaverage      = C.data_path + meg + 'block_milk_hear_EMEG_fsaverage'
    # fname_milk_hand_fsaverage      = C.data_path + meg + 'block_milk_hand_EMEG_fsaverage'
    # fname_milk_emotional_fsaverage = C.data_path + meg + 'block_milk_emotional_EMEG_fsaverage'
    # fname_milk_neutral_fsaverage   = C.data_path + meg + 'block_milk_neutral_EMEG_fsaverage'
    # fname_milk_pwordc_fsaverage    = C.data_path + meg + 'block_milk_pwordc_EMEG_fsaverage'
    # fname_milk_target_fsaverage    = C.data_path + meg + 'block_milk_target_EMEG_fsaverage'

    # fname_LD_visual_fsaverage    = C.data_path + meg + 'block_LD_visual_EMEG_fsaverage'
    # fname_LD_hear_fsaverage      = C.data_path + meg + 'block_LD_hear_EMEG_fsaverage'
    # fname_LD_hand_fsaverage      = C.data_path + meg + 'block_LD_hand_EMEG_fsaverage'
    # fname_LD_emotional_fsaverage = C.data_path + meg + 'block_LD_emotional_EMEG_fsaverage'
    # fname_LD_neutral_fsaverage   = C.data_path + meg + 'block_LD_neutral_EMEG_fsaverage'
    # fname_LD_pwordc_fsaverage    = C.data_path + meg + 'block_LD_pwordc_EMEG_fsaverage'
    # fname_LD_pworda_fsaverage    = C.data_path + meg + 'block_LD_pworda_EMEG_fsaverage'
    # fname_LD_filler_fsaverage    = C.data_path + meg + 'block_LD_filler_EMEG_fsaverage'


    stc_SD_words_fsaverage.save(fname_SD_words_fsaverage)
    stc_LD_words_fsaverage.save(fname_LD_words_fsaverage)

    # stc_fruit_visual_fsaverage.save(fname_fruit_visual_fsaverage)
    # stc_fruit_hear_fsaverage.save(fname_fruit_hear_fsaverage)
    # stc_fruit_hand_fsaverage.save(fname_fruit_hand_fsaverage)
    # stc_fruit_emotional_fsaverage.save(fname_fruit_emotional_fsaverage)
    # stc_fruit_neutral_fsaverage.save(fname_fruit_neutral_fsaverage)
    # stc_fruit_pwordc_fsaverage.save(fname_fruit_pwordc_fsaverage)
    # stc_fruit_target_fsaverage.save(fname_fruit_target_fsaverage)
    
    # stc_odour_visual_fsaverage.save(fname_odour_visual_fsaverage)
    # stc_odour_hear_fsaverage.save(fname_odour_hear_fsaverage)
    # stc_odour_hand_fsaverage.save(fname_odour_hand_fsaverage)
    # stc_odour_emotional_fsaverage.save(fname_odour_emotional_fsaverage)
    # stc_odour_neutral_fsaverage.save(fname_odour_neutral_fsaverage)
    # stc_odour_pwordc_fsaverage.save(fname_odour_pwordc_fsaverage)
    # stc_odour_target_fsaverage.save(fname_odour_target_fsaverage)
    
    # stc_milk_visual_fsaverage.save(fname_milk_visual_fsaverage)
    # stc_milk_hear_fsaverage.save(fname_milk_hear_fsaverage)
    # stc_milk_hand_fsaverage.save(fname_milk_hand_fsaverage)
    # stc_milk_emotional_fsaverage.save(fname_milk_emotional_fsaverage)
    # stc_milk_neutral_fsaverage.save(fname_milk_neutral_fsaverage)
    # stc_milk_pwordc_fsaverage.save(fname_milk_pwordc_fsaverage)
    # stc_milk_target_fsaverage.save(fname_milk_target_fsaverage)
    
    # stc_LD_visual_fsaverage.save(fname_LD_visual_fsaverage)
    # stc_LD_hear_fsaverage.save(fname_LD_hear_fsaverage)
    # stc_LD_hand_fsaverage.save(fname_LD_hand_fsaverage)
    # stc_LD_emotional_fsaverage.save(fname_LD_emotional_fsaverage)
    # stc_LD_neutral_fsaverage.save(fname_LD_neutral_fsaverage)
    # stc_LD_pwordc_fsaverage.save(fname_LD_pwordc_fsaverage)
    # stc_LD_pworda_fsaverage.save(fname_LD_pworda_fsaverage)
    # stc_LD_filler_fsaverage.save(fname_LD_filler_fsaverage)

    #  ##...................................EEG ..............................##
    