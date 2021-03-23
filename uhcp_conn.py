#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:12:43 2020

@author: sr05
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import sn_config as C
from surfer import Brain
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2

for win in np.arange(0, len(C.con_time_window)):
    tmin = C.con_time_window[win]
    tmax = C.con_time_window[win]+ C.con_time_window_len
    for freq in np.arange(0, len(C.con_freq_band)-1):
        fmin = C.con_freq_band[freq]
        fmax = C.con_freq_band[freq+1]     
        for i in np.arange(0, len(subjects)-15):
            n_subjects = len(subjects)
            meg = subjects[i]
            print('Participant : ' , i, '/ win : ',win, '/ freq : ',freq)
            
            # Reading epochs
            epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
            epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
                
            epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
            epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

            
            epochs_SD = epochs_sd['words'].crop(tmin,tmax)
            epochs_LD = epochs_ld['words'].crop(tmin,tmax)
        
            # Equalize trial counts to eliminate bias (which would otherwise be
            # introduced by the abs() performed below)
            equalize_epoch_counts([epochs_SD, epochs_LD])
            
            # Reading inverse operator
            inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
            inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'
        
            inv_op_SD = read_inverse_operator(inv_fname_SD) 
            inv_op_LD = read_inverse_operator(inv_fname_LD) 
                        
            stc_SD = apply_inverse_epochs(epochs_SD, inv_op_SD,lambda2,method ='MNE', 
                                  pick_ori="normal", return_generator=True)
            stc_LD = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                                    pick_ori="normal", return_generator=True)
        

            labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',\
                                                subjects_dir=C.data_path)
                
            label_names = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TF_ROI-lh','L_TE2a_ROI-lh',
                           'L_TE2p_ROI-lh' ,'L_TE1a_ROI-lh','L_TE1m_ROI-lh']
            
#                                ,'L_TE1p_ROI-lh',
#                           'L_PHT_ROI-lh','L_TPOJ1_ROI-lh','L_TPOJ2_ROI-lh','L_TPOJ3_ROI-lh',
#                           'L_PSL_ROI-lh','L_STV_ROI-lh','L_44_ROI-lh','L_45_ROI-lh',
#                           'L_47l_ROI-lh','L_IFJa_ROI-lh','L_IFJp_ROI-lh','L_IFSa_ROI-lh',
#                           'L_IFSp_ROI-lh',
#                           'R_TGd_ROI-rh','R_TGv_ROI-rh','R_TF_ROI-rh','R_TE2a_ROI-rh',
#                           'R_TE2p_ROI-rh','R_TE1a_ROI-rh','R_TE1m_ROI-rh','R_TE1p_ROI-rh',
#                           'R_PHT_ROI-rh','R_TPOJ1_ROI-rh','R_TPOJ2_ROI-rh','R_TPOJ3_ROI-rh',
#                           'R_PSL_ROI-rh','R_STV_ROI-rh','R_44_ROI-rh','R_45_ROI-rh',
#                           'R_47l_ROI-rh','R_IFJa_ROI-rh','R_IFJp_ROI-rh','R_IFSa_ROI-rh',
#                           'R_IFSp_ROI-rh']
##            
            my_label=[]
            for j in np.arange(0,len(label_names )):
                my_label.append([label for label in labels if label.name == label_names[j]][0])

            
            src_SD = inv_op_SD['src']
            src_LD = inv_op_LD['src']
            # Average the source estimates within each label using sign-flips to reduce
            # signal cancellations, also here we return a generator
             
            label_ts_SD = mne.extract_label_time_course(stc_SD, my_label, src_SD,\
                          mode='mean_flip', return_generator=True)       
            label_ts_LD = mne.extract_label_time_course(stc_LD, my_label, src_LD,\
                          mode='mean_flip', return_generator=True)  
            

            con_SD,freqs_SD,times_SD,n_epochs_SD,n_tapers_SD=spectral_connectivity(\
                        label_ts_SD,method='imcoh', mode='fourier',sfreq=C.sfreq,\
                        fmin=fmin,fmax=fmax,n_jobs=1)   
            
            con_LD,freqs_LD,times_LD,n_epochs_LD,n_tapers_LD=spectral_connectivity(\
                        label_ts_LD,method='imcoh', mode='fourier',sfreq=C.sfreq,\
                        fmin=fmin,fmax=fmax,n_jobs=1)

            C.im_coh_sd[win, freq, i, :, :] = con_SD.copy().mean(2)
            C.im_coh_ld[win, freq, i, :, :] = con_LD.copy().mean(2)

        C.im_coh_sd_ld[win, freq, :, :]= C.im_coh_sd[win, freq, :, :, :].copy().mean(0) - C.im_coh_ld[win,\
                      freq,:,:,:].copy().mean(0)
            
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))
brain.add_annotation('HCPMMP1')


aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]



for m in np.arange(0,len(my_label)):
    if m==0:
        ATL = my_label[m]
    else:
        ATL = ATL + my_label[m]
        
brain.add_label(ATL, borders=False)

for m in np.arange(0,len(my_label)):
    
    brain.add_label(my_label[m], borders=True)



#con_SD_LD = dict()
#
#con_SD_LD['imcoh'] = C.ImCoh_SD_LD 
# np.save('/home/sr05/Python/'+'connectivity',C.ImCoh_SD_LD )
# con_SD_LD['imcoh'] = np.load('/home/sr05/Python/connectivity.npy')
#for win in np.arange(0, len(C.con_time_window)):
#    tmin = C.con_time_window[win]
#    tmax = C.con_time_window[win]+ C.con_time_window_len
#    for freq in np.arange(0, len(C.con_freq_band)-1):
#  
#        fmin = C.con_freq_band[freq]
#        fmax = C.con_freq_band[freq+1]             
#        con_SD_LD['imcoh'] = C.ImCoh_SD_LD[win,freq,:,:]    
#
#        # labels = mne.read_labels_from_annot( 'fsaverage', parc='aparc',
#                                            # subjects_dir=C.data_path)
#        # unknwon = labels.pop(68)
#        label_colors = [label.color for label in my_label]   
#        label_names = [label.name for label in my_label]
#    
#        lh_labels = [name for name in label_names if name.endswith('lh')]
#        
#        # Get the y-location of the label
#        label_ypos = list()
#        for name in lh_labels:
#            idx = label_names.index(name)
#            ypos = np.mean(my_label[idx].pos[:, 1])
#            label_ypos.append(ypos)
#        
#        # Reorder the labels based on their location
#        lh_labels = [label for (yp, label) in (zip(label_ypos, lh_labels))]
#        
#        # For the right hemi
#        rh_labels = ['R' + label[1:-2] + 'rh' for label in lh_labels]
#        
#        # Save the plot order and create a circular layout
#        node_order = list()
#        node_order.extend(lh_labels[::-1])  # reverse the order
#        node_order.extend(rh_labels)
#        
#        node_angles = circular_layout(label_names, node_order, start_pos=90,
#                                      group_boundaries=[0, len(label_names) / 2])
#        
#        # Plot the graph using node colors from the FreeSurfer parcellation. We only
#        # show the 300 strongest connections.
#        fig_con, axes_con = plot_connectivity_circle(con_SD_LD['imcoh'], label_names, n_lines=50,\
#                          node_angles=node_angles, node_colors=label_colors,\
#                          title='All-to-All Connectivity(ImCoh) SD-LD'+\
#                          f'{tmin:.3f}' +'_'+f'{tmax:.3f}'+'_'+f'{fmin:.3f}' +'_'+\
#                          f'{fmax:.3f}')  
#        fig_con.savefig(C.pictures_path_Source_estimate+ 'All-to-All Connectivity(ImCoh)50_SD-LD_'+\
#                          f'{tmin:.3f}' +'_'+f'{tmax:.3f}'+'_'+f'{fmin:.3f}' +'_'+\
#                          f'{fmax:.3f}50hcp.jpg')    
#        
