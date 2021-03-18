#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:10:38 2020

@author: sr05
"""


import mne
import numpy as np
import SN_config as C
# import my_baseline_correction 

from mne.stats import (permutation_cluster_1samp_test,
                       summarize_clusters_stc,permutation_cluster_test)
from scipy import stats as stats
# from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import (spatio_temporal_cluster_test, f_threshold_mway_rm,
                       f_mway_rm, summarize_clusters_stc)
from matplotlib import pyplot as plt
import statsmodels.stats.multicomp as multi

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_MRI
# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = '/imaging/rf02/TypLexMEG/fsaverage/label'
# X = np.zeros([20484, 1, 18, 2])
# X = np.zeros([20484,20, len(subjects), 2])

def stc_baseline_correction(X):
    time_dim = len(X.times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    baseline_timepoints = X.times[0:301]

    baseline_mean = X.data[:,0:len(baseline_timepoints)].mean(1)

    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean),1]),\
                                  time_dim  ,axis=1)
    corrected_stc = X - baseline_mean_mat
    return corrected_stc
  
def SN_semantic_ROIs():
    # Loading Human Connectom Project parcellation
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=C.data_path,verbose=True)
    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',\
                                        subjects_dir=C.data_path)
    
 
    ##............................. Control Regions ............................##
        
    # Temporal area - Splitting STSvp 
    label_STSvp = ['L_STSvp_ROI-lh']
    my_STSvp=[]
    for j in np.arange(0,len(label_STSvp )):
        my_STSvp.append([label for label in labels if label.name == \
                         label_STSvp[j]][0])
    
    for m in np.arange(0,len(my_STSvp)):
        if m==0:
            STSvp = my_STSvp[m]
        else:
            STSvp = STSvp + my_STSvp[m]
            
            
    [STSvp1,STSvp2,STSvp3,STSvp4,STSvp5,STSvp6]=mne.split_label(label=STSvp,parts\
        =('L_STSvp1_ROI-lh','L_STSvp2_ROI-lh','L_STSvp3_ROI-lh','L_STSvp4_ROI-lh',
          'L_STSvp5_ROI-lh','L_STSvp6_ROI-lh',),subject='fsaverage',subjects_dir=\
          C.data_path)
        
    # Temporal area - Splitting PH 
    label_PH = ['L_PH_ROI-lh']
    my_PH=[]
    for j in np.arange(0,len(label_PH )):
        my_PH.append([label for label in labels if label.name == label_PH[j]][0])
    
    for m in np.arange(0,len(my_PH)):
        if m==0:
            PH = my_PH[m]
        else:
            PH = PH + my_PH[m]
    
    [PH1,PH2]=mne.split_label(label=PH,parts=('L_PH1_ROI-lh','L_PH2_ROI-lh')\
              ,subject='fsaverage',subjects_dir=C.data_path)
    [PH21,PH22,PH23,PH24]=mne.split_label(label=PH2,parts=\
              ('L_PH21_ROI-lh','L_PH22_ROI-lh','L_PH23_ROI-lh','L_PH24_ROI-lh'),\
              subject='fsaverage',subjects_dir=C.data_path)
    
    
    # Temporal area - Splitting TE2p  
    label_TE2p = ['L_TE2p_ROI-lh']
    my_TE2p=[]
    for j in np.arange(0,len(label_TE2p )):
        my_TE2p.append([label for label in labels if label.name == label_TE2p[j]][0])
    
    for m in np.arange(0,len(my_TE2p)):
        if m==0:
            TE2p = my_TE2p[m]
        else:
            TE2p = TE2p + my_TE2p[m]
            
    
    [TE2p1,TE2p2]=mne.split_label(label=TE2p,parts=('L_TE2p1_ROI-lh',\
        'L_TE2p2_ROI-lh'),subject='fsaverage',subjects_dir=C.data_path)
    
    
    # Temporal area
    label_TE1p = ['L_TE1p_ROI-lh']
    my_TE1p=[]
    for j in np.arange(0,len(label_TE1p )):
        my_TE1p.append([label for label in labels if label.name == label_TE1p[j]][0])
    
    for m in np.arange(0,len(my_TE1p)):
        if m==0:
            TE1p = my_TE1p[m]
        else:
            TE1p = TE1p + my_TE1p[m]
            
    TG= STSvp1 + STSvp2 + STSvp3 + STSvp4 + TE2p1 + PH24 +TE1p
    
    
    
    ##.......................... Representation Regions .........................##
            
    # Left ATL area - splitting TE2a      
    label_TE2a = ['L_TE2a_ROI-lh']
    my_TE2a=[]
    for j in np.arange(0,len(label_TE2a )):
        my_TE2a.append([label for label in labels if label.name == label_TE2a[j]][0])
    
    for m in np.arange(0,len(my_TE2a)):
        if m==0:
            l_TE2a = my_TE2a[m]
        else:
            l_TE2a = l_TE2a + my_TE2a[m]
            
    
    [l_TE2a1,l_TE2a2,l_TE2a3]=mne.split_label(label=l_TE2a,parts=\
        ('L_TE2a1_ROI-lh','L_TE2a2_ROI-lh','L_TE2a3_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
        
        
    # Left ATL area - splitting TE1m 
    label_TE1m = ['L_TE1m_ROI-lh']
    my_TE1m=[]
    for j in np.arange(0,len(label_TE1m )):
        my_TE1m.append([label for label in labels if label.name == label_TE1m[j]][0])
    
    for m in np.arange(0,len(my_TE1m)):
        if m==0:
            l_TE1m = my_TE1m[m]
        else:
            l_TE1m = l_TE1m + my_TE1m[m]
            
    
    [l_TE1m1,l_TE1m2,l_TE1m3]=mne.split_label(label=l_TE1m,parts=\
        ('L_TE1m1_ROI-lh','L_TE1m2_ROI-lh','L_TE1m3_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)       
    [l_TE1m11,l_TE1m12,l_TE1m13]=mne.split_label(label=l_TE1m1,parts=\
        ('L_TE1m11_ROI-lh','L_TE1m12_ROI-lh','L_TE1m13_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    [l_TE1m21,l_TE1m22,l_TE1m23]=mne.split_label(label=l_TE1m2,parts=\
        ('L_TE1m21_ROI-lh','L_TE1m22_ROI-lh','L_TE1m23_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
    # Left ATL area  
    label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TE1a_ROI-lh']
    
    
    my_ATL=[]
    for j in np.arange(0,len(label_ATL )):
        my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])
    
    for m in np.arange(0,len(my_ATL)):
        if m==0:
            l_ATL = my_ATL[m]
        else:
            l_ATL = l_ATL + my_ATL[m]
            
    l_ATL = l_ATL + l_TE2a2 + l_TE2a3 + l_TE1m13 + l_TE1m23
    
    
    # Right ATL area - splitting TE2a      
    label_TE2a = ['R_TE2a_ROI-rh']
    my_TE2a=[]
    for j in np.arange(0,len(label_TE2a )):
        my_TE2a.append([label for label in labels if label.name == label_TE2a[j]][0])
    
    for m in np.arange(0,len(my_TE2a)):
        if m==0:
            r_TE2a = my_TE2a[m]
        else:
            r_TE2a = r_TE2a + my_TE2a[m]
            
    
    [r_TE2a1,r_TE2a2,r_TE2a3]=mne.split_label(label=r_TE2a,parts=\
        ('R_TE2a1_ROI-rh','R_TE2a2_ROI-rh','R_TE2a3_ROI-rh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
    # Right ATL area - splitting TE1m 
    label_TE1m = ['R_TE1m_ROI-rh']
    my_TE1m=[]
    for j in np.arange(0,len(label_TE1m )):
        my_TE1m.append([label for label in labels if label.name == label_TE1m[j]][0])
    
    for m in np.arange(0,len(my_TE1m)):
        if m==0:
            r_TE1m = my_TE1m[m]
        else:
            r_TE1m = r_TE1m + my_TE1m[m]
            
    
    [r_TE1m1,r_TE1m2,r_TE1m3]=mne.split_label(label=r_TE1m,parts=\
        ('R_TE1m1_ROI-rh','R_TE1m2_ROI-rh','R_TE1m3_ROI-rh'),subject='fsaverage',\
        subjects_dir=C.data_path)       
    
    [r_TE1m31,r_TE1m32,r_TE1m33]=mne.split_label(label=r_TE1m3,parts=\
        ('R_TE1m31_ROI-rh','R_TE1m32_ROI-rh','R_TE1m33_ROI-rh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
    # Right ATL area  
    label_ATL = ['R_TGd_ROI-rh','R_TGv_ROI-rh','R_TE1a_ROI-rh']
    
    
    my_ATL=[]
    for j in np.arange(0,len(label_ATL )):
        my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])
    
    for m in np.arange(0,len(my_ATL)):
        if m==0:
            r_ATL = my_ATL[m]
        else:
            r_ATL = r_ATL + my_ATL[m]
            
    r_ATL = r_ATL + r_TE2a2 + r_TE2a3 + r_TE1m33
    
    ## ............................ Angular Gyrus .............................. ##
    
    label_AG = ['L_PGi_ROI-lh','L_PGp_ROI-lh','L_PGs_ROI-lh']
    
    my_AG=[]
    for j in np.arange(0,len(label_AG)):
        my_AG.append([label for label in labels if label.name == label_AG[j]][0])
    
    for m in np.arange(0,len(my_AG )):
        if m==0:
            AG = my_AG[m]
        else:
            AG = AG  + my_AG[m]      
    
    
    ## ....................... Inferior Frontal Gyrus  ......................... ##
         
            
    label_IFG = ['L_44_ROI-lh','L_45_ROI-lh','L_47l_ROI-lh','L_p47r_ROI-lh']   
    my_IFG=[]
    for j in np.arange(0,len(label_IFG )):
        my_IFG.append([label for label in labels if label.name == label_IFG[j]][0])
    
    for m in np.arange(0,len(my_IFG)):
        if m==0:
            IFG = my_IFG[m]
        else:
            IFG = IFG + my_IFG[m] 
            
    ## ............................, Visual Area ............................... ##
    label_V1 = ['L_V1_ROI-lh','L_V2_ROI-lh','L_V3_ROI-lh','L_V4_ROI-lh']        
    my_V1 =[]        
    for j in np.arange(0,len(label_V1 )):
        my_V1.append([label for label in labels if label.name == label_V1[j]][0])
    V1= my_V1[0]
    
                   
    my_labels =[l_ATL,r_ATL,TG,IFG,AG,V1]
    return my_labels


SN_ROI = SN_semantic_ROIs()    

X = np.zeros([18 ,6, 1201])
Y = np.zeros([18 ,6, 1201])


for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]

    print('Participant : ' , i)

    # Reading epochs
    epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'     
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
        
    epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)
    
    epochs_SD = epochs_sd['words'] 
    epochs_LD = epochs_ld['words'] 

   
    # Reading inverse operator
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    inv_op_SD = read_inverse_operator(inv_fname_SD) 
    inv_op_LD = read_inverse_operator(inv_fname_LD) 

    
    # Evoked responses 
    evoked_SD = epochs_SD.average().set_eeg_reference(ref_channels = \
                        'average',projection=True)
    evoked_LD = epochs_LD.average().set_eeg_reference(ref_channels = \
                        'average',projection=True)

    # Applying inverse solution to get sourse signals    
    stc_sd = apply_inverse(evoked_SD, inv_op_SD,lambda2,method ='MNE', 
                           pick_ori=None)
    stc_ld = apply_inverse(evoked_LD, inv_op_LD,lambda2,method ='MNE',
                           pick_ori=None)

    # stc_sd_corrected = stc_baseline_correction(stc_sd) 
    # stc_ld_corrected = stc_baseline_correction(stc_ld) 
   
    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']
    
    morphed_labels = mne.morph_labels(SN_ROI,subject_to=sub_to,\
                  subject_from='fsaverage',subjects_dir=C.data_path)

    
    # label_ts_SD = mne.extract_label_time_course(stc_SD, morphed_labels,\
    #               src_SD, mode='mean_flip')       
    # label_ts_LD = mne.extract_label_time_course(stc_LD, morphed_labels,\
    #               src_LD,mode='mean_flip') 
    
    # X[i,0:6,:] = label_ts_SD[:,0:850] 
    # Y[i,0:6,:] = label_ts_LD[:,0:850] 
        
    for j in np.arange(0,6):
        morphed_labels[j].subject = sub_to   
        
        label_vertices =  stc_sd.in_label(morphed_labels[j])  
        stc_sd_corrected = stc_baseline_correction(label_vertices) 
        X[i,j,:]  = abs(stc_sd_corrected.data).copy().mean(0)
        
        label_vertices =  stc_ld.in_label(morphed_labels[j])  
        stc_ld_corrected = stc_baseline_correction(label_vertices) 
        Y[i,j,:]  = abs(stc_ld_corrected.data).copy().mean(0)
        
        
        # label_vertices =  stc_ld.in_label(morphed_labels[j])    
        # Y[i,j,:]  = abs(label_vertices.data).copy().mean(0)

Z = X[:,:,0:850] - Y[:,:,0:850]
# r = Z.copy().mean(0).mean(1).reshape(6,1)
# R = np.repeat(r,850).reshape(6,850)
times= epochs_SD.times[0:850]

S=np.zeros([18,4,850])   

S[:,0:2,:] =  X[:,0:2,0:850]
S[:,2:4,:] =  Y[:,0:2,0:850]
# Z = Z - R
# t-test and cluster-based correction
T = np.arange(-300,550)
p_threshold = 0.05
n_permutations=5000
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
t_thresh = np.repeat(t_threshold,len(T),axis=0)
lb=['lATL','rATL','TG','IFG','AG','PVA']


# for j in np.arange(0,6):
    
#     T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
#     Z[:,j,:], n_jobs=4, threshold=t_threshold, connectivity=None,
#     n_permutations=n_permutations, out_type='mask')
    
#     plt.figure()
#     plt.subplot(211)
#     plt.title('t-test and cluster-based permutation on '+lb[j])
#     plt.plot(times, X[:,j,0:850].mean(axis=0),'b',label=lb[j]+" time-series: SD")
#     plt.plot(times, Y[:,j,0:850].mean(axis=0),'r',label=lb[j]+" time-series: LD")
#     plt.plot(times, Z[:,j,:].mean(axis=0),'m',label=lb[j]+" time-series: SD - LD")

#     plt.ylabel("EEG/MEG")
#     plt.legend(loc='upper left')
#     plt.subplot(212)
    
#     for i_c, c in enumerate(clusters):
#         print(i_c , c)
#         c = c[0]
#         if cluster_p_values[i_c] <= 0.05:
#             h = plt.axvspan(times[c.start], times[c.stop - 1],
#                             color='r', alpha=0.3)
#             plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

#         else:
#             h1= plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
#                         alpha=0.3)
#             plt.legend((h1, ), ('t-test p-value < 0.05', ),loc='upper left')
#             # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

#     hf = plt.plot(times, T_obs, 'g')
   
#     plt.xlabel("time (ms)")
#     plt.ylabel("t-values")
#     plt.show()
#     plt.savefig(C.pictures_path_Source_estimate+ 't-test_timeseries_abs_'+lb[j]+'_.png')
# # 
# 
# plt.close('all')


###############################################################################

# ## 2way repeated measure
   
fvals, pvals = f_mway_rm(S, factor_levels=[2,2], effects='all')

thresh = mne.stats.f_threshold_mway_rm(18,factor_levels=[2,2], effects='all')

# T = np.arange(0,550)
my_colors = ['darkblue', 'darkgreen','darkred']
rect_colors = ['lightsteelblue', 'paleturquoise','lightcoral']

y_label =['SD/LD', 'ATLs', 'SD/LD by ATLs']

p_thresh = np.repeat(0.05,len(T),axis=0)


plt.figure()

for k in np.arange(1,4):  
    plt.subplot(3,1,k)
    if k==1:
        plt.title("Two-way Repeated Measure ANOVA")
    plt.plot(T, fvals[k-1], my_colors[k-1])
   
    plt.ylabel(y_label[k-1])
    plt.legend(loc='upper left')
    
    
    for c,f in enumerate( fvals[k-1]):
        # print(c , f)
        if f >= thresh[0]:
            h = plt.axvspan(T[c], T[c],
                            color=rect_colors[k-1], alpha=0.3)
            plt.legend((h, ), ('f-test p-value < 0.05', ),loc='upper left')
          
    plt.xlabel("time (ms)")

    plt.show()
    
