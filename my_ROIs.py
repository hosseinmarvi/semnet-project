#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:55:20 2020

@author: sr05
"""


import numpy as np
import mne
import sn_config as C
from surfer import Brain
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2


mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=C.data_path,verbose=True)
        

labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',\
                                    subjects_dir=C.data_path)
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))
#brain.add_annotation('HCPMMP1')
    
# label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TF_ROI-lh','L_TE2a_ROI-lh','L_TE2p_ROI-lh' ,'L_TE1a_ROI-lh','L_TE1m_ROI-lh']

label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TE2a_ROI-lh','L_TE1a_ROI-lh','L_TE1m_ROI-lh']


my_ATL=[]
for j in np.arange(0,len(label_ATL )):
    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])

for m in np.arange(0,len(my_ATL)):
    if m==0:
        ATL = my_ATL[m]
    else:
        ATL = ATL + my_ATL[m]


label_IPC = ['L_PGi_ROI-lh','L_PGp_ROI-lh','L_PGs_ROI-lh']


my_IPC=[]
for j in np.arange(0,len(label_IPC)):
    my_IPC.append([label for label in labels if label.name == label_IPC[j]][0])

for m in np.arange(0,len(my_IPC )):
    if m==0:
        IPC = my_IPC[m]
    else:
        IPC = IPC  + my_IPC[m]      

        
label_ITG = ['L_PH_ROI-lh']


my_ITG=[]
for j in np.arange(0,len(label_ITG )):
    my_ITG.append([label for label in labels if label.name == label_ITG[j]][0])

for m in np.arange(0,len(my_ITG)):
    if m==0:
        ITG = my_ITG[m]
    else:
        ITG = ITG + my_ITG[m]
        
        
label_IFG = ['L_44_ROI-lh','L_45_ROI-lh','L_47l_ROI-lh','L_p47r_ROI-lh']   
my_IFG=[]
for j in np.arange(0,len(label_IFG )):
    my_IFG.append([label for label in labels if label.name == label_IFG[j]][0])

for m in np.arange(0,len(my_IFG)):
    if m==0:
        IFG = my_IFG[m]
    else:
        IFG = IFG + my_IFG[m] 
        
        
label_V1 = ['L_V1_ROI-lh','L_V2_ROI-lh','L_V3_ROI-lh','L_V4_ROI-lh']        
my_V1 =[]        
for j in np.arange(0,len(label_V1 )):
    my_V1.append([label for label in labels if label.name == label_V1[j]][0])
V1= my_V1[0]


label_MTG = ['L_PHT_ROI-lh']#,'L_TE1p_ROI-lh']   
my_MTG=[]
for j in np.arange(0,len(label_MTG)):
    my_MTG.append([label for label in labels if label.name == label_MTG[j]][0])

for m in np.arange(0,len(my_MTG)):
    if m==0:
        MTG = my_MTG[m]
    else:
        MTG = MTG + my_MTG[m] 
        
        
        
#for m in np.arange(0,len(my_V1)):
#    
#    brain.add_label(my_V1[m], borders=False)
#        
#
my_color=['blue','green','red']
for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False,color=my_color[m])
#    
#
#for m in np.arange(0,len(my_ITG)):
#    
#    brain.add_label(my_ITG[m], borders=False)
#        
#            
#for m in np.arange(0,len(my_MTG)):
#    
#    brain.add_label(my_MTG[m], borders=False)
#            
#        
for m in np.arange(0,len(my_IPC)):
    
    brain.add_label(my_IPC[m], borders=True)        
#        
#for m in np.arange(0,len(my_IFG)):
#    
#    brain.add_label(my_IFG[m], borders=False)          
 
        
          
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))        

brain.add_label(ATL, borders=False,color='blue')
brain.add_label(V1, borders=False, color='yellow')
brain.add_label(MTG, borders=False, color='green')
brain.add_label(ITG, borders=False, color='green')
brain.add_label(IPC, borders=False, color='purple')
brain.add_label(IFG, borders=False,color='red')



