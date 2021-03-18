#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:05:45 2020

@author: sr05
"""
import os
import mne
import sys
import time
import pickle
import numpy as np
import sn_config as C
from numpy.linalg import inv
from sklearn import linear_model
from mne.epochs import equalize_epoch_counts
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score 


# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    
ROI_x=1
ROI_y=0
s=time.time()
i=0
meg = subjects[i]
sub_to = MRI_sub[i][1:15]
SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/transformation/trans_SD_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'.json'
out_file_name=[SD_file_name]


 # morph labels from fsaverage to each subject
labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                          subject_from='fsaverage',subjects_dir=data_path)
 
 # read epochs
epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
 
epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

 # crop epochs
epochs_SD = epochs_sd['words'].copy().crop(-.200,.550).resample(500)
epochs_LD = epochs_ld['words'].copy().crop(-.200,.550).resample(500)

 # equalize trial counts to eliminate bias 
equalize_epoch_counts([epochs_SD, epochs_LD])

inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'
inv_fname_epoch=[inv_fname_SD , inv_fname_LD]
 
e=0
epoch=epochs_SD
 
GOF_ave=[]


     

output=[0]*2
 # read inverse operator,apply inverse operator     
inv_op = read_inverse_operator(inv_fname_epoch[e])     
stc = apply_inverse_epochs(epoch, inv_op,lambda2,method ='MNE', 
                       pick_ori="normal", return_generator=False)

for j,idx in enumerate([ROI_x, ROI_y]):
    labels[idx].subject = sub_to   
    # define dimentions of matrix (vertices X timepoints), & initializing    
    v, t= stc[idx].in_label(labels[idx]).data.shape      
    X= np.zeros([len(stc),v,t])       
    # create output array of size (vertices X stimuli X timepoints)    
    for s in np.arange(0,len(stc)):
        S= stc[s].in_label(labels[idx]).data
        X[s,:,:]=S.copy() -np.matlib.repmat(np.mean(S,0).reshape(1,t),v,1)
    output[j]= X 
     
X= output[0]
Y= output[1]

GOF_ave={}


for t in np.arange(100,101):
    # initialize the correlation coefficeint array of sizr vertices X 1
    GOF_r2score=np.zeros([X.shape[0],1])
    GOF_correlation=np.zeros([X.shape[0],1])
    GOF_explained_variance=np.zeros([X.shape[0],1])
    GOF_MSE=np.zeros([X.shape[0],1])
    GOF_RSS=np.zeros([X.shape[0],1])

    print('sample: ',t,' from: ',X.shape[-1])
    for s in np.arange(200,220):
        print('sample: ',t,' from: ',X.shape[-1],'/ stimulus: ',s,' from: ',X.shape[0])
        # define train and test sets based on leave_one_out approach
        X_train=np.delete(X[:,:,t],s,0)
        Y_train=np.delete(Y[:,:,t],s,0)    
        X_test=X[s,:,t]
        Y_test=Y[s,:,t]                        
       
        # ridge legression: Y=BX,Y[y_vertices X nb_stimuli], 
        # X[x_vertices X nb_stimuli],B[y_vertices X x_vertices]
        reg = linear_model.RidgeCV(alphas=np.logspace(-10,10,100),normalize=True)
        reg.fit(X_train, Y_train)
        print('alpha: ',reg.alpha_)
        # beta=reg.coef_
        Y_predicted=reg.predict(X_test.reshape(X_test.shape[0],1).transpose())
    
        # compute:  Y_predicted= B*X_test  
                       
        GOF_r2score[s,0]= r2_score(Y_test,Y_predicted.reshape(Y_test.shape[0]))              
        GOF_correlation[s,0]= np.corrcoef(Y_test.reshape(1,Y_test.shape[0]),Y_predicted)[0,1]
        GOF_explained_variance[s,0]= explained_variance_score(Y_test.reshape(Y_test.shape[0],1).transpose(),Y_predicted)
        GOF_MSE[s,0]= mean_squared_error(Y_test.reshape(Y_test.shape[0],1).transpose(),Y_predicted)
        GOF_RSS[s,0]= np.sum((Y_test-Y_predicted.reshape(Y_test.shape[0]))**2)
    # compute the explained variance for every single timepoint
    # based on the average over all train-test sets            
    GOF_ave['r2score']=np.mean(GOF_r2score,0)
    GOF_ave['correlation']=np.mean(GOF_correlation,0)
    GOF_ave['explained_variance']=np.mean(GOF_explained_variance,0)
    GOF_ave['MSE']=np.mean(GOF_MSE,0)   
    GOF_ave['RSS']=np.mean(GOF_RSS,0)

with open(out_file_name[e], "wb") as fp:   #Pickling
    pickle.dump( GOF_ave, fp)

