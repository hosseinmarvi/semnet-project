

"""
Created on Wed Jul  1 11:12:43 2020

@author: sr05
"""
import os
import pickle
import joblib
import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse_epochs, apply_inverse, read_inverse_operator
from mne.connectivity import spectral_connectivity,seed_target_indices, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from mne.stats import (permutation_cluster_1samp_test,spatio_temporal_cluster_test,
                       summarize_clusters_stc,permutation_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from scipy import stats as stats
from mne.epochs import equalize_epoch_counts
import time
from joblib import Parallel,delayed


start=time.time()
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    

# times = np.arange(-300,901)
stc_total_LD=[[[0]*2 for f in range(4)] for k in range(6)]
n_subjects = len(subjects)
win=0
k=0
f=0
n_subjects = len(subjects)
s=time.time()

def Conn_test(i):            
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]

    print('Participant : ' , i)

    morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                  subject_from='fsaverage',subjects_dir=data_path)
        

    # Reading epochs
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
        
    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

    
    epochs_LD = epochs_ld['words'].copy().resample(500)

    # Reading inverse operator
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    inv_op_LD = read_inverse_operator(inv_fname_LD) 
                
   
    stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                            pick_ori="normal", return_generator=False)
    times=epochs_LD.times

    stc_LD_t =[]
    

    src_LD = inv_op_LD['src']

 
    for n in np.arange(0,len(stc_ld)):
        stc_LD_t.append(stc_baseline_correction(stc_ld[n],times))
   
    t_min = C.con_time_window[win]
    t_max = C.con_time_window[win]+ C.con_time_window_len
    stc_LD=[]
    for n in np.arange(0,len(stc_ld)):
         stc_LD.append(stc_LD_t[n].copy().crop(t_min*1e-3,t_max*1e-3))

    morphed_labels[k].name = C.rois_labels[k]
    
   
    seed_ts_ld = mne.extract_label_time_course(stc_LD, morphed_labels[k], \
               src_LD, mode='mean_flip',return_generator=False)
               
    f_min=C.con_freq_band[f]
    f_max=C.con_freq_band[f+1]
    print('Participant : ' , i, '/ ROI: ',k,' win:', win,\
          ' freq: ',f)
    comb_ts_ld = zip(seed_ts_ld, stc_LD)  
    vertices_SD = [src_LD[j]['vertno'] for j in range(2)]
    n_signals_tot = 1 + len(vertices_SD[0]) + len(vertices_SD[1])
    indices = seed_target_indices([0], np.arange(1, n_signals_tot))

   
    coh_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        comb_ts_ld, method='coh', mode='fourier', indices=indices,
        sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=6)
                
    return coh_LD


X=Parallel(n_jobs=2,verbose=50)(delayed(Conn_test)(i) for i in range(2))
# XX=[Conn_test(i) for i in range(2)]

# stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/stc_Coh_bands_SD.json'
# stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/stc_Coh_bands_LD.json'



# with open(stc_SD_file_name, "wb") as fp:   #Pickling
#     pickle.dump(stc_total_SD, fp)
    
# with open(stc_LD_file_name, "wb") as fp:   #Pickling
#     pickle.dump(stc_total_LD, fp)
e=time.time()
print(e-s)



    
