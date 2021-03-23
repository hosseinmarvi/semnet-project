
"""
Created on Wed Jul  1 11:12:43 2020

@author: sr05
"""
import os
import pickle
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
method='coh'
stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/betweenLabels_'+method+'_SD.json'
stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/betweenLabels_'+method+'_LD.json'
# f_min=C.con_freq_band[0]
# f_max=C.con_freq_band[-1]
# freqs=np.arange(f_min,f_max+2,2)
cwt_n_cycles = freqs/3
n_subjects = len(subjects)



times = np.arange(-300,901)
stc_total_SD=np.zeros([18,6,6,19,600])
stc_total_LD=np.zeros([18,6,6,19,600])
           
for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]

    print('Participant : ' , i)

    morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                  subject_from='fsaverage',subjects_dir=data_path)
        

    # Reading epochs
    epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
        
    epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

    
    epochs_SD = epochs_sd['words'].copy().resample(500)
    epochs_LD = epochs_ld['words'].copy().resample(500)

    # Equalize trial counts to eliminate bias (which would otherwise be
    # introduced by the abs() performed below)
    equalize_epoch_counts([epochs_SD, epochs_LD])
    
    # Reading inverse operator
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    inv_op_SD = read_inverse_operator(inv_fname_SD) 
    inv_op_LD = read_inverse_operator(inv_fname_LD) 
                
    stc_sd = apply_inverse_epochs(epochs_SD, inv_op_SD,lambda2,method ='MNE', 
                          pick_ori="normal", return_generator=False)
    stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                            pick_ori="normal", return_generator=False)
    times=epochs_SD.times
    stc_SD=[]
    stc_LD=[]
    
    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']

    for n in np.arange(0,len(stc_sd)):
        stc_SD.append(stc_baseline_correction(stc_sd[n],times))
        stc_LD.append(stc_baseline_correction(stc_ld[n],times))

    for k in np.arange(0,6):
        morphed_labels[k].name = C.rois_labels[k]

    labels_sd = mne.extract_label_time_course(stc_SD, morphed_labels, \
                src_SD, mode='mean_flip',return_generator=False)
    labels_ld = mne.extract_label_time_course(stc_LD, morphed_labels, \
                src_LD, mode='mean_flip',return_generator=False)
  

    
    con_SD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        labels_sd, method=method, mode='cwt_morlet',cwt_n_cycles=cwt_n_cycles,
        sfreq=500, fmin=f_min, fmax=f_max,cwt_freqs=freqs,  n_jobs=10)
   
    con_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        labels_ld, method=method, mode='cwt_morlet', cwt_n_cycles=cwt_n_cycles,
        sfreq=500, fmin=f_min, fmax=f_max, cwt_freqs=freqs, n_jobs=10)

    stc_total_SD[i:,:,:,:]= con_SD
    stc_total_LD[i:,:,:,:]= con_LD

end=time.time()
print(end-start)




with open(stc_SD_file_name, "wb") as fp:   #Pickling
    pickle.dump(stc_total_SD, fp)
    
with open(stc_LD_file_name, "wb") as fp:   #Pickling
    pickle.dump(stc_total_LD, fp)
#########################################################

with open(stc_SD_file_name, "rb") as fp:   # Unpickling
    conn_SD = pickle.load(fp)

with open(stc_LD_file_name, "rb") as fp:   # Unpickling
    conn_LD = pickle.load(fp)
# tail = 0  
# a=175
# b=174
# # # T=epochs.times 
# T=np.arange(-300,900,2)*1e-3
# t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., n_subjects - 1)
# times=T[a:-b]

##############################################################################
# ## plot each ROI across subjects: columns 0-5->SD, 6:11->LD 
# ## sequence of ROIs: 0:lATL, 1:rATL, 2:TG, 3:IFG, 4:AG, 5:PVA
# labels_name=C.ROIs_lables

# for i in np.arange(0,6): 
#     for j in np.arange(i+1,6):
#         X=conn_SD[:,j,i,:,:].copy().mean(0)
#         Y=conn_LD[:,j,i,:,:].copy().mean(0)
#         vmax = max(X.max(),Y.max())
#         vmin = min(X.min(),Y.min())
#         v=max(abs(vmax),abs(vmin))
#         plt.figure()
#         plt.subplot(3,1,1)
#         plt.imshow( X,vmax=v,vmin=-v,
#                         extent=[times[1], times[-1], freqs[0], freqs[-1]],
#                         aspect='auto', origin='lower', cmap='RdBu_r') 
#         plt.title(method+': '+C.ROIs_lables[i]+'-'+C.ROIs_lables[j])
#         plt.colorbar()
#         plt.ylabel('SD (HZ)')
        
#         vmax = max(X.max(),Y.max())
#         vmin = min(X.min(),Y.min())
#         v=max(abs(vmax),abs(vmin))
#         plt.subplot(3,1,2)
#         plt.imshow( Y,vmax=v,vmin=-v,
#                         extent=[times[1], times[-1], freqs[0], freqs[-1]],
#                         aspect='auto', origin='lower', cmap='RdBu_r') 
#         plt.colorbar()
#         plt.ylabel('LD (HZ)')
        
#         vmax = (X-Y).max()
#         vmin = (X-Y).min()
#         v=max(abs(vmax),abs(vmin))
        
#         plt.subplot(3,1,3)
#         plt.imshow( X-Y,vmax=v,vmin=-v,
#                         extent=[times[1], times[-1], freqs[0], freqs[-1]],
#                         aspect='auto', origin='lower', cmap='RdBu_r') 
#         plt.colorbar()
#         plt.ylabel('SD-LD (HZ)')
#         plt.savefig(C.pictures_path_Source_estimate+ 'betweenLabels_'+method+'_'+C.ROIs_lables[i]+'-'+C.ROIs_lables[j]+'.png')
# plt.close('all')

##############################################################################
### t-test and cluster-based correction for each ROI

# lb=C.ROIs_lables
# # # difference of SD (0:6) and LD(6:12) for aech ROI and individual
# Z= conn_SD[:,:,:,:,a:-b]-conn_LD[:,:,:,:,a:-b]

# for i in np.arange(0,6): 
#     for j in np.arange(i+1,6):
        
#         T_obs, clusters, cluster_p_values, H0 = \
#             permutation_cluster_1samp_test(Z[:,j,i,:,:], n_permutations=C.n_permutations,
#                                             threshold=t_threshold, tail=tail,
#                                             connectivity=None,out_type='mask',
#                                             verbose=True)
            
#         T_obs_plot = np.nan * np.ones_like(T_obs)
#         for c, p_val in zip(clusters, cluster_p_values):
#             if p_val <= C.pvalue:
#                 T_obs_plot[c] = T_obs[c]
                
#         T_obs_ttest = np.nan * np.ones_like(T_obs)
#         for r in np.arange(0,freqs.shape[0]):
#             for c in np.arange(0,times.shape[0]):
#                 if abs(T_obs[r,c])>t_threshold:
#                     T_obs_ttest[r,c] =  T_obs[r,c]
                
#         vmax = np.max(T_obs)
#         vmin = np.min(T_obs)
#         v=max(abs(vmax),abs(vmin))
#         plt.figure()
#         # plotting the t-values    
#         plt.subplot(311)
#         plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
#                     extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect='auto', origin='lower', vmin=-v, vmax=v)
#         plt.colorbar()
#         plt.ylabel('Frequency (Hz)')
#         plt.title(method+': '+C.ROIs_lables[i]+' to '+C.ROIs_lables[j]+' (SD vs LD)')
        
#         # plotting the uncorreted t-test
#         plt.subplot(312)
#         plt.imshow(T_obs, cmap=plt.cm.bone,
#                     extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect='auto', origin='lower', vmin=-v, vmax=v)
        
#         plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
#                     extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect='auto', origin='lower', vmin=-v, vmax=v)
#         plt.colorbar()
#         plt.ylabel('Frequency (Hz)')
    
#         # plotting the correted t-test
#         plt.subplot(313)
#         plt.imshow(T_obs, cmap=plt.cm.gray,label='cluster-based permutation test',
#                     extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        
#         plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,label='cluster-based permutation test',
#                     extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                     aspect='auto', origin='lower', vmin=-v, vmax=v)
#         plt.colorbar()
#         plt.xlabel('Time (ms)')
#         plt.ylabel('Frequency (Hz)')
    
#         plt.show()      
#         plt.savefig(C.pictures_path_Source_estimate+ 't-test_'+method+'_'+C.ROIs_lables[i]+'-'+C.ROIs_lables[j]+'.png')
