import numpy as np

def my_baseline_correction(X):
    time_dim = len(X.times)
    baseline_timepoints = X.times[np.where(X.times<0)]
    axis0,axis1 = (X.data).shape
    baseline_mean = X.data[:,0:len(baseline_timepoints)].mean(1)
    if axis0==time_dim:
        axis = 0
    else:
        axis = 1
    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean),1]),\
                                  time_dim  ,axis=axis )
    corrected_data = X.data - baseline_mean_mat
    return corrected_data
    
    