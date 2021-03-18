#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:08:48 2021

@author: sr05
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import RidgeCV
from sklearn import preprocessing
import numpy.matlib
from sklearn.model_selection import (train_test_split,KFold)
from joblib import Parallel, delayed
import math

std_pow=[-2,-1.5,-1,-.5,0,.5,1,1.5,2]
sig_pow_ave=np.zeros([len(std_pow),1])
repeat=1000
for s in np.arange(0,len(std_pow)):
    sig_pow=np.zeros([repeat,1])
    for i in range(repeat):
        print(s, i)
        sig=np.random.normal(0,10**std_pow[s],[200,1])
        sig_pow[i,0]=np.mean(sig**2)
    sig_pow_ave[s,0]=np.mean(sig_pow)
        