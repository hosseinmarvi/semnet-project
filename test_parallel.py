#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:59:38 2021

@author: sr05
"""
import multiprocessing
import collections
import time

    
from math import sqrt
from joblib import Parallel, delayed
import numpy as np
import timeit

T=np.zeros([100,3])

def convolve_random(size):
    return np.convolve(np.random.random_sample(size), np.random.random_sample(size))
# 
for n in np.arange(0,100):
    print(n)
    s=time.time()
    Parallel(n_jobs=8)(delayed(convolve_random)(40000+i*1000) for i in range(8))
    e=time.time()
    T[n,0]=e-s
      
    s=time.time()
    Parallel(n_jobs=2)(delayed(convolve_random)(40000+i*1000) for i in range(8))
    e=time.time()
    T[n,1]=e-s
    
    
    s=time.time()
    [convolve_random(40000+i*1000) for i in range(8)]
    e=time.time()
    T[n,2]=e-s
