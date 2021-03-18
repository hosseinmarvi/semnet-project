#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:59:50 2020

@author: sr05
"""
import time
import numpy as np
from joblib import Parallel,delayed

s=time.time()
def my_function(size):
    # return [id, id**2 ]
    return np.convolve(np.random.random_sample(size),np.random.random_sample(size))

# s=time.time()

# x=my_function(40000)
# e=time.time()

# x=[my_function(40000+i*1000) for i in range(8)]

# e=time.time()

# print(e-s)


x=Parallel(n_jobs=8)(delayed(my_function)(40000+i*1000) for i in range(8))
e=time.time()

print(e-s)
