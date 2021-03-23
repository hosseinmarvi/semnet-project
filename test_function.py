#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:09:37 2020

@author: sr05
"""
import sn_config as C
import numpy as np
import os
import pickle
import time
import sys
s=time.time()

def test_function(size):
    s=time.time()
    stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/a_conv_'+str(size)+'.json'
    a_conv=np.convolve(np.random.random_sample(size),np.random.random_sample(size))

    with open(stc_SD_file_name, "wb") as fp:   #Pickling
        pickle.dump(a_conv, fp)
    e=time.time()
    print(e-s)


if len(sys.argv) == 1:

    sbj_ids = np.arange(1, len(C.subjects))*1000 

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]    

for s in sbj_ids:
    test_function(s)
    
    
# sbj_ids=np.arange(1,len(C.subjects))*1000    
# for s in sbj_ids:
#         test_function(s)
#     # [x,y]=test_function(s)
        

# size=12*1000        
# stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/a_conv_'+str(size)+'.json'
        
# with open(stc_SD_file_name, "rb") as fp:   # Unpickling
#     a = pickle.load(fp)
