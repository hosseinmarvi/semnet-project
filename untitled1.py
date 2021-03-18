#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 21:10:48 2021

@author: sr05
"""
import numpy as np


def mask_function(X, cut_off):

    r, c = X.shape()
    for i in np.arange(0, r):
        for j in np.arange(0, c):
            if X[i, j] < cut_off:
                X[i, j] = 0
