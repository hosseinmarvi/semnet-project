#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:12:04 2021

@author: sr05
"""
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (cross_validate, KFold)
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score



# plt.close('all')
