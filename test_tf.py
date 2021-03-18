#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:43:24 2021

@author: sr05
"""
import numpy as np
import tensorflow as tf

data = np.arange(0, 100)
input_data = data[:-10]
targets = data[10:]
dataset = tf.keras.preprocessing.timeseries_dataset_from_array(input_data, targets,
                                                               sequence_length=10,
                                                               sampling_rate=1,
                                                               sequence_stride=10,
                                                               shuffle=False)
for batch in dataset:
    inputs, targets = batch
    
    assert np.array_equal(inputs[0], data[:10])  # First sequence: steps [0-9]
    # Corresponding target: step 10
    assert np.array_equal(targets[0], data[10])
    break
