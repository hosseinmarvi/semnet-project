#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:03:40 2020

@author: sr05
"""

raw_fruit.plot_psd(tmin=0, tmax=None)
raw_milk.plot_psd(tmin=0, tmax=None)
raw_odour.plot_psd(tmin=0, tmax=None)

raw_fruit_notch_BPF1.plot_psd(tmin=0, tmax=None, fmax=80)
raw_milk_notch_BPF1.plot_psd(tmin=0, tmax=None, fmax=80)
raw_odour_notch_BPF1.plot_psd(tmin=0, tmax=None, fmax=80)

raw_fruit_notch_BPF01.plot_psd(tmin=0, tmax=None, fmax=80)
raw_milk_notch_BPF01.plot_psd(tmin=0, tmax=None, fmax=80)
raw_odour_notch_BPF01.plot_psd(tmin=0, tmax=None, fmax=80)

