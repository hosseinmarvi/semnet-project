#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:23:18 2020

@author: sr05
"""

bad_channels_fruit = {'meg16_0030': ['EEG034'],
                     'meg16_0032': ['EEG045','EEG002', 'EEG008', 'EEG004',
                     'EEG005'],
                     'meg16_0034': ['EEG059', 'EEG074', 'EEG073', 'EEG072', 'EEG071', 'EEG066'],
                     'meg16_0035': ['EEG071', 'EEG069'],
                     'meg16_0042': ['EEG055', 'EEG072', 'EEG071'],
                     'meg16_0045': ['EEG039', 'EEG058'],
                     'meg16_0052': [],
                     'meg16_0056': ['EEG001'],
                     'meg16_0069': ['EEG043', 'EEG057', 'EEG046'],
                     'meg16_0070': ['EEG043', 'EEG058', 'EEG047', 'EEG054', 'EEG046'],
                     'meg16_0072': ['EEG066', 'EEG039'],
                     'meg16_0073': ['EEG071', 'EEG072'],
                     'meg16_0075': ['EEG073', 'EEG071', 'EEG068', 'EEG072'],
                     'meg16_0078': ['EEG039', 'EEG059', 'EEG074', 'EEG070', 'EEG072'],
                     'meg16_0082': ['EEG057'],
                     'meg16_0086': ['EEG071', 'EEG068'],
                     'meg16_0097': ['EEG013'],
                     'meg16_0122': ['EEG067'],
                     'meg16_0125': ['EEG035']} 



bad_channels_odour = {'meg16_0030': ['EEG008', 'EEG028', 'EEG034'],
                     'meg16_0032': ['EEG045','EEG002', 'EEG008'],
                     'meg16_0034': ['EEG071', 'EEG066', 'EEG069'],
                     'meg16_0035': ['EEG057', 'EEG069', 'EEG071', 'EEG067'],
                     'meg16_0042': ['EEG071'],
                     'meg16_0045': ['EEG034', 'EEG039', 'EEG058'],
                     'meg16_0052': [],
                     'meg16_0056': ['EEG001'],
                     'meg16_0069': ['EEG043', 'EEG046', 'EEG057', 'EEG068', 'EEG071'],
                     'meg16_0070': ['EEG072', 'EEG053', 'EEG058', 'EEG046', 'EEG047', 'EEG054'],
                     'meg16_0072': ['EEG066', 'EEG039'],
                     'meg16_0073': ['EEG072', 'EEG071'],
                     'meg16_0075': ['EEG073', 'EEG068', 'EEG071', 'EEG072'], 
                     'meg16_0078': ['EEG039', 'EEG029', 'EEG074', 'EEG059', 'EEG073', 'EEG072'],
                     'meg16_0082': ['EEG057', 'EEG047'],
                     'meg16_0086': ['EEG071', 'EEG068', 'EEG072', 'EEG073', 'EEG074'],
                     'meg16_0097': ['EEG013'], 
                     'meg16_0122': ['EEG067'],
                     'meg16_0125': ['EEG035']}

bad_channels_milk = {'meg16_0030': ['EEG034'],
                     'meg16_0032': ['EEG045', 'EEG071'],
                     'meg16_0034': ['EEG059', 'EEG073', 'EEG072', 'EEG066', 'EEG071'],
                     'meg16_0035': ['EEG069', 'EEG071', 'EEG067'],
                     'meg16_0042': ['EEG071', 'EEG072', 'EEG073'],
                     'meg16_0045': ['EEG039'],
                     'meg16_0052': [],
                     'meg16_0056': ['EEG071'],
                     'meg16_0069': ['EEG045', 'EEG043', 'EEG056', 'EEG046', 'EEG048'],
                     'meg16_0070': ['EEG058', 'EEG072', 'EEG071', 'EEG053', 'EEG047', 'EEG046', 'EEG054', 'EEG034'],
                     'meg16_0072': ['EEG039', 'EEG066', 'EEG071'],
                     'meg16_0073': ['EEG071', 'EEG072'],
                     'meg16_0075': ['EEG002', 'EEG073', 'EEG071', 'EEG072', 'EEG068'],
                     'meg16_0078': ['EEG039', 'EEG074', 'EEG070', 'EEG069'],
                     'meg16_0082': ['EEG058', 'EEG057', 'EEG047', 'EEG046'], 
                     'meg16_0086': ['EEG034', 'EEG039', 'EEG071'],
                     'meg16_0097': [],
                     'meg16_0122': ['EEG067'],
                     'meg16_0125': ['EEG035']}

bad_channels_LD = { 'meg16_0030': ['EEG034', 'EEG053', 'EEG046', 'EEG067', 'EEG070', 'EEG071'],
                 'meg16_0032': ['EEG048', 'EEG045', 'EEG027'],
                 'meg16_0034': ['EEG010', 'EEG072', 'EEG071', 'EEG069', 'EEG073', 'EEG074', 
                                'EEG060', 'EEG059','EEG066'],
                 'meg16_0035': ['EEG071', 'EEG069', 'EEG068', 'EEG070', 'EEG073'],
                 'meg16_0042': ['EEG072', 'EEG071', 'EEG074', 'EEG073', 'EEG070'],
                 'meg16_0045': ['EEG034', 'EEG058', 'EEG071', 'EEG066'],
                 'meg16_0052': ['EEG067'],
                 'meg16_0056': ['EEG069', 'EEG074'],
                 'meg16_0069': ['EEG043', 'EEG057', 'EEG047', 'EEG046'],
                 'meg16_0070': ['EEG019', 'EEG053', 'EEG072', 'EEG071'],
                 'meg16_0072': ['EEG066'],
                 'meg16_0073': ['EEG072', 'EEG071'],
                 'meg16_0075': ['EEG073', 'EEG071', 'EEG068', 'EEG072'],
                 'meg16_0078': ['EEG027', 'EEG016', 'EEG074', 'EEG073'],
                 'meg16_0082': ['EEG047'],
                 'meg16_0086': ['EEG034', 'EEG071'],
                 'meg16_0097': ['EEG013'],
                 'meg16_0122': ['EEG067'],
                 'meg16_0125': ['EEG035', 'EEG032', 'EEG069', 'EEG071', 'EEG067']}

