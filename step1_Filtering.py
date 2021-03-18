
"""
Created on Thu Mar  5 19:16:48 2020

@author: sr05
"""

import mne
import os
import numpy as np
# from matplotlib import pyplot as plt

#import scipy.io as scio
"""
#****************************************************************************#
#                                 Filtering Data                               #
#****************************************************************************#
"""


main_path = '/imaging/rf02/Semnet/'
data_path = '/imaging/rf02/Semnet/'	# where subdirs for MEG data are
new_path = '/imaging/sr05/SemNet/SemNetData'



list_all =  ['/meg16_0030/160216/',#0
            '/meg16_0032/160218/', #1
            '/meg16_0034/160219/', #2
            '/meg16_0035/160222/', #3
            '/meg16_0042/160229/', #4
            '/meg16_0045/160303/', #5
            '/meg16_0052/160310/', #6
            '/meg16_0056/160314/', #7
            '/meg16_0069/160405/', #8 
            '/meg16_0070/160407/', #9
            '/meg16_0072/160408/', #10
            '/meg16_0073/160411/', #11
            '/meg16_0075/160411/', #12
            '/meg16_0078/160414/', #13
            '/meg16_0082/160418/', #14
            '/meg16_0086/160422/', #15
            '/meg16_0097/160512/', #16 
            '/meg16_0122/160707/', #17
            '/meg16_0125/160712/', #18
            ]
bad_channels_fruit = {'meg16_0030': ['EEG034','EEG026', 'EEG028', 'EEG038', 
                                     'EEG048','EEG049', 'EEG056','EEG027',
                                     'EEG037','EEG045'],
                     'meg16_0032': ['EEG045','EEG002', 'EEG008', 'EEG004',
                                    'EEG005'],
                     'meg16_0034': ['EEG059', 'EEG074', 'EEG073', 'EEG072', 
                                    'EEG071', 'EEG066', 'EEG008', 'EEG050', 
                                    'EEG007', 'EEG016', 'EEG025', 'EEG002', 
                                    'EEG028', 'EEG056', 'EEG008', 'EEG007',
                                    'EEG005'],
                     'meg16_0035': ['EEG071','EEG069','EEG008','EEG016',
                                    'EEG002','EEG005','EEG068','EEG003', 
                                    'EEG065','EEG072','EEG001','EEG017',
                                    'EEG073','EEG004','EEG018','EEG002', 
                                    'EEG074','EEG070','EEG067','EEG015',
                                    ],
                     'meg16_0042': ['EEG055', 'EEG072', 'EEG071', 'EEG002',
                                    'EEG004', 'EEG005', 'EEG006', 'EEG007',
                                    'EEG008', 'EEG048', 'EEG027', 'EEG049',
                                    'EEG009', 'EEG010', 'EEG015', 'EEG016',
                                    'EEG017', 'EEG018', 'EEG019', 'EEG020',
                                    'EEG021', 'EEG025', 'EEG026', 'EEG028', 
                                    'EEG030', 'EEG031', 'EEG032', 'EEG036', 
                                    'EEG037', 'EEG038', 'EEG041', 'EEG042',
                                    'EEG047'],
                     'meg16_0045': ['EEG039', 'EEG058', 'EEG020', 'EEG042',
                                    'EEG059', 'EEG060', 'EEG070', 'EEG047', 
                                    'EEG048', 'EEG041', 'EEG058', 'EEG069', 
                                    'EEG070', 'EEG073', 'EEG068', 'EEG006',
                                    'EEG013', 'EEG023',],
                     'meg16_0052': ['EEG004', 'EEG045', 'EEG022', 'EEG024',
                                    'EEG046'],
                     'meg16_0056': ['EEG001', 'EEG002', 'EEG037', 'EEG007',
                                    'EEG008', 'EEG005', 'EEG069', 'EEG073',
                                    'EEG004', 'EEG009', 'EEG010', 'EEG011', 
                                    'EEG016', 'EEG017', 'EEG019', 'EEG020', 
                                    'EEG021', 'EEG026', 'EEG028', 'EEG030', 
                                    'EEG031', 'EEG037', 'EEG038', 'EEG048',
                                    'EEG049'],
                     'meg16_0069': ['EEG043', 'EEG057', 'EEG046', 'EEG002',
                                    'EEG004', 'EEG007', 'EEG008', 'EEG005', 
                                    'EEG006', 'EEG010'],
                     'meg16_0070': ['EEG043', 'EEG058', 'EEG047', 'EEG054', 
                                    'EEG046', 'EEG004'],
                     'meg16_0072': ['EEG066', 'EEG039','EEG074','EEG039',
                                    'EEG010', 'EEG016', 'EEG027',],
                     'meg16_0073': ['EEG071', 'EEG072', 'EEG002', 'EEG004',
                                    'EEG007', 'EEG008', 'EEG036', 'EEG005'],
                     'meg16_0075': ['EEG073', 'EEG071', 'EEG068', 'EEG072',
                                    'EEG028', 'EEG048', 'MEG1421','EEG018',
                                    'EEG001', 'EEG009', 'EEG016', 'EEG019',
                                    'EEG020', 'EEG021', 'EEG025', 'EEG026', 
                                    'EEG027', 'EEG030', 'EEG031', 'EEG032',
                                    'EEG036', 'EEG038', 'EEG042', 'EEG043', 
                                    'EEG047', 'EEG010', 'EEG011', 'EEG015',
                                    'EEG017', 'EEG037', 'EEG004'],
                     'meg16_0078': ['EEG039', 'EEG059', 'EEG074', 'EEG070',
                                    'EEG072','MEG2321', 'EEG068', 'EEG004',
                                    'EEG009', 'EEG029','MEG1331','EEG006',
                                    'EEG051', 'EEG008', 'EEG002', 'EEG005',
                                    'EEG015', 'EEG007', 'EEG016', 'EEG017',
                                    'EEG027', 'EEG038', 'EEG049', 'EEG050',
                                    'EEG060',
                                    'EEG010', 'EEG028', 'EEG021', 'EEG018',
                                    'EEG019', 'EEG020', 'EEG037', 'EEG031',
                                    'EEG048', 'EEG053', 'EEG046', 'EEG047',
                                    'EEG071', 'EEG073', 'EEG069'],
                     'meg16_0082': ['EEG057', 'EEG038', 'EEG043', 'EEG047',
                                    'EEG004', 'EEG008', 'EEG009', 'EEG010',
                                    'EEG011', 'EEG017', 'EEG019', 'EEG020',
                                    'EEG021', 'EEG026', 'EEG030', 'EEG042',
                                    'EEG016', 'EEG038', 'EEG015', 'EEG005',
                                    'EEG003', 'EEG004', 'EEG007'],
                     'meg16_0086': ['EEG071', 'EEG068','EEG025','EEG038',
                                    'EEG039', 'EEG070','EEG052','EEG021',
                                    'EEG002', 'EEG008'],
                     'meg16_0097': ['EEG013', 'EEG023', 'EEG072', 'EEG073',
                                    'EEG009', 'EEG010', 'EEG017', 'EEG019',
                                    'EEG020', 'EEG021', 'EEG025', 'EEG026', 
                                    'EEG027', 'EEG030', 'EEG032', 'EEG038',
                                    'EEG009', 'EEG003', 'EEG004', 'EEG006',
                                    'EEG001'],
                     'meg16_0122': ['EEG067', 'EEG047', ],
                     'meg16_0125': ['EEG035', 'EEG041']} 

bad_channels_milk = {'meg16_0030': ['EEG034','EEG037','EEG045','EEG028'],
                     'meg16_0032': ['EEG045', 'EEG071'],
                     'meg16_0034': ['EEG059','EEG073','EEG072','EEG066',
                                    'EEG071','EEG039','EEG050','EEG002', 
                                    'EEG004','EEG005','EEG006','EEG007',
                                    'EEG008','EEG016','EEG009','EEG028',
                                    'EEG068','EEG067','EEG069','EEG056',
                                    'EEG003','EEG039','EEG050','EEG074'],
                     'meg16_0035': ['EEG069','EEG071','EEG067','EEG003', 
                                    'EEG016','EEG065','EEG008','EEG005',
                                    'EEG017','EEG002','EEG004','EEG049',
                                    'EEG072','EEG073','EEG068','EEG065',
                                    'EEG002'],
                     'meg16_0042': ['EEG071', 'EEG072', 'EEG073', 'EEG002',
                                    'EEG005', 'EEG006', 'EEG007', 'EEG008',
                                    'EEG015', 'EEG016', 'EEG045', 'EEG016',
                                    'EEG004', 'EEG009', 'EEG010', 'EEG017', 
                                    'EEG018', 'EEG019', 'EEG020', 'EEG021', 
                                    'EEG025', 'EEG026', 'EEG027', 'EEG028', 
                                    'EEG030', 'EEG031', 'EEG032', 'EEG036', 
                                    'EEG037', 'EEG038', 'EEG042', 'EEG047', 
                                    'EEG048', 'EEG067', 'EEG055', 'EEG011'],
                     'meg16_0045': ['EEG039', 'EEG001', 'EEG011', 'EEG012',
                                    'EEG070', 'EEG073', 'EEG047', 'EEG053',
                                    'EEG068', 'EEG069', 'EEG003', 'EEG005', 
                                    'EEG006', 'EEG013', 'EEG014', 'EEG015',
                                    'EEG058', 'EEG059', 'EEG060', 'EEG065', 
                                    'EEG070', 'EEG010', 'EEG020', 'EEG021',
                                    'EEG022', 'EEG030', 'EEG031', 'EEG040',
                                    'EEG041', 'EEG042', 'EEG053', 'EEG054', 
                                    'EEG071', 'EEG047', 'EEG048', 'EEG057', 
                                    'EEG024', 'EEG025', 'EEG049', 'EEG067',
                                    'EEG004', 'EEG009', 'EEG018', 'EEG019',
                                    'EEG043', 'EEG052', 'EEG010', 'EEG036'],
                     'meg16_0052': ['EEG024', 'EEG045'],
                     'meg16_0056': ['EEG071','EEG002', 'EEG008', 'EEG017',
                                    'EEG003', 'EEG072', 'EEG074', 'EEG007',
                                    'EEG016', 'EEG026'],
                     'meg16_0069': ['EEG045', 'EEG043', 'EEG056', 'EEG046',
                                    'EEG048', 'MEG1811','MEG1612', 'MEG1622',
                                    'MEG1623', 'MEG1632', 'MEG1812'],
                     'meg16_0070': ['EEG058', 'EEG072', 'EEG071', 'EEG053',
                                    'EEG047', 'EEG046', 'EEG054', 'EEG034',
                                    'EEG001', 'EEG002', 'EEG003', 'EEG004', 
                                    'EEG005', 'EEG007', 'EEG008', 'EEG011',
                                    'EEG013', 'EEG014', 'EEG016', 'EEG018', 
                                    'EEG019', 'EEG021', 'EEG022', 'EEG023', 
                                    'EEG024', 'EEG026', 'EEG027', 'EEG029', 
                                    'EEG030', 'EEG031', 'EEG032', 'EEG033', 
                                    'EEG037', 'EEG038', 'EEG039', 'EEG040', 
                                    'EEG041', 'EEG042', 'EEG043', 'EEG044', 
                                    'EEG050', 'EEG052', 'EEG054', 'EEG055', 
                                    'EEG066', 'EEG067', 'EEG028'],
                     'meg16_0072': ['EEG039', 'EEG066', 'EEG071', 'EEG004',
                                    'EEG070', 'EEG004'],
                     'meg16_0073': ['EEG071', 'EEG072', 'EEG002', 'EEG007', 
                                    'EEG008', 'EEG074', 'EEG011', 'EEG004'],
                     'meg16_0075': ['EEG002', 'EEG073', 'EEG071', 'EEG072', 
                                    'EEG068', 'EEG051', 'EEG009', 'EEG016', 
                                    'EEG017', 'EEG019', 'EEG020', 'EEG021', 
                                    'EEG048', 'EEG009', 'EEG025', 'EEG026',
                                    'EEG027', 'EEG030', 'EEG036', 'EEG038', 
                                    'EEG042'],
                     'meg16_0078': ['EEG039', 'EEG074', 'EEG070', 'EEG069',
                                    'EEG002', 'EEG006', 'EEG051', 'EEG011',
                                    'EEG021', 'EEG035', 'EEG049', 'EEG004',
                                    'EEG001', 'EEG007', 'EEG008', 'EEG027',
                                    'EEG028', 'EEG035', 'EEG049', 'EEG004',
                                    'EEG038', 'EEG037', 'EEG036', 'EEG050',
                                    'EEG059', 'EEG060', 'EEG029',
                                    'EEG005', 'EEG003', 'EEG068', 'EEG073'],
                     'meg16_0082': ['EEG058', 'EEG057', 'EEG047', 'EEG046',
                                    'EEG038', 'EEG049', 'EEG008', 'EEG030',
                                    'EEG043', 'EEG045', 'EEG002', 'EEG004', 
                                    'EEG005', 'EEG007','EEG001', 'EEG003',
                                    'EEG025', 'EEG026','EEG015', 'EEG016',
                                    'EEG009', 'EEG010',], 
                     'meg16_0086': ['EEG034', 'EEG039', 'EEG071','EEG007',
                                    'EEG002', 'EEG004', 'EEG008','EEG070',
                                    'EEG065', 'EEG017','EEG041'],
                     'meg16_0097': ['EEG044', 'EEG001', 'EEG011', 'EEG012',
                                    'EEG019', 'EEG021', 'EEG024', 'EEG034', 
                                    'EEG045', 'EEG046', 'EEG051', 'EEG054', 
                                    'EEG056', 'EEG057', 'EEG067', 'EEG068', 
                                    'EEG069', 'EEG071', 'EEG072', 'EEG073',
                                    'EEG009', 'EEG030', 'EEG055','MEG0211', 
                                    'MEG0221','EEG003', 'EEG006', 'EEG004',
                                    'EEG009', 'EEG001'],
                     'meg16_0122': ['EEG067'],
                     'meg16_0125': ['EEG035','EEG041','EEG045']}


bad_channels_odour = {'meg16_0030': ['EEG008', 'EEG028', 'EEG034', 'EEG002', 
                                     'EEG071', 'EEG074', 'EEG004', 'EEG007',
                                     'EEG067', 'EEG038', 'EEG007', 'EEG016', 
                                     'EEG017', 'EEG005', 'EEG006'],
                     'meg16_0032': ['EEG045','EEG002','EEG008'],
                     'meg16_0034': ['EEG071','EEG066','EEG069','EEG025',
                                    'EEG050','EEG028','EEG007','EEG056', 
                                    'EEG068','EEG010','EEG016','EEG001',
                                    'EEG003','EEG017','EEG067','EEG025',
                                    'EEG074','EEG049','EEG003','EEG059',
                                    'EEG039','EEG072','EEG073'],
                     'meg16_0035': ['EEG057','EEG069','EEG071','EEG067',
                                    'EEG065','EEG072','EEG073'],
                     'meg16_0042': ['EEG071','EEG002','EEG030', 'EEG031',
                                    'EEG072','EEG009','EEG004','EEG010',
                                    'EEG017'],
                     'meg16_0045': ['EEG034','EEG039','EEG058','EEG014',
                                    'EEG036','EEG037','EEG039','EEG047',
                                    'EEG048','EEG070','EEG071','EEG010',
                                    'EEG019','EEG020','EEG030','EEG042',
                                    'EEG024','EEG059','EEG069','EEG069', 
                                    'EEG073','EEG015','EEG024','EEG025', 
                                    'EEG038','EEG039','EEG047','EEG048',
                                    'EEG049','EEG059','EEG071','EEG009',
                                    'EEG032','EEG042','EEG007','EEG013',
                                    'EEG025','EEG060','EEG001','EEG066'],
                     'meg16_0052': ['EEG039','EEG022','EEG024'],
                     'meg16_0056': ['EEG001','EEG002', 'EEG007', 'EEG008',
                                    'EEG004'],
                     'meg16_0069': ['EEG043', 'EEG046', 'EEG057', 'EEG068', 
                                    'EEG071','MEG1621'],
                     'meg16_0070': ['EEG072', 'EEG053', 'EEG058', 'EEG046',
                                    'EEG047', 'EEG054', 'EEG008', 'EEG009', 
                                    'EEG043'],
                     'meg16_0072': ['EEG066', 'EEG039', 'EEG068', 'EEG007',
                                    'EEG020', 'EEG027', 'EEG045', 'EEG074',
                                    'EEG071', 'EEG072', 'EEG001', 'EEG069', 
                                    'EEG070', 'EEG002', 'EEG004', 'EEG005'],
                     'meg16_0073': ['EEG072', 'EEG071', 'EEG008', 'EEG004',
                                    'EEG011'],
                     'meg16_0075': ['EEG073', 'EEG068', 'EEG071', 'EEG072',
                                    'MEG1421','EEG009', 'EEG016', 'EEG020',
                                    'MEG0111','MEG0141','MEG1431','EEG003', 
                                    'EEG004', 'EEG009', 'EEG010', 'EEG011', 
                                    'EEG016', 'EEG017', 'EEG018', 'EEG019',
                                    'EEG020', 'EEG021', 'EEG025', 'EEG026', 
                                    'EEG027', 'EEG030', 'EEG031', 'EEG032', 
                                    'EEG036', 'EEG038', 'EEG042', 'EEG048',
                                    'EEG065', 'MEG0121', 'MEG0131', 'MEG0141',
                                    'MEG1421', 'MEG1431', 'MEG1541', 'MEG2621'], 
                     'meg16_0078': ['EEG039', 'EEG029', 'EEG074', 'EEG059', 
                                    'EEG073', 'EEG072', 'EEG002', 'EEG004', 
                                    'EEG005', 'EEG006', 'EEG007', 'EEG008',
                                    'EEG016', 'EEG028', 'EEG049', 'EEG003', 
                                    'EEG068', 'EEG017', 'EEG015', 'EEG027',
                                    'EEG025', 'EEG026', 'EEG050', 'EEG060',
                                    'EEG009', 'EEG010', 'EEG038', 'EEG034'],
                     'meg16_0082': ['EEG057', 'EEG047', 'EEG010', 'EEG011', 
                                    'EEG017', 'EEG019', 'EEG020', 'EEG021', 
                                    'EEG026', 'EEG027', 'EEG030', 'EEG038',
                                    'EEG018', 'EEG028', 'EEG041', 'EEG051',
                                    'EEG015', 'EEG016', 'EEG009', 'EEG008',
                                    'EEG004', 'EEG005', 'EEG002'],
                     'meg16_0086': ['EEG071', 'EEG068', 'EEG072', 'EEG073',
                                    'EEG074', 'EEG049', 'EEG025', 'EEG047',
                                    'EEG051', 'EEG039', 'EEG052'],
                     'meg16_0097': ['EEG013', 'EEG045', 'EEG023', 'EEG056',
                                    'EEG068', 'EEG069', 'EEG009', 'EEG003',
                                    'EEG004', 'EEG006', 'EEG008', 'EEG001'], 
                     'meg16_0122': ['EEG067'],
                     'meg16_0125': ['EEG035','EEG045']}


lfreq=0.1
h_freq=45

# for i in np.arange(14, len(list_all)):
for i in [13]:

    print('***********************Participant : ', i+1)
    meg = list_all[i]

    raw_fname_fruit = main_path + meg + 'block_fruit_tsss_raw.fif'
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)#, preload=True
    raw_fname_milk = main_path + meg + 'block_milk_tsss_raw.fif'
    raw_milk = mne.io.Raw(raw_fname_milk , preload=True)#, preload=True
    raw_fname_odour = main_path + meg + 'block_odour_tsss_raw.fif'
    raw_odour  = mne.io.Raw(raw_fname_odour, preload=True)#, preload=True
    

    print('***********************Participant : ', i+1)
     
    raw_fruit.info['bads'] = bad_channels_fruit[meg[1:11]]
    raw_milk.info['bads']  = bad_channels_milk[meg[1:11]]
    raw_odour.info['bads'] = bad_channels_odour[meg[1:11]]

    print('***********************Participant : ', i+1)
    raw_fruit.interpolate_bads(reset_bads = True , mode = 'accurate')
    raw_milk.interpolate_bads(reset_bads = True , mode = 'accurate')
    raw_odour.interpolate_bads(reset_bads = True , mode = 'accurate')
    print('***********************Participant : ', i+1)
    
    raw_fruit.set_eeg_reference( ref_channels = 'average')
    raw_milk.set_eeg_reference( ref_channels = 'average')
    raw_odour.set_eeg_reference( ref_channels = 'average')
    
    print('***********************Participant : ', i+1)
    picks_fruit = mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_milk = mne.pick_types(raw_milk.info, meg=True, eeg=True, eog=False,
            stim=False )
    picks_odour = mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=False,
            stim=False )
#
#
#
    raw_fruit_notch = raw_fruit.copy().notch_filter(freqs=50 , picks = picks_fruit)
    raw_milk_notch  = raw_milk.copy().notch_filter(freqs=50 , picks = picks_milk)
    raw_odour_notch = raw_odour.copy().notch_filter(freqs=50 , picks = picks_odour)


    print('***********************Participant : ', i+1)
    raw_fruit_notch_BPF = raw_fruit_notch.copy().filter(l_freq=lfreq, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_fruit)
    raw_milk_notch_BPF =  raw_milk_notch.copy().filter(l_freq=lfreq, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_milk)
    raw_odour_notch_BPF = raw_odour_notch.copy().filter(l_freq=lfreq, 
                 h_freq=h_freq, fir_design='firwin' , picks = picks_odour)
   
    print('***********************Participant : ', i+1)

    if not os.path.isdir(new_path + meg):
        os.makedirs(new_path + meg)

    
    out_name_fruit = new_path + meg + 'block_fruit_tsss_notch_BPF0.1_45_raw.fif'
    out_name_milk  = new_path + meg + 'block_milk_tsss_notch_BPF0.1_45_raw.fif'
    out_name_odour = new_path + meg + 'block_odour_tsss_notch_BPF0.1_45_raw.fif'

	
             
    raw_fruit_notch_BPF.save(out_name_fruit, overwrite=True)
    raw_milk_notch_BPF.save(out_name_milk, overwrite=True)
    raw_odour_notch_BPF.save(out_name_odour, overwrite=True)


