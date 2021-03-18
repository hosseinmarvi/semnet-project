
"""
Created on Mon Apr  6 21:56:26 2020

@author: sr05
"""



import mne
import numpy as np


main_path = '/imaging/rf02/Semnet/'	# where subdirs for MEG data are


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


tmin, tmax = -0.3, 0.6
stim_delay = 0.034 # delay in s
category_code = np.arange(1,6)
stimulus_code = np.arange(41,91)
FP_F = list()
FP_M = list()
FP_O = list()
WP = list()


FN_F = list()
FN_M = list()
FN_O = list()
PW = list()

C_F_v = list()
C_F_h = list()
C_F_ha = list()
C_F_n = list()
C_F_e = list()
C_F_errorFP = list()

C_M_v = list()
C_M_h = list()
C_M_ha = list()
C_M_n = list()
C_M_e = list()
C_M_errorFP = list()


C_O_v = list()
C_O_h = list()
C_O_ha = list()
C_O_n = list()
C_O_e = list()
C_O_errorFP = list()

C_L_v = list()
C_L_h = list()
C_L_ha = list()
C_L_n = list()
C_L_e = list()
C_L_errorFP = list()


tC_F_v = list()
tC_F_h = list()
tC_F_ha = list()
tC_F_n = list()
tC_F_e = list()
tC_F_errorFP = list()

tC_M_v = list()
tC_M_h = list()
tC_M_ha = list()
tC_M_n = list()
tC_M_e = list()
tC_M_errorFP = list()


tC_O_v = list()
tC_O_h = list()
tC_O_ha = list()
tC_O_n = list()
tC_O_e = list()
tC_O_errorFP = list()

tC_L_v = list()
tC_L_h = list()
tC_L_ha = list()
tC_L_n = list()
tC_L_e = list()



for i in np.arange(0,len(list_all)):
    print('Participant : ', i)
    meg = list_all[i]
    

    # raw_fname_fruit = main_path + meg + 'block_fruit_tsss_raw.fif'
    # raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    # raw_fname_milk = main_path + meg + 'block_milk_tsss_raw.fif'
    # raw_milk = mne.io.Raw(raw_fname_milk , preload=True)
    # raw_fname_odour = main_path + meg + 'block_odour_tsss_raw.fif'
    # raw_odour  = mne.io.Raw(raw_fname_odour, preload=True)
    raw_fname_lexical = main_path + meg + 'block_LD_tsss_raw.fif'
    raw_lexical = mne.io.Raw(raw_fname_lexical, preload=True)
    

    # picks_fruit = mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=True,
    #           stim=False)
    # picks_milk = mne.pick_types(raw_milk.info, meg=True, eeg=True, eog=True,
    #           stim=False)
    # picks_odour = mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=True,
    #           stim=False)      
    picks_lexical = mne.pick_types(raw_lexical.info, meg=True, eeg=True, eog=True,
              stim=False) 
    
    
    # events_f = mne.find_events(raw_fruit, stim_channel='STI101',
    #           min_duration=0.001 , shortest_event=1)
    # events_m = mne.find_events(raw_milk, stim_channel='STI101',
    #           min_duration=0.001 , shortest_event=1) 
    # events_o = mne.find_events(raw_odour, stim_channel='STI101',
    #           min_duration=0.001 , shortest_event=1)   
    events_l = mne.find_events(raw_lexical, stim_channel='STI101',
              min_duration=0.001 , shortest_event=1) 
   

   
    # event_id_sd = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
    #             'pwordc': 6 ,'target': 8 }
    event_id_ld = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
                'pwordc': 6 , 'pworda': 7 , 'filler' :9}
#****************************************************************************#
#This section finds the number of all events 
#****************************************************************************#
        
    # epochs_f = mne.Epochs(raw_fruit, events_f, event_id_sd, tmin, tmax, 
    #         picks=picks_fruit, proj=True, baseline=(tmin, 0))
    
    # epochs_m = mne.Epochs(raw_milk, events_m, event_id_sd, tmin, tmax, 
    #         picks=picks_milk, proj=True, baseline=(tmin, 0))
   
    # epochs_o = mne.Epochs(raw_odour, events_o, event_id_sd, tmin, tmax, 
    #         picks=picks_odour, proj=True, baseline=(tmin, 0))
    
    epochs_l = mne.Epochs(raw_lexical, events_l, event_id_ld, tmin, tmax, 
            picks=picks_lexical, proj=True, baseline=(tmin, 0))
    # n=0
    # events_hear = np.zeros(1000)
    # for event in np.arange(0,len(events_f)):
    #     if events_f[event,2]==2:
    #         events_hear[n]=2
    #         events_hear[n+1]=events_f[event+1,2]
    #         n=n+2
        
        
# np.sort(events_f[:,2])
# for i in np.where(events_f[:,2]==8):
#         events_hear = events_f [i-1,2]    
    
    # C_F_visual = 0
    # C_F_hear = 0
    # C_F_hand = 0
    # C_F_neutral = 0
    # C_F_emotional = 0
    # C_F_FP = 0
    # # events_f_tesst= events_f.copy()
    # for evcnt in np.arange(0,events_f.shape[0]):    
    #     if (events_f[evcnt,2]==1 and events_f[evcnt+1,2] in stimulus_code):
    #        C_F_visual =  C_F_visual + 1
    #     elif events_f[evcnt,2]==1 and events_f[evcnt+1,2] not in stimulus_code:
    #         events_f[evcnt,2] = 9999
               
    #     if (events_f[evcnt,2]==2 and events_f[evcnt+1,2] in stimulus_code):
    #        C_F_hear =  C_F_hear + 1
    #     elif events_f[evcnt,2]==2 and events_f[evcnt+1,2] not in stimulus_code:
    #         events_f[evcnt,2] = 9999
            
    #     if (events_f[evcnt,2]==3 and events_f[evcnt+1,2] in stimulus_code):
    #        C_F_hand =  C_F_hand + 1
    #     elif events_f[evcnt,2]==3 and events_f[evcnt+1,2] not in stimulus_code:
    #         events_f[evcnt,2] = 9999
           
    #     if (events_f[evcnt,2]==4 and events_f[evcnt+1,2] in stimulus_code):
    #        C_F_neutral =  C_F_neutral + 1
    #     elif events_f[evcnt,2]==4 and events_f[evcnt+1,2] not in stimulus_code:
    #         events_f[evcnt,2] = 9999
           
    #     if (events_f[evcnt,2]==5 and events_f[evcnt+1,2] in stimulus_code):
    #        C_F_emotional =  C_F_emotional + 1   
    #     elif events_f[evcnt,2]==5 and events_f[evcnt+1,2] not in stimulus_code:
    #         events_f[evcnt,2] = 9999
           
    # C_M_visual = 0
    # C_M_hear = 0
    # C_M_hand = 0
    # C_M_neutral = 0
    # C_M_emotional = 0
    # C_M_FP = 0         
       
    # for evcnt in np.arange(0,events_m.shape[0]):    
    #     if (events_m[evcnt,2]==1 and events_m[evcnt+1,2] in stimulus_code):
    #        C_M_visual =  C_M_visual + 1
    #     elif events_m[evcnt,2]==1 and events_m[evcnt+1,2] not in stimulus_code:
    #         events_m[evcnt,2] = 9999
           
    #     if (events_m[evcnt,2]==2 and events_m[evcnt+1,2] in stimulus_code):
    #        C_M_hear =  C_M_hear + 1
    #     elif events_m[evcnt,2]==2 and events_m[evcnt+1,2] not in stimulus_code:
    #         events_m[evcnt,2] = 9999
            
    #     if (events_m[evcnt,2]==3 and events_m[evcnt+1,2] in stimulus_code):
    #        C_M_hand =  C_M_hand + 1
    #     elif events_m[evcnt,2]==3 and events_m[evcnt+1,2] not in stimulus_code:
    #         events_m[evcnt,2] = 9999
           
    #     if (events_m[evcnt,2]==4 and events_m[evcnt+1,2] in stimulus_code):
    #        C_M_neutral =  C_M_neutral + 1
    #     elif events_m[evcnt,2]==4 and events_m[evcnt+1,2] not in stimulus_code:
    #         events_m[evcnt,2] = 9999
            
    #     if (events_m[evcnt,2]==5 and events_m[evcnt+1,2] in stimulus_code):
    #        C_M_emotional =  C_M_emotional + 1 
    #     elif events_m[evcnt,2]==5 and events_m[evcnt+1,2] not in stimulus_code:
    #         events_m[evcnt,2] = 9999
 
    # C_O_visual = 0
    # C_O_hear = 0
    # C_O_hand = 0
    # C_O_neutral = 0
    # C_O_emotional = 0
    # C_O_FP = 0
    
    # for evcnt in np.arange(0,events_o.shape[0]):    
    #     if (events_o[evcnt,2]==1 and events_o[evcnt+1,2] in stimulus_code):
    #        C_O_visual =  C_O_visual + 1
    #     elif events_o[evcnt,2]==1 and events_o[evcnt+1,2] not in stimulus_code:
    #         events_o[evcnt,2] = 9999
            
    #     if (events_o[evcnt,2]==2 and events_o[evcnt+1,2] in stimulus_code):
    #        C_O_hear =  C_O_hear + 1
    #     elif events_o[evcnt,2]==2 and events_o[evcnt+1,2] not in stimulus_code:
    #         events_o[evcnt,2] = 9999
            
    #     if (events_o[evcnt,2]==3 and events_o[evcnt+1,2] in stimulus_code):
    #        C_O_hand =  C_O_hand + 1
    #     elif events_o[evcnt,2]==3 and events_o[evcnt+1,2] not in stimulus_code:
    #         events_o[evcnt,2] = 9999
            
    #     if (events_o[evcnt,2]==4 and events_o[evcnt+1,2] in stimulus_code):
    #        C_O_neutral =  C_O_neutral + 1
    #     elif events_o[evcnt,2]==4 and events_o[evcnt+1,2] not in stimulus_code:
    #         events_o[evcnt,2] = 9999
            
    #     if (events_o[evcnt,2]==5 and events_o[evcnt+1,2] in stimulus_code):
    #        C_O_emotional =  C_O_emotional + 1  
    #     elif events_o[evcnt,2]==5 and events_o[evcnt+1,2] not in stimulus_code:
    #         events_o[evcnt,2] = 9999
           
   
    C_L_visual = 0
    C_L_hear = 0
    C_L_hand = 0
    C_L_neutral = 0
    C_L_emotional = 0
    C_L_FP = 0 
    
    for evcnt in range(events_l.shape[0]):    
        if (events_l[evcnt,2]==1 and events_l[evcnt+1,2] in stimulus_code):
           C_L_visual =  C_L_visual + 1
        elif events_l[evcnt,2]==1 and events_l[evcnt+1,2] not in stimulus_code:
            events_l[evcnt,2] = 9999
            
        if (events_l[evcnt,2]==2 and events_l[evcnt+1,2] in stimulus_code):
           C_L_hear =  C_L_hear + 1
        elif events_l[evcnt,2]==2 and events_l[evcnt+1,2] not in stimulus_code:
            events_l[evcnt,2] = 9999
           
        if (events_l[evcnt,2]==3 and events_l[evcnt+1,2] in stimulus_code):
           C_L_hand =  C_L_hand + 1
        elif events_l[evcnt,2]==3 and events_l[evcnt+1,2] not in stimulus_code:
            events_l[evcnt,2] = 9999
           
        if (events_l[evcnt,2]==4 and events_l[evcnt+1,2] in stimulus_code):
           C_L_neutral =  C_L_neutral + 1
        elif events_l[evcnt,2]==4 and events_l[evcnt+1,2] not in stimulus_code:
            events_l[evcnt,2] = 9999
           
        if (events_l[evcnt,2]==5 and events_l[evcnt+1,2] in stimulus_code):
           C_L_emotional =  C_L_emotional + 1  
        elif events_l[evcnt,2]==5 and events_l[evcnt+1,2] not in stimulus_code:
            events_l[evcnt,2] = 9999
           
           
    # tC_F_v.append(C_F_visual)
    # tC_F_h.append(C_F_hear)
    # tC_F_ha.append(C_F_hand)
    # tC_F_n.append(C_F_neutral)
    # tC_F_e.append(C_F_emotional)
    
    # tC_M_v.append(C_M_visual)
    # tC_M_h.append(C_M_hear)
    # tC_M_ha.append(C_M_hand)
    # tC_M_n.append(C_M_neutral)
    # tC_M_e.append(C_M_emotional)
    
    # tC_O_v.append(C_O_visual)
    # tC_O_h.append(C_O_hear)
    # tC_O_ha.append(C_O_hand)
    # tC_O_n.append(C_O_neutral)
    # tC_O_e.append(C_O_emotional)
    
    tC_L_v.append(C_L_visual)
    tC_L_h.append(C_L_hear)
    tC_L_ha.append(C_L_hand)
    tC_L_n.append(C_L_neutral)
    tC_L_e.append(C_L_emotional)


    # events_fruit = events_f.copy()
    # events_milk = events_m.copy()
    # events_odour = events_o.copy()
    events_lexical = events_l.copy()   
    
#****************************************************************************#    
#This part has been written to identify those events (1,2,3,4,5) which 
#participants should have not press any bottun (False Positive: FP)
#****************************************************************************#
    # FP_fruit = 0
    # FN_fruit = 0
    # for evcnt in np.arange(0,events_fruit.shape[0]-2):    
    #     if (events_fruit[evcnt,2] in category_code  and events_fruit[evcnt+2,2]>100):
    #        events_fruit[evcnt,2]=7777
    #        FP_fruit = FP_fruit + 1
           
    #     elif(events_fruit[evcnt,2] == 8 and events_fruit[evcnt+2,2]<100):
    #         events_fruit[evcnt,2]=8888
    #         FN_fruit = FN_fruit + 1
           
           
    # FP_milk = 0
    # FN_milk = 0
    # for evcnt in np.arange(0,events_milk.shape[0]-2):    
    #     if (events_milk[evcnt,2] in category_code  and events_milk[evcnt+2,2]>100):
    #         events_milk[evcnt,2] = 7777
    #         FP_milk = FP_milk + 1
           
    #     elif(events_milk[evcnt,2] == 8 and events_milk[evcnt+2,2]<100):
    #         events_milk[evcnt,2]=8888
    #         FN_milk = FN_milk + 1
           
    # FP_odour = 0
    # FN_odour = 0
    # for evcnt in np.arange(0,events_odour.shape[0]-2):    
    #     if (events_odour[evcnt,2]in category_code  and  events_odour[evcnt+2,2]>100):
    #        events_odour[evcnt,2]=7777
    #        FP_odour = FP_odour + 1
           
    #     elif(events_odour[evcnt,2] == 8 and events_odour[evcnt+2,2]<100):
    #         events_odour[evcnt,2]=8888
    #         FN_odour = FN_odour + 1
           
           
                   
    WP_LD = 0
    PW_LD = 0
    for evcnt in np.arange(0,events_lexical.shape[0]-2):    
        if (events_lexical[evcnt,2] in category_code  and  events_lexical[evcnt+2,2]!=16384):
            events_lexical[evcnt,2] = 7777
            WP_LD = WP_LD + 1
           
        elif(events_lexical[evcnt,2] in np.array([6,7,9]) and events_lexical[evcnt+2,2]!=4096):
            events_lexical[evcnt,2] = 8888
            PW_LD = PW_LD + 1
           
    # FP_F.append(FP_fruit)
    # FP_M.append(FP_milk)
    # FP_O.append(FP_odour)
    WP.append(WP_LD)
    
    # FN_F.append(FN_fruit)
    # FN_M.append(FN_milk)
    # FN_O.append(FN_odour)
    PW.append(PW_LD)



#****************************************************************************#
#This section finds the number of those events (words) without any button press
#****************************************************************************#
    
    # epochs_fruit = mne.Epochs(raw_fruit, events_fruit, event_id_sd, tmin, tmax, 
    #         picks=picks_fruit, proj=True, baseline=(tmin, 0))
    
    # epochs_milk = mne.Epochs(raw_milk, events_milk , event_id_sd, tmin, tmax, 
    #         picks=picks_milk, proj=True, baseline=(tmin, 0))
   
    # epochs_odour = mne.Epochs(raw_odour, events_odour, event_id_sd, tmin, tmax, 
    #         picks=picks_odour, proj=True, baseline=(tmin, 0))
    
    epochs_lexical = mne.Epochs(raw_lexical, events_lexical, event_id_ld, tmin, tmax, 
            picks=picks_lexical, proj=True, baseline=(tmin, 0))
    
 
    
    
    # C_F_visual = 0
    # C_F_hear = 0
    # C_F_hand = 0
    # C_F_neutral = 0
    # C_F_emotional = 0
    # C_F_FP = 0

    # for evcnt in np.arange(0,events_fruit.shape[0]):    
    #     if (events_fruit[evcnt,2]==1):
    #        C_F_visual =  C_F_visual + 1
    #     if (events_fruit[evcnt,2]==2):
    #        C_F_hear =  C_F_hear + 1
    #     if (events_fruit[evcnt,2]==3):
    #        C_F_hand =  C_F_hand + 1
    #     if (events_fruit[evcnt,2]==4):
    #        C_F_neutral =  C_F_neutral + 1
    #     if (events_fruit[evcnt,2]==5):
    #        C_F_emotional =  C_F_emotional + 1 


    # C_M_visual = 0
    # C_M_hear = 0
    # C_M_hand = 0
    # C_M_neutral = 0
    # C_M_emotional = 0
    # C_M_FP = 0
    # for evcnt in np.arange(0,events_milk.shape[0]):    
    #     if (events_milk[evcnt,2]==1):
    #        C_M_visual =  C_M_visual + 1
    #     if (events_milk[evcnt,2]==2):
    #        C_M_hear =  C_M_hear + 1
    #     if (events_milk[evcnt,2]==3):
    #        C_M_hand =  C_M_hand + 1
    #     if (events_milk[evcnt,2]==4):
    #        C_M_neutral =  C_M_neutral + 1
    #     if (events_milk[evcnt,2]==5):
    #        C_M_emotional =  C_M_emotional + 1 
#
#
    # C_O_visual = 0
    # C_O_hear = 0
    # C_O_hand = 0
    # C_O_neutral = 0
    # C_O_emotional = 0
    # C_O_FP = 0
    # for evcnt in np.arange(0,events_odour.shape[0]):    
    #     if (events_odour[evcnt,2]==1):
    #        C_O_visual =  C_O_visual + 1
    #     if (events_odour[evcnt,2]==2):
    #        C_O_hear =  C_O_hear + 1
    #     if (events_odour[evcnt,2]==3):
    #        C_O_hand =  C_O_hand + 1
    #     if (events_odour[evcnt,2]==4):
    #        C_O_neutral =  C_O_neutral + 1
    #     if (events_odour[evcnt,2]==5):
    #        C_O_emotional =  C_O_emotional + 1 
           
    C_L_visual = 0
    C_L_hear = 0
    C_L_hand = 0
    C_L_neutral = 0
    C_L_emotional = 0
    C_L_FP = 0           
    for evcnt in np.arange(0,events_lexical.shape[0]):    
        if (events_lexical[evcnt,2]==1):
           C_L_visual =  C_L_visual + 1
        if (events_lexical[evcnt,2]==2):
           C_L_hear =  C_L_hear + 1
        if (events_lexical[evcnt,2]==3):
           C_L_hand =  C_L_hand + 1
        if (events_lexical[evcnt,2]==4):
           C_L_neutral =  C_L_neutral + 1
        if (events_lexical[evcnt,2]==5):
           C_L_emotional =  C_L_emotional + 1 

           
    # C_F_v.append(C_F_visual)
    # C_F_h.append(C_F_hear)
    # C_F_ha.append(C_F_hand)
    # C_F_n.append(C_F_neutral)
    # C_F_e.append(C_F_emotional)
    
    # C_M_v.append(C_M_visual)
    # C_M_h.append(C_M_hear)
    # C_M_ha.append(C_M_hand)
    # C_M_n.append(C_M_neutral)
    # C_M_e.append(C_M_emotional)
    
    # C_O_v.append(C_O_visual)
    # C_O_h.append(C_O_hear)
    # C_O_ha.append(C_O_hand)
    # C_O_n.append(C_O_neutral)
    # C_O_e.append(C_O_emotional)

    C_L_v.append(C_L_visual)
    C_L_h.append(C_L_hear)
    C_L_ha.append(C_L_hand)
    C_L_n.append(C_L_neutral)
    C_L_e.append(C_L_emotional)
    
