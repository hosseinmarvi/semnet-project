brain = Brain('fsaverage', 'split', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(800, 400))
# Right ATL area - splitting TE2a      
label_TE2a = ['R_TE2a_ROI-rh']
my_TE2a=[]
for j in np.arange(0,len(label_TE2a )):
    my_TE2a.append([label for label in labels if label.name == label_TE2a[j]][0])

for m in np.arange(0,len(my_TE2a)):
    if m==0:
        r_TE2a = my_TE2a[m]
    else:
        r_TE2a = r_TE2a + my_TE2a[m]
        

[r_TE2a1,r_TE2a2,r_TE2a3]=mne.split_label(label=r_TE2a,parts=\
    ('R_TE2a1_ROI-rh','R_TE2a2_ROI-rh','R_TE2a3_ROI-rh'),subject='fsaverage',\
    subjects_dir=C.data_path)

brain.add_label(r_TE2a1, borders=False,color='green')
brain.add_label(r_TE2a2, borders=False,color='red')
brain.add_label(r_TE2a3, borders=False,color='yellow')


# Right ATL area - splitting TE1m 
label_TE1m = ['R_TE1m_ROI-rh']
my_TE1m=[]
for j in np.arange(0,len(label_TE1m )):
    my_TE1m.append([label for label in labels if label.name == label_TE1m[j]][0])

for m in np.arange(0,len(my_TE1m)):
    if m==0:
        r_TE1m = my_TE1m[m]
    else:
        r_TE1m = r_TE1m + my_TE1m[m]
        

[r_TE1m1,r_TE1m2,r_TE1m3]=mne.split_label(label=r_TE1m,parts=\
    ('R_TE1m1_ROI-rh','R_TE1m2_ROI-rh','R_TE1m3_ROI-rh'),subject='fsaverage',\
    subjects_dir=C.data_path)       
[r_TE1m11,r_TE1m12,r_TE1m13]=mne.split_label(label=r_TE1m1,parts=\
    ('R_TE1m11_ROI-rh','R_TE1m12_ROI-rh','R_TE1m13_ROI-rh'),subject='fsaverage',\
    subjects_dir=C.data_path)
[r_TE1m21,r_TE1m22,r_TE1m23]=mne.split_label(label=r_TE1m2,parts=\
    ('R_TE1m21_ROI-rh','R_TE1m22_ROI-rh','R_TE1m23_ROI-rh'),subject='fsaverage',\
    subjects_dir=C.data_path)
[r_TE1m31,r_TE1m32,r_TE1m33]=mne.split_label(label=r_TE1m3,parts=\
    ('R_TE1m31_ROI-rh','R_TE1m32_ROI-rh','R_TE1m33_ROI-rh'),subject='fsaverage',\
    subjects_dir=C.data_path)


brain.add_label(r_TE1m31, borders=False,color='green')
brain.add_label(r_TE1m32, borders=False,color='red')
brain.add_label(r_TE1m33, borders=False,color='yellow')

# Right ATL area  
label_ATL = ['R_TGd_ROI-rh','R_TGv_ROI-rh','R_TE1a_ROI-rh']


my_ATL=[]
for j in np.arange(0,len(label_ATL )):
    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])

for m in np.arange(0,len(my_ATL)):
    if m==0:
        r_ATL = my_ATL[m]
    else:
        r_ATL = r_ATL + my_ATL[m]
        
        
#label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TE2a_ROI-lh','L_TE1a_ROI-lh','L_TE1m_ROI-lh']
#
#
#my_ATL=[]
#for j in np.arange(0,len(label_ATL )):
#    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])
#
#for m in np.arange(0,len(my_ATL)):
#    if m==0:
#        ATL = my_ATL[m]
#    else:
#        ATL = ATL + my_ATL[m]
        
        
r_ATL = r_ATL + r_TE2a2 + r_TE2a3 + r_TE1m13 + r_TE1m23

brain.add_label(r_ATL, borders=False,color='blue')
brain.add_label(ATL, borders=False,color='blue')


my_color=['blue','green','red','orange','purple']
for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False,color=my_color[m])
