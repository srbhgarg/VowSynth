#this fine generates variables list that is needed for other functions
#variables filelist, stylelist, vowellist, subjlist are saved in demographics.mat

import sys
import scipy.io as scio
import os
import numpy as np

stylelist=[]
filelist=[]
vowellist=[]
subjlist=[]
if 0:
  for s in range(10):
    for p in range(3):
        part=str(p+1)
        subj=str(1101+s)
        if(not((subj=='1105' and p==0) or (subj=='1108') or (subj=='1103' and p==1))):
            #read the segmentation times
            mat=scio.loadmat('scratch/avc/lisa_data/token_times2/'+subj+'_part'+part+'_times.mat')
            times=mat['times']
            label=mat['str']
            style=mat['labels']
            stframe=round(times[0,i]*fps)-7
            endframe=round(times[1,i]*fps)+7
            out_file='scratch/avc/lisa_data_opt/'+subj+'/'+subj+'_part'+part+'_'+str(label[0,i]).strip('[\']')+'_'+str(stframe)+'_'+str(endframe)+'_mouth'
            newlabels.append(label[0,i])
            styles.append(style[0,i]) 
else:
  filename = sys.argv[1]
  coord=open(filename,"r");
  Lines=coord.readlines()
  fps=30
  for line in Lines:
    temp=line.rstrip('\n')
    subj=temp.split('/')[6]

    if(int(subj) > 1200):
        temp=temp[:-5] #remove string '_both' from 1200 subjects

    fields=temp.split('_')
    part=fields[3][-1]
    label=fields[4]
    stframe=fields[5]
    endframe=fields[6]
    mat=scio.loadmat('scratch/avc/lisa_data/token_times2/'+subj+'_part'+part+'_times.mat')
    times=mat['times']
    newlabel=mat['str']
    newstyle=mat['labels']
    idx=np.abs(times[0,:]-((float(stframe)+7)/fps)).argmin()
    if((round(times[1,idx]*fps)+7)==float(endframe)):
        #print(subj, label, newlabel[0,idx], newstyle[0,idx])
        stylelist.append(newstyle[0,idx])		
        vowellist.append(label)		
        subjlist.append(subj)		
        filelist.append(temp)		
    else:
        print(idx, round(times[0,idx]*fps)-7, round(times[1,idx]*fps)+7, stframe, endframe)
        print("==================> Entry not found ", subj, label, newlabel[0,idx])

output_filename = sys.argv[2]
print(np.shape(filelist))
np.save(output_filename,[stylelist, vowellist, subjlist, filelist])
