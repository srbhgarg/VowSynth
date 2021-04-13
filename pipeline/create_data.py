#run: python create_data.py

from os import listdir
from os.path import isfile, join, getmtime
from scipy.io import wavfile
from sklearn import preprocessing
from librosa import feature, effects
from scipy.signal import savgol_filter
import scipy.io as spio
#to play audio
from IPython.display import Audio 
import numpy as np
import cv2
import parselmouth
from parselmouth.praat import call
import os

#### Hyper parameters ###########################
#pad video with 0 or interpolation
pad_data=True # false = interpolation, true= pad with zeros or last frame
opt=0 # 3: aligned+CNN-CRF 2: aligned+SAN, 1: SAN, 0: CNN-CRF ; here aligned means face aligned i.e. rotated and scaled
data_augment=True
normalize_video=False # make video landmarks between -1 to 1
##########################



#Below are fixed parameters
freq=25   # to set dynamic range of the spectrograms
shp=16384 #length of audio from each video. This is the minimum.- 9600
frameWidth=2
frameHeight=29 #68
if (pad_data==True):
    frameCount=80 #50 before or 80
else:
    framCount=30
maxCount=1;

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler_std= preprocessing.StandardScaler()
scaler_max= preprocessing.MaxAbsScaler()

Xtrain=[]
Xtrain_aud=[]
Xtrain_aug=[] #stores augmented data
fps=30
landmarks_indices=[*range(0,68)] #5:13 is lower jaw and 49:68 is lips

[stylelist, vowellist, subjlist, filelist]=np.load('demographics.npy')
newstyles=stylelist.astype(np.int32)-1 #change the range from 1-2 to 0-1
print(np.shape(filelist)[0]) #2829
for jk in range(np.shape(filelist)[0]):
    i=jk
    
    basename = os.path.basename(filelist[i])
    afile=filelist[i]+'_audio.wav'

    if (opt==1):
        vfile='./results/san/'+basename+'_san.npy'
        #vfile=filelist[i]+'_san.npy'
        option='_san'
    elif (opt==2):
        vfile='./results/san_align/'+basename+'_san.npy'
        #vfile=filelist[i]+'_aligned_san.npy'
        option='_aligned_san'
    elif (opt==3):
        vfile=filelist[i]+'_aligned_crf.npy'
        option='_aligned_cnn_crf'
    else:
        vfile='./results/cnn_crf/'+basename+'_cnn_crf.mat'
        #vfile=filelist[i]+'_cnn_crf.mat'
        option='_cnn_crf'

    #_both.wmv contains both audio and video
    print(i, vfile)
    sr,aud = wavfile.read(afile)

    #add the 7frame buffer back to the audio
    smples=np.int(sr*7/fps)
    aud = np.pad(np.float32(aud), (smples, smples), 'constant', constant_values=(0, 0))

    if (opt==1 or opt ==2):
        #image is of shape timepts x landmark pts x 1 x coordinates e.g. (28, 68, 2)
        image1=np.load(vfile)
    else:
        #image is of shape timepts x landmark pts x 1 x coordinates e.g. (28, 68, 2)
        video=spio.loadmat(vfile)
        image1=video['joint_mean']

    # PREPROCESSING AUDIO BELOW:
    #remove silence from the audio
    aud, index=effects.trim(np.float32(aud), top_db=freq)
   
    #16000 13.44 40.32 7168 21504 14336
    #compute video frame numbers from index
    stframe=np.floor(fps*index[0]/sr)
    endframe=np.ceil(fps*index[1]/sr)
    
    #trim video based on audio
    image1=image1[np.int(stframe):np.int(endframe),:,:]

 
    image=[]
    image_aug=[]
    for tp in range(np.shape(image1)[0]):
        #tp x landmarks x coordinates
        # measurements based on nose center point 33
        #normalize based on distance between the eyes and eye: 45 and 39
        #normalize basedon distance between eye and nose: 33 and 27
        
        #for x-axis
        tempX = np.float32(image1[tp,landmarks_indices,0] - image1[tp,33,0])/(image1[1,45,0] - image1[1,39,0])
       
        #mirror image of the face
        augX=  np.float32(image1[tp,33,0] - image1[tp,landmarks_indices,0])/(image1[1,45,0] - image1[1,39,0])
        #for y-axis
        tempY = np.float32(image1[tp,landmarks_indices,1] - image1[tp,33,1])/(image1[1,33,1] - image1[1,27,1])
        if(tp==0):
            firstX=tempX
            firstY=tempY
            augfirstX=augX

        #all the other measurements are w.r.t to the first frame
        image.append(np.concatenate((tempX-firstX,tempY-firstY), axis=0))
        if data_augment:
            image_aug.append(np.concatenate((augX-augfirstX,tempY-firstY), axis=0))
    

    #because cv has width x height and numpy is heightxwidth

    dim = (np.shape(landmarks_indices)[0]*2,frameCount)
    # resize image
    if pad_data:
        option=option+'_pad'
        if 0:
            #pad with edge values instead of 0
            hsv=np.pad(image,((0, frameCount-np.shape(image)[0]),(0,0)), mode='edge')
            if data_augment:
                hsv_aug=np.pad(image_aug,((0, frameCount-np.shape(image)[0]),(0,0)), mode='edge')
            #print(np.shape(image),np.shape(hsv))
 
            ##pad on both sides with 0
            #offset= np.int(np.round((frameCount-np.shape(image)[0])/2))
            #hsv=np.zeros((frameCount,frameHeight*frameWidth));
            #hsv[offset:offset+np.shape(image)[0],:]=image
        else:
            ##pad with edge values at the end
            #hsv=np.pad(image,((0, frameCount-np.shape(image)[0]),(0,0)), mode='edge')
            hsv=np.pad(image,((0, frameCount-np.shape(image)[0]),(0,0)) , 'constant', constant_values=0)
            if data_augment:
                #hsv_aug=np.pad(image_aug,((0, frameCount-np.shape(image)[0]),(0,0)), mode='edge')
                hsv_aug=np.pad(image_aug,((0, frameCount-np.shape(image)[0]),(0,0)) , 'constant', constant_values=0)
            #print(np.shape(image),np.shape(hsv))
        
    else:
        option=option+'_interpolate'
        #(56, 58) (64, 58)
        hsv = cv2.resize(np.float32(image), dim, interpolation = cv2.INTER_CUBIC)
        if data_augment:
            hsv_aug = cv2.resize(np.float32(image_aug), dim, interpolation = cv2.INTER_CUBIC)
            
        
    #plt.subplot(2,1,1)
    #plt.plot(hsv)
    for lp in range(np.shape(hsv)[1]):
        #smooth the video data: TODO - play with window size
        yhat = savgol_filter(hsv[:,lp], 5, 4) # window size 35, polynomial order 3
        hsv[:,lp] = yhat
    
    if data_augment:
        for lp in range(np.shape(hsv_aug)[1]):
            #smooth the video data: TODO - play with window size
            yhat = savgol_filter(hsv_aug[:,lp], 5, 4) # window size 35, polynomial order 3
            hsv_aug[:,lp] = yhat

        Xtrain_aug.append(hsv_aug) 
        
    print(np.shape(image), np.shape(hsv), sr)
    # append the next video file
    if normalize_video:
        test=scaler_std.fit_transform(hsv)
        Xtrain.append(test)
    else:
        Xtrain.append(hsv)#np.nan_to_num(buf)

    #=====AUDIO===============================================
    if (pad_data):
        if(shp > np.shape(aud)[0]):
            #pad at the end with 0
            aud = np.pad(np.float32(aud), (0, shp-np.shape(aud)[0]), 'constant', constant_values=(0, 0))
        else:
            aud = aud[0:shp]
    else:
        #either this (not so good)
        #aud =effects.time_stretch(np.float32(aud), np.shape(aud)[0]/shp)
        #or this
        sound = parselmouth.Sound(aud.T, sampling_frequency=sr)
        #create manipulation object
        manipulation = call(sound, "To Manipulation",0.01, 50, 200)

        ##extract durationtier and add a duration point
        duration_tier = call(manipulation, "Extract duration tier")
        duration = call(duration_tier, "Add point", sound.end_time, shp/(np.shape(aud)[0]*1.0))

        #replace the duration tier in the manipulation object
        call([duration_tier, manipulation], "Replace duration tier")

        #Publish resynthesis
        sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")
        
        aud=sound_octave_up.values.T
        #if(sound_octave_up.n_samples < shp):
        #    aud=np.pad(aud[:,0],np.round((shp-sound_octave_up.n_samples)/2).astype('int32') ,mode='wrap')
    len_aud=np.shape(aud)[0]

    if(len_aud<16384):
    #    ##pad adds in both directions
    #    #aud=np.pad(aud,16384-len_aud,mode='wrap')
    #    ##we want to pad at the end only
    #    #print(np.shape(aud))
         aud=np.concatenate([aud[:,0], np.zeros(shp-len_aud)])
    aud=np.reshape(aud[0:shp],(shp,1))
    #standarize the data to be between -1 to +1
    #aud=scaler.fit_transform(aud)
    aud=scaler_std.fit_transform(aud)
    aud=scaler_max.fit_transform(aud)
    Xtrain_aud.append(aud)
np.shape(Xtrain_aud)

np.save('AV_data'+option,[Xtrain, Xtrain_aud, Xtrain_aug])
