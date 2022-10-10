#to get PESQ STOI scores from the audio
import random
random.seed(5)
import numpy as np
np.random.seed(5)
import os
os.environ['PYTHONHASHSEED']=str(5)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+

import scipy.io as scio
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import clear_output
from librosa.effects import preemphasis
from librosa.core import piptrack

#read videos and audio as training data for my own video encoder. If using vgg net check next cell
from os import listdir
from os.path import isfile, join, getmtime
from scipy.io import wavfile
from sklearn import preprocessing
from librosa import feature, effects
from scipy.signal import savgol_filter
#to play audio
from IPython.display import Audio 


#compute MCD and use dtw to align
import pysptk as pysptk

options='_cnn_crf_interpolate' #for model filename
string='_bestmodel_C3'+options
#method 1
def mcd(C, C_hat):
    """C and C_hat are NumPy arrays of shape (T, D),
    representing mel-cepstral coefficients.

    """
    K = 10 / np.log(10) * np.sqrt(2)
    return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis=1)))

#method 2: searches for minima using dynamic time warping if the two MFCCs are not aligned
logSpecDbConst = 10.0 / np.log(10.0) * np.sqrt(2.0)

#x,y should be float64
def logSpecDbDist(x, y):
    size = x.shape[0]
    assert y.shape[0] == size

    sumSqDiff = 0.0
    for k in range(size):
        diff = x[k] - y[k]
        sumSqDiff += diff * diff

    dist = np.sqrt(sumSqDiff) * logSpecDbConst
    return dist



def getCostMatrix(xs, ys, costFn):
    assert len(xs) > 0 and len(ys) > 0

    costMat = np.array([ [ costFn(x, y) for y in ys ] for x in xs ])
    assert np.shape(costMat) == (len(xs), len(ys))
    return costMat

def getCumCostMatrix(costMat):
    xSize, ySize = np.shape(costMat)

    cumMat = np.zeros((xSize + 1, ySize + 1))
    cumMat[0, 0] = 0.0
    cumMat[0, 1:] = float('inf')
    cumMat[1:, 0] = float('inf')
    for i in range(xSize):
        for j in range(ySize):
            cumMat[i + 1, j + 1] = min(
                cumMat[i, j],
                cumMat[i, j + 1],
                cumMat[i + 1, j]
            )
            cumMat[i + 1, j + 1] += costMat[i, j]

    return cumMat

def getBestPath(cumMat):
    xSize = np.shape(cumMat)[0] - 1
    ySize = np.shape(cumMat)[1] - 1
    assert xSize > 0 and ySize > 0

    i, j = xSize - 1, ySize - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        _, (i, j) = min(
            (cumMat[i, j], (i - 1, j - 1)),
            (cumMat[i, j + 1], (i - 1, j)),
            (cumMat[i + 1, j], (i, j - 1))
        )
        path.append((i, j))
    path.reverse()

    return path

def dtw(xs, ys, costFn):
    """Computes an alignment of minimum cost using dynamic time warping.
    A path is a sequence of (x-index, y-index) pairs corresponding to a pairing
    of frames in xs to frames in ys.
    The cost of a path is the sum of costFn applied to (x[i], y[j]) for each
    point (i, j) in the path.
    A path is valid if it is:
        - contiguous: neighbouring points on the path are never more than 1
          apart in either x-index or y-index
        - monotone: non-decreasing x-index as we move along the path, and
          similarly for y-index
        - complete: pairs the first frame of xs to the first frame of ys (i.e.
          it starts at (0, 0)) and pairs the last frame of xs to the last frame
          of ys
    Contiguous and monotone amount to saying that the following changes in
    (x-index, y-index) are allowed: (+0, +1), (+1, +0), (+1, +1).
    This function computes the minimum cost a valid path can have.
    Returns the minimum cost and a corresponding path.
    If there is more than one optimal path then one is chosen arbitrarily.
    """
    costMat = getCostMatrix(xs, ys, costFn)
    cumMat = getCumCostMatrix(costMat)
    minCost = cumMat[len(xs), len(ys)]
    path = getBestPath(cumMat)
    return minCost, path

#compute pesq and stoi scores
from pesq import pesq
from pystoi import stoi
import parselmouth
from parselmouth.praat import call
from librosa import feature, effects
from sklearn import preprocessing

freq=25
shp=16384

[stylelist, vowellist, subjlist, filelist]=np.load('demographics.npy')



scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler_std= preprocessing.StandardScaler()
scaler_max= preprocessing.MaxAbsScaler()


oidx=[i for i, value in enumerate(subjlist) if (value == '1103' or value =='1203')]  #test indices
test_idx=oidx
#loop over two styles
for st in range(1):

    pesq_wb=[]
    pesq_nb=[]
    st=[]
    md=[]
    for j in range(np.shape(test_idx)[0]):#488, 860, 695
        testid=test_idx[j]#860
        #testid=860+i
        
        afile=filelist[testid]+'_audio.wav'
        #print(testid, afile)
        sr,aud = wavfile.read(afile)
        aud, index=effects.trim(np.float32(aud), top_db=freq)
        aud=np.append(aud,np.zeros([1024]))
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
        len_aud=np.shape(aud)[0]  
        aud=np.reshape(aud[0:shp],(shp,1))
        aud=scaler_std.fit_transform(aud)
        gtaud=scaler_max.fit_transform(aud)[:,0]

        #print(np.shape(gtaud), np.shape(degaud))

        #degaudiofile=audio_filelist[testid]

        degaudiofile = './results/gl/gl_audio_'+string+str(testid)+'.wav'

        sr,degaud = wavfile.read(degaudiofile)

        if(np.shape(degaud)[0] > np.shape(gtaud)[0]):
            degaud=degaud[0:np.shape(gtaud)[0]]
        elif(np.shape(degaud)[0] < np.shape(gtaud)[0]):
            gtaud=gtaud[0:np.shape(degaud)[0]]


        #compute pesq scores
        pq_wb=pesq(sr, gtaud, degaud, 'wb')
        pq_nb=pesq(sr, gtaud, degaud, 'nb')
        pesq_wb.append(pq_wb)
        pesq_nb.append(pq_nb)    

        # Clean and gen should have the same length, and be 1D
        dis = stoi.stoi(gtaud, degaud, sr, extended=False)
        st.append(dis)


        #Compute MCD
        num=np.int(np.shape(degaud)[0]/1024)-5
        #print(np.shape(gtaud), np.shape(degaud), sr, num)
        costFn=logSpecDbDist
        frame=1
        nat=[]
        for ij in range(1,num):
            x=gtaud[1024*ij:1024*(ij+1)]
            #print(np.shape(x),x)
            try:
                mgc_temp = pysptk.mgcep(x)
            except:
                print("Error at: ", ij)
            else:
                nat.append(mgc_temp)

        if 1:
            #print(np.shape(mgc_temp))
            #degaud=np.reshape(degaud[0:shp],(shp,1))
            synth=[]
            for kidx in range(1,num):
                try:
                    x=degaud[1024*kidx:1024*(kidx+1)]
                    #print(np.shape(x),x)
                    mgc_temp = pysptk.mgcep(x)
                except:
                    print("Error at: ", kidx)
                else:
                    synth.append(mgc_temp)

        #print(np.shape(nat), np.shape(synth))
        nat=np.array(nat)
        synth=np.array(synth)
        nat = nat[:, 1:]
        synth = synth[:, 1:]
        minCost, path = dtw(nat, synth, costFn)
        frames = len(nat)
        mcdscore=minCost/frames

        md.append(mcdscore)
    print("PESQ_wb: ", np.mean(pesq_wb),"PESQ_nb: ",np.mean(pesq_nb),"STOI: ", np.mean(st),"MCD: ", np.mean(md))

 
