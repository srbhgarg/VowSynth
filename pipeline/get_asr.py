import librosa
import random
random.seed(5)
import numpy as np
from sklearn import preprocessing
from scipy.signal import savgol_filter
from librosa import feature, effects
from scipy.signal import resample
np.random.seed(0)
from sklearn.metrics import confusion_matrix
from scipy.io import wavfile
#to turn warnings off
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


freq=25
import parselmouth
from parselmouth.praat import call


[stylelist, vowellist, subjlist, filelist]=np.load('demographics.npy')
newlabels= vowellist
print(np.shape(filelist)[0])
shp=16384

oidx=[i for i, value in enumerate(subjlist) if (value == '1103' or value =='1203')]  #test indices
vidx=[i for i, value in enumerate(subjlist) if (value == '1102' or value =='1202')] #validation indices
tidx=[x for x in range(len(subjlist)) if x not in (vidx or oidx)]
test_idx=oidx
train_idx=tidx

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler_std= preprocessing.StandardScaler()
scaler_max= preprocessing.MaxAbsScaler()

def get_feat(y, sr):
    S = np.abs(librosa.stft(y))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10) #20 x 39
    feat= np.mean(mfccs,1)
    feat=np.append(feat, np.max(mfccs,1))
    feat=np.append(feat, np.min(mfccs,1))
    ##print(np.shape(feat), np.shape(mfccs))  

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) #12 x 39
    feat = np.append(feat, np.mean(chroma_stft,1))
    #feat=np.append(feat, np.max(chroma_stft,1))
    
    #print(np.shape(feat))
    
    p2 = librosa.feature.poly_features(S=S, order=4)
    #feat=np.append(feat, np.mean(p2,1))
    #feat=np.append(feat, np.max(p2,1))
    #feat=np.append(feat, np.min(p2,1))
    ##print(np.shape(feat))
    
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    feat=np.append(feat, np.mean(tonnetz,1))
    #feat=np.append(feat, np.max(tonnetz))
    #feat=np.append(feat, np.min(tonnetz))
    
    #print(np.shape(feat))
    
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr) # 39
    feat=np.append(feat, np.mean(spec_cent))
    feat=np.append(feat, np.max(spec_cent))
    #feat=np.append(feat, np.min(spec_cent))
    ##print(np.shape(feat))
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr) #1x39
    feat=np.append(feat, np.mean(spec_bw))
    feat=np.append(feat, np.max(spec_bw))
    feat=np.append(feat, np.min(spec_bw))
    ##print(np.shape(feat))
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  #39
    feat=np.append(feat, np.mean(rolloff))
    feat=np.append(feat, np.max(rolloff))
    feat=np.append(feat, np.min(rolloff))
    ##print(np.shape(feat))
    
    zcr = librosa.feature.zero_crossing_rate(y) #2002
    feat=np.append(feat, np.mean(zcr))
    #feat=np.append(feat, np.max(zcr))
    #feat=np.append(feat, np.min(zcr))
    ##print(np.shape(feat))
    return feat

sidx=0
def get_allfeat(y, sr, nframes=33):
    S = np.abs(librosa.stft(y))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) #20 x 39
    mfccs = resample(mfccs, nframes, axis=1)
    feat= np.concatenate(mfccs)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) #12 x 39
    chroma_stft = resample(chroma_stft, nframes, axis=1)
    feat = np.append(feat, np.concatenate(chroma_stft))
    
    p2 = librosa.feature.poly_features(S=S, order=3)
    p2=p2[:,sidx:nframes]
    
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz = resample(tonnetz, nframes, axis=1)
    feat=np.append(feat, np.concatenate(tonnetz))
    
    #print(np.shape(feat))
    S, phase = librosa.magphase(librosa.stft(y))
    S_power = S ** 2
    flatness=librosa.feature.spectral_flatness(S=S_power, power=1.0)
    flatness = resample(flatness, nframes, axis=1)
    feat=np.append(feat, np.concatenate(flatness))
    
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr) # 39
    spec_cent = resample(spec_cent, nframes, axis=1)
    feat=np.append(feat, np.concatenate(spec_cent))
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr) #1x39
    spec_bw = resample(spec_bw, nframes, axis=1)
    feat=np.append(feat, np.concatenate(spec_bw))
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  #39
    rolloff = resample(rolloff, nframes, axis=1)
    feat=np.append(feat, np.concatenate(rolloff))
    
    zcr = librosa.feature.zero_crossing_rate(y) #2002
    #feat=np.append(feat, np.mean(zcr))
    #feat=np.append(feat, np.max(zcr))
    #feat=np.append(feat, np.min(zcr))
    ##print(np.shape(feat))
    return feat


Xfeat=[]
for i in range(np.shape(filelist)[0]):#(1349):
    afile=filelist[i]+'_audio.wav'
    #aud , sr = librosa.load(afile)
    sr,aud = wavfile.read(afile)
    if(1):
        aud, index=effects.trim(np.float32(aud), top_db=freq)
        #either this (not so good)
        #aud =effects.time_stretch(np.float32(aud), np.shape(aud)[0]/shp)
        #or this
        #sound = parselmouth.Sound(afile)
        sound = parselmouth.Sound(aud.T, sampling_frequency=sr)
        #create manipulation object
        #time step=0, max # formants=5, max formant freq =5500, window len = 0.025, preemphasis=50
        manipulation = call(sound, "To Manipulation",0.01, 50, 200)

        ##extract durationtier and add a duration point
        duration_tier = call(manipulation, "Extract duration tier")
        duration = call(duration_tier, "Add point", sound.end_time, shp/(np.shape(aud)[0]*1.0))

        #replace the duration tier in the manipulation object
        call([duration_tier, manipulation], "Replace duration tier")

        #Publish resynthesis
        sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")

        formants = call(sound_octave_up, "To Formant (burg)", 0.0,5, 5500, 0.025,50) #create a praat pitch object
        formant=[]
        t=0.0
        for time in range(10):
            t+=0.0625
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
            formant.extend([f1/f2,f1/f3, f1/f4, f2/f3, f2/f4 , f3/f4])
            

        aud=sound_octave_up.values.T

        if(np.shape(aud)[0] < shp):
            aud=np.pad(aud, ((0, shp-np.shape(aud)[0]),(0,0)), 'constant', constant_values=((0, 0),(0,0)))

        aud=np.reshape(aud[0:shp],(shp,1))
        #standarize the data to be between -1 to +1
        
        aud=scaler_std.fit_transform(aud)
    y=scaler_max.fit_transform(aud)
    
    #feat =get_feat(y[:,0],16000)
    feat =get_allfeat(y[:,0],sr)
    temp=[]
    temp.extend(feat)
    #formants have nan in them, replace them with zeros or interpolate them
    if(np.any(np.isnan(formant))):
        formant=np.nan_to_num(formant,0)
    temp.extend(formant)

    Xfeat.append(temp)
    #summarize over frames
    #mean median

print("gt feat: ", np.shape(Xfeat))
if 1:
    allfeat = scaler_std.fit_transform(np.array(Xfeat))
else:
    allfeat = Xfeat
print(np.shape(feat), np.shape(allfeat))


#mfccs  (10, 30)
#chroma_stft  (12, 30)
#p2  (5, 45)
#tonnetz  (6, 30)
#spec_cent  (1, 30)
#spec_bw  (1, 30)
#rolloff  (1, 45)
#mfccs  (10, 30)
#chroma_stft  (12, 30)
#p2  (5, 33)
#tonnetz  (6, 30)
#spec_cent  (1, 30)
#spec_bw  (1, 30)
#rolloff  (1, 33)


#train classifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

names = [ "Random Forest"]
classifiers = [
    RandomForestClassifier(max_depth=16, n_estimators=150, max_features=10, random_state=137)]

#normalize the features
X_train=allfeat[train_idx]
y_train=newlabels[train_idx]

X_test=allfeat[test_idx]
y_test=newlabels[test_idx]

#training on gt data and testing on gt
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Ground Truth: ",name, 1-score)


testfeat=[]
testlab=[]
options='_cnn_crf_interpolate' #for model filename
string='_bestmodel_C3'+options
for j in range(np.shape(test_idx)[0]):#488, 860, 695
    testid=test_idx[j]
    #afile = './results/gl/gl_audio_'+string+str(testid)+'.wav'
    afile = './results/gt_phase/phase_audio_'+string+str(testid)+'.wav'
    #y , sr = librosa.load(afile)
    sr,aud = wavfile.read(afile)

    sound = parselmouth.Sound(aud.T, sampling_frequency=sr)
    manipulation = call(sound, "To Manipulation",0.01, 50, 200)
    sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")
    formants = call(sound_octave_up, "To Formant (burg)", 0.0,5, 5500, 0.025,50) #create a praat pitch object
    formant=[]
    t=0.0
    for time in range(10):
        t+=0.0625
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        formant.extend([f1/f2,f1/f3, f1/f4, f2/f3, f2/f4 , f3/f4])


    if(np.shape(aud)[0] < shp):
        aud=np.pad(aud, ((0, shp-np.shape(aud)[0])), 'constant', constant_values=(0,0))
    aud=np.reshape(aud[0:shp],(shp,1))
    aud=scaler_std.fit_transform(aud)
    y=scaler_max.fit_transform(aud)
    #print([afile,newlabels[testid][0]], sr)
    feat = get_allfeat(y[:,0],sr)
    temp=[]
    temp.extend(feat)
    #formants have nan in them, replace them with zeros or interpolate them
    if(np.any(np.isnan(formant))):
        formant=np.nan_to_num(formant,0)

    temp.extend(formant)
    testfeat.append(temp)
    testlab.append(newlabels[testid])

X_test2 = scaler_std.fit_transform(np.array(testfeat))
y_test2=testlab

print("Xfeat: ", np.shape(X_test2))

allpred=[]
#training on gt data and testing on all generated test data
for name, clf in zip(names, classifiers):
    #clf.fit(X_train, y_train)
    pred = clf.predict(X_test2)
    allpred.append(pred)
    score = clf.score(X_test2, y_test2)
    print("Generated: ",name, 1-score)
    print(allpred)

