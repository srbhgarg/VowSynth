import numpy as np

import tensorflow as tf

options='_cnn_crf_interpolate'

style=1 
gender=0
#load data
[stylelist, vowellist, subjlist, filelist]=np.load('demographics.npy')
[Xtrain, Xtrain_aud, Xtrain_aug] =np.load('AV_data_avg'+options+'.npy', allow_pickle=True)

vidx=[i for i, value in enumerate(subjlist) if (value =='1209' or value == '1109')] #validation indices
oidx=[i for i, value in enumerate(subjlist) if (value == '1201' or value =='1203' or value=='1210' or value =='1103' or value =='1101' or value=='1110')]  #test indices
tidx=[x for x in range(len(subjlist)) if x not in (vidx) and x not in (oidx)]

#

plain=np.where(stylelist =='1')[0]
clear=np.where(stylelist =='2')[0]
subjlist=subjlist.astype(int)
male=np.where(subjlist <1200)[0]
female=np.where(subjlist >1199)[0]

if (style==1):
    #train on plain, test on plain
    #plain =1, clear =2
    train_idx=np.intersect1d(tidx, plain)
    val_idx=np.intersect1d(vidx, plain)
    test_idx=np.intersect1d(oidx, plain)
elif(style==2):
    #train on plain, test on clear
    train_idx=np.intersect1d(tidx, plain)
    val_idx=np.intersect1d(vidx, plain)
    test_idx=np.intersect1d(oidx, clear)
elif(style==3):
    #train on clear, test on plain
    train_idx=np.intersect1d(tidx,clear)
    val_idx=np.intersect1d(vidx, clear)
    test_idx=np.intersect1d(oidx, plain)
elif(style==4):
    #train on clear, test on clear
    train_idx=np.intersect1d(tidx,clear)
    val_idx=np.intersect1d(vidx, clear)
    test_idx=np.intersect1d(oidx, clear)
else:
    #no style
    train_idx=tidx
    val_idx=vidx
    test_idx=oidx

if(gender==1):
    #train on male, test on male
    train_idx=np.intersect1d(train_idx,male)
    val_idx=np.intersect1d(val_idx, male)
    test_idx=np.intersect1d(test_idx, male)
elif(gender==2):
    #train on male, test on female
    train_idx=np.intersect1d(train_idx,male)
    val_idx=np.intersect1d(val_idx, male)
    test_idx=np.intersect1d(test_idx, female)
elif(gender==3):
    #train on female, test on male
    train_idx=np.intersect1d(train_idx, female)
    val_idx=np.intersect1d(val_idx, female)
    test_idx=np.intersect1d(test_idx, male)
elif(gender==4):
    #train on female, test on female
    train_idx=np.intersect1d(train_idx, female)
    val_idx=np.intersect1d(val_idx, female)
    test_idx=np.intersect1d(test_idx, female)

sr=16000
_LOG_EPS = 1e-6
frame_length= 128 # np.int32(sr*0.005) #window len - 80
frame_step=np.int32(np.round(sr*0.002)) # time step -32 - increase value decreases X-ticks - 0.002 or 0.006
fft_len=512 # increase value increases y ticks 
alpha=0.8

def get_spectrogram(waveform):
    dyn_range=70
    
    stfts = tf.signal.stft(np.float32(np.transpose(waveform)), frame_length, frame_step,fft_length=fft_len, pad_end=False)
    spectrograms = tf.math.abs(stfts)
    spec_pred_db=10*tf.math.log(spectrograms + _LOG_EPS)
    
    #adjust dynamic range
    mx=tf.math.reduce_max(spec_pred_db)
    spec_pred_db= tf.clip_by_value(spec_pred_db, mx-dyn_range, mx)
    spect = tf.nn.leaky_relu(spec_pred_db, alpha=alpha) 
    #spect=spec_pred_db
    return spect

shp=16384
import parselmouth
from parselmouth.praat import call

def get_formants(Xtrain_aud):

    sound = parselmouth.Sound(Xtrain_aud.T, sampling_frequency=16000)
    manipulation = call(sound, "To Manipulation",0.01, 50, 200)
    spect = call(sound, "To Spectrogram",0.008, 5000, 0.002, 20.0, 'Gaussian')

    ##extract durationtier and add a duration point
    duration_tier = call(manipulation, "Extract duration tier")
    duration = call(duration_tier,"Add point", sound.end_time, shp/(np.shape(Xtrain_aud)[0]*1.0))

    #replace the duration tier in the manipulation object
    call([duration_tier, manipulation], "Replace duration tier")

    #Publish resynthesis
    sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")

    formants = call(sound_octave_up, "To Formant (burg)", 0.002,5, 5500, 0.004,50) #time_step=0.002s and window length= 0.004*2=0.008
    
    #np.shape(Xtrain_aud[i])[0]
    #t=0.5*np.shape(Xtrain_aud)[0]/16000
    t=0.002
    fmnt=[]
    for tidx in range(509):
        t=t+0.002
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        bw1 = call(formants, "Get bandwidth at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        bw2 = call(formants, "Get bandwidth at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        bw3 = call(formants, "Get bandwidth at time", 3, t, 'Hertz', 'Linear')
        fmnt.append([f1,f2,f3, bw1, bw2, bw3])
    return fmnt

if 0:
    spec_all=[]
    formant_all=[]
    #Prepare input and output data
    for i in range(2829):
        waveform=Xtrain_aud[i]
        #509 time steps and 257 freq bins
        spec=get_spectrogram(waveform)
        spec_all.append(spec.numpy())
        #tf.print(i, tf.shape(spec_all))
        formant = get_formants(waveform)
        formant_all.append(formant)
        #print(i, np.shape(formant_all))
    np.save('formant_data_tracking_all_steps',[spec_all, formant_all])
else:
    [spec_all, formant_all]= np.load('formant_data_tracking_all_steps.npy', allow_pickle=True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,TimeDistributed, MaxPooling2D, Conv2D, Flatten, Input, Dense, BatchNormalization, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError

## DEFINE LOSS FUNCTION
def formant_loss():
    def floss(y_true, y_pred):
        mse=MeanSquaredError()
        y_pred=tf.squeeze(y_pred)
        #tf.print(y_true, y_pred)
        y_true=tf.multiply(y_true,[1/30, 1/150, 1/250, 1/20,1/30,1/38.5])
        y_pred=tf.multiply(y_pred,[1/30, 1/150, 1/250, 1/20,1/30,1/38.5])
        #tf.print(y_true[0], y_pred[0], mse(y_true, y_pred))
        #tf.print('--------------------')
        return mse(y_true, y_pred)
    def relative_floss(y1, y2):
        index_=2.0
        axis_fit_degree=50.0
        refer = 1.0/axis_fit_degree+(1.0-1.0/axis_fit_degree)*(tf.abs(y1)+tf.abs(y2))
        relative_loss = tf.abs(y1-y2)/refer
        cost = tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, index_), 0))
        return cost
    return floss
    #return relative_floss

spec_all=np.stack(spec_all)
spec_all=spec_all[:,0,:,:]
spec_all=np.expand_dims(spec_all, axis=3)
formant_all=np.stack(formant_all)
### CREATE MODEL
#spec shape 1 509 257
dense=2
if (dense==1):
    sinput = Input(shape=np.shape(spec_all[0]))
    network = Flatten(data_format='channels_last')(sinput)
    network = Dense(1024)(network)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('relu')(network)
    network = Dense(256)(network)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('relu')(network)
    network = Dense(6)(network)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('tanh')(network)

    model= Model(sinput, network, name='formant_estimate')
elif(dense==2):
    #tf.print(tf.shape(spec_all), tf.shape(spec_all[0]))
    # bs x 509 x 257 x 1 x1
    #audio to formant using RNN
    sinput = Input(shape=(1,1,509,257))
    #sinput = Input(shape=np.shape(spec_all[0]))
    #network = MaxPooling2D(pool_size=(10,1), strides=(10,1))(sinput)
    #network = Conv2D(1, kernel_size=(5,5))(network)  #for smothing the spectrogram
    network = TimeDistributed(Dense(200))(sinput)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('relu')(network)
    network = TimeDistributed(Dense(128))(network)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('relu')(network)
    #network = Dense(128)(network)
    #network=BatchNormalization(momentum=0.9)(network)
    #network=Activation('relu')(network)
    network = TimeDistributed(Dense(6))(network)
    #network=BatchNormalization(momentum=0.9)(network)
    model= Model(sinput, network, name='formant_estimate')
else:
    sinput = Input(shape=np.shape(spec_all[0]))
    network = Conv2D(16, kernel_size=(5,5))(sinput)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('relu')(network)
    network = MaxPooling2D(pool_size=(2,2), strides=(2,2))(network)
    network = Dropout(0.5)(network)

    network = Conv2D(32, kernel_size=(5,5))(network)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('relu')(network)
    network = MaxPooling2D(pool_size=(2,2), strides=(2,2))(network)
    network = Dropout(0.5)(network)
    
    #network = Conv2D(32, kernel_size=(3,3))(network)
    #network=BatchNormalization(momentum=0.9)(network)
    #network=Activation('relu')(network)
    #network = MaxPooling2D(pool_size=(2,2), strides=(2,2))(network)
    

    network = Conv2D(16, kernel_size=(5,5))(network)
    network=BatchNormalization(momentum=0.9)(network)
    network=Activation('relu')(network)
    network = MaxPooling2D(pool_size=(2,2), strides=(2,2))(network)
    network = Dropout(0.5)(network)
    #network = Conv2D(32, kernel_size=(5,5), activation='relu')(network)
    #network = MaxPooling2D(pool_size=(2,2), strides=(2,2))(network)
    network = Flatten()(network)
    network = Dense(32)(network)
    network = Dense(6)(network)
    #network=Activation('tanh')(network)
    model= Model(sinput, network, name='formant_estimate')

model.summary()
CB=[]
CB.append( tf.keras.callbacks.ModelCheckpoint(filepath='Formant_estimate_tracking_test_dense_LR001_fn_CB.h5',monitor='val_loss',save_best_only=True))
#CB.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.5, patience=10))


decay_rate=1e-5
#optim=optimizers.Adam(learning_rate=0.1, decay=decay_rate)
optim=optimizers.Adam(learning_rate=0.01) #old was 0.01
model.compile(loss=formant_loss(), optimizer=optim, metrics=['mse'])

#bsx 509x257x1x1
spec_all=np.expand_dims(spec_all,axis=4)
#bsx1x1x257x509
spec_all=np.transpose(spec_all,(0,4,3,1,2))

#set nan to 0, 36 values were found to be nan
formant_all[np.argwhere(np.isnan(formant_all))]=0


count=2050
#history = model.fit(spec_all[train_idx], formant_all[train_idx], batch_size=125, epochs=count, validation_data=(spec_all[val_idx], formant_all[val_idx]),verbose=2, callbacks=CB)

#model.save_weights('formant_estimate_tracking_2050epochs_lr001_3layers.h5')
model.load_weights('formant_estimate_tracking_2050epochs_lr001_3layers.h5')
for j in range(len(test_idx)):
    idx=test_idx[j]
    track_val=model.predict(np.expand_dims(spec_all[idx], axis=0) )
    val=np.squeeze(track_val)[257] #pick from the center
    #print(vowellist[idx], val[0], val[1], val[2], formant_all[idx,257,0], formant_all[idx,257,1], formant_all[idx,257,2])
    print(vowellist[idx], val[3], val[4], val[5], formant_all[idx,257,3], formant_all[idx,257,4], formant_all[idx,257,5])

