#input: model name, test_idx
#output: _gl or combined_audio wav files, output filelist

import numpy as np
import os
from librosa.core import griffinlim


fname='best_model'
options='_cnn_crf_interpolate' #for model filename
[stylelist, vowellist, subjlist, filelist]=np.load('demographics.npy')
[Xtrain, Xtrain_aud, Xtrain_aug] =np.load('AV_data'+options+'.npy')
oidx=[i for i, value in enumerate(subjlist) if (value == '1103' or value =='1203')]  #test indices

test_idx=oidx

landmarks_indices=[*range(4,13),*range(48,68)] #5:13 is lower jaw and 49:68 is lips
frameCount=30
rate=16000
X_out=[]

# Hyperparameter
include_phase=2 #0 = griffin_lim, 1=template phase, 2= both


import scipy.io as scio
## DC gans for sound
## create generator
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Conv2DTranspose, Input, LayerNormalization
from tensorflow.keras.layers import LeakyReLU, Flatten, Dense, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow import expand_dims
from tensorflow import shape, image
from tensorflow import reshape
from tensorflow.keras import optimizers
def Conv1DTranspose(inputs, filters, kernel_width, stride=4, padding='same',upsample='zeros'):

    if upsample == 'zeros':
        out=Conv2DTranspose(filters,(1, kernel_width),strides=(1, stride),padding='same')(expand_dims(inputs, axis=1))
        return out[:, 0]
    elif upsample == 'nn':
        batch_size = shape(inputs)[0]
        _, w, nch = inputs.get_shape().as_list()
   
        x = inputs
        x = expand_dims(x, axis=1)
        x = image.resize_nearest_neighbor(x, [1, w * stride])
        x = x[:, 0]
        return Conv1d(x,filters,kernel_width,1,padding='same')
    else:
        raise NotImplementedError


learning_rate=0.01
decay_rate=1e-5
gopt = optimizers.Adam(lr=learning_rate, decay=decay_rate)



depth = 128
dim = 32

# get the output of the last layer
   

# input will be features extracted from the video ..channels are probably features from lips, eyebrow and head movement


# input is a vector of 2048 random numbers and convert it to 16x1024 size for convolution

# for every video time frame we want to generate 533.34 (512 nearest power of 2) audio samples
# 1 4 16 64 256 512

#input 58 x 48 or 64 , 80 timepoints x 58 landmark points +1 style

#g_input = Input(shape=[64]) # dense output of classifier
#gen=Dense(dim*depth)(g_input)
include_style=False
if include_style:
    #add style info in the input
    g_input = Input(shape=[frameCount,frameHeight*frameWidth+1])
else:
    g_input = Input(shape=[frameCount,np.shape(landmarks_indices)[0]*2])


gen=Flatten(data_format='channels_first')(g_input)
gen=Dense(dim*depth)(gen)
gen=reshape(gen, [-1, dim, depth])
gen=BatchNormalization(momentum=0.9)(gen)
gen=Activation('relu')(gen)

gen=Conv1DTranspose(gen,filters=int(depth/2), kernel_width=25, stride=2)
gen=BatchNormalization(momentum=0.9)(gen)
gen=Activation('relu')(gen)


gen=Conv1DTranspose(gen,filters=int(depth/4), kernel_width=25, padding='same')
gen=BatchNormalization(momentum=0.9)(gen)
gen=Activation('relu')(gen)

gen=Conv1DTranspose(gen,filters=int(depth/8), kernel_width=25, padding='same')
gen=BatchNormalization(momentum=0.9)(gen)
gen=Activation('relu')(gen)

gen=Conv1DTranspose(gen,filters=int(depth/16), kernel_width=25, padding='same', stride=4)
gen=BatchNormalization(momentum=0.9)(gen)
gen=Activation('relu')(gen)

# ...add the following block
#gen=Conv1DTranspose(gen,filters=int(depth/16), kernel_width=25, padding='same', stride=4)
#gen=BatchNormalization(momentum=0.9)(gen)
#gen=Activation('relu')(gen)


# filters==1 needs to be checked
gen=Conv1DTranspose(gen,filters=1, kernel_width=25, padding='same', stride=4)
gen_V=Activation('tanh')(gen)


generator = Model(g_input,gen_V, name='gener')
generator.summary()


generator.load_weights("./results/"+fname+options+'_fn_C3.h5')

string='_bestmodel_C3'+options
#os.mkdir('./results/gl/')
#os.mkdir('./results/gt_phase/')


_LOG_EPS = 1e-6
frame_length= 128 # np.int32(sr*0.005) #window len - 80
frame_step=np.int32(np.round(rate*0.002)) # time step -32 - increase value decreases X-ticks - 0.002 or 0.006
fft_len=512 # increase value increases y ticks 

for i in range(np.shape(test_idx)[0]):
    testid=test_idx[i]
    reshape_input=np.float16(reshape(np.asarray(Xtrain[testid]),[1,frameCount,2*68]))
    reshape_input = reshape_input[:,:,np.concatenate((landmarks_indices,np.array(landmarks_indices)+68))]

    gen_wave=generator.predict(reshape_input)
    gt_wave=Xtrain_aud[testid]

    spec_pred = tf.signal.stft(np.float32(np.transpose(gen_wave[0])), frame_length, frame_step,fft_length=fft_len, pad_end=False)
    X_mag_pred = tf.abs(spec_pred)

    spec_phase_true = tf.signal.stft(np.float32(np.transpose(Xtrain_aud[1])), frame_length, frame_step,fft_length=fft_len, pad_end=False)

    if (include_phase==0):
        mag_pred = X_mag_pred[0,:,:].numpy()
        wav_audio = griffinlim(np.transpose(mag_pred), hop_length=frame_step, win_length=frame_length)
        scio.wavfile.write('./results/gl/gl_audio_'+string+str(testid)+'.wav',rate,wav_audio)
    elif(include_phase==1):
        im= tf.constant([1j], dtype=tf.complex128)
        phase=tf.cast(tf.math.angle(spec_phase_true[0]),dtype=tf.complex128)
        mag=tf.cast(X_mag_pred[0], dtype=tf.complex128)
        combined = tf.math.multiply(mag, tf.math.exp(im*phase))
        combined_audio = tf.math.real(tf.signal.inverse_stft(combined, frame_length, frame_step, fft_len))
        X_out.append(combined_audio)
        scio.wavfile.write('./results/gt_phase/phase_audio_'+string+str(testid)+'.wav',rate,combined_audio.numpy())
    else:
        #both
        mag_pred = X_mag_pred[0,:,:].numpy()
        wav_audio = griffinlim(np.transpose(mag_pred), hop_length=frame_step, win_length=frame_length)
        scio.wavfile.write('./results/gl/gl_audio_'+string+str(testid)+'.wav',rate,wav_audio)

        im= tf.constant([1j], dtype=tf.complex128)
        phase=tf.cast(tf.math.angle(spec_phase_true[0]),dtype=tf.complex128)
        mag=tf.cast(X_mag_pred[0], dtype=tf.complex128)
        combined = tf.math.multiply(mag, tf.math.exp(im*phase))
        combined_audio = tf.math.real(tf.signal.inverse_stft(combined, frame_length, frame_step, fft_len))
        X_out.append(combined_audio)
        scio.wavfile.write('./results/gt_phase/phase_audio_'+string+str(testid)+'.wav',rate,combined_audio.numpy())


