import random
random.seed(5)
import numpy as np
np.random.seed(5)
import os
os.environ['PYTHONHASHSEED']=str(5)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
import tensorflow as tf
tf.random.set_seed(5)

import scipy.io as scio
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D,Conv3D, MaxPooling3D, concatenate, Dropout
from tensorflow.keras.layers import Conv2DTranspose, Input, LayerNormalization
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import LeakyReLU, Flatten, Dense, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import optimizers

from keras.utils import to_categorical
from IPython.display import clear_output
from librosa.effects import preemphasis
from librosa.core import piptrack
#import tensorflow_probability as tfp


####Hyper-parameters used
batch_size=25
region=2 # 0: jaw,  1: mouth only, 2: both mouth and jaw for landmark indices; 3: all face
style=1 #0: no style; 1: train in plain test on plain; 2: train on plain test on clear; 3 train on clear test on plain; 4: train on clear, test on clear
gender=0 #0: no gender; 1: train on male test on male; 2: train on male test on female; 3: train on female test on male; 4: train on female test on female
norm=1; #l1 or l2 norm in cost function
data_augment=False; #augment video data
frameCount=30
shp=16384 #length of audio from each video. This is the minimum.- 9600
###############
options='_cnn_crf_interpolate'

#load data
[stylelist, vowellist, subjlist, filelist]=np.load('demographics.npy')
[Xtrain, Xtrain_aud, Xtrain_aug] =np.load('AV_data'+options+'.npy')

vidx=[i for i, value in enumerate(subjlist) if (value =='1209' or value == '1109')] #validation indices
oidx=[i for i, value in enumerate(subjlist) if (value == '1201' or value =='1203' or value=='1210' or value =='1103' or value =='1101' or value=='1110')]  #test indices
tidx=[x for x in range(len(subjlist)) if x not in (vidx) and x not in (oidx)]

#select 100 random numbers between 0 to 2829 
#idx = np.random.choice(len(subjlist), 100, replace=False)



if region==2:
    landmarks_indices=[*range(4,13),*range(48,68)] #5:13 is lower jaw and 49:68 is lips
elif region ==1:
    landmarks_indices=[*range(48,68)] #mouth only
elif region ==0:
    landmarks_indices=[*range(4,13)] #jaw only
else:
    landmarks_indices=[*range(68)] #full face 


for_w= 10 #formant term weight in loss
grad_w= 1#grad term weight in loss
spec_w= 1 #spectrogram term weight in loss 
mel_w = 0#spec_w #weight on the mel term in spec loss
#use spec loss
spec=1 #use spec loss
mel=0 #use mel loss
cl_learning=1
apply_pre_emp=False #apply pre-emphasis filter
apply_reg=False #apply reg in loss function using average spectrogram

options=options+'_bsize_'+str(batch_size)+'_gender_'+str(gender)+'_style_'+str(style)+'_for_'+str(for_w)+'_spec_'+str(spec_w)+'_grad_'+str(grad_w)+'_mel_'+str(mel_w) #for model filename
##
#step 3: Create Average spectrogram
#plot spectrogram
#55 for subj1
#52 for sub2
#44 for sub3
sr=16000


#winlen=0.02,
#winstep=0.025,

_LOG_EPS = 1e-6
frame_length= 128 # np.int32(sr*0.005) #window len - 80
frame_step=np.int32(np.round(sr*0.002)) # time step -32 - increase value decreases X-ticks - 0.002 or 0.006
fft_len=512 # increase value increases y ticks 
epochs=2500
fac = fft_len/16

#from griffin lim paper: https://ieeexplore.ieee.org/document/1164317
#if the window length (L) is a multiple of four times the window shift (S)

print(frame_length, frame_step, fft_len)
if apply_pre_emp:
    alpha=0.8
    dyn_range=35
else:
    alpha=0.8
    dyn_range=70

def get_spectrogram(waveform):
    
    stfts = tf.signal.stft(np.float32(np.transpose(waveform)), frame_length, frame_step,fft_length=fft_len, pad_end=False)
    spectrograms = tf.abs(stfts)
    spec_pred_db=10*tf.math.log(spectrograms + _LOG_EPS)
    
    #adjust dynamic range
    mx=tf.math.reduce_max(spec_pred_db)
    spec_pred_db= tf.clip_by_value(spec_pred_db, mx-dyn_range, mx)
    spect = tf.nn.leaky_relu(spec_pred_db, alpha=alpha) 
    return spect
    
def get_mel_spec(waveform):
    
    #The window-size, hop-size and mel dimension are 800, 200, and 80 respectively.
    
    stfts = tf.signal.stft(np.float32(np.transpose(waveform)), frame_length, frame_step,fft_length=fft_len, pad_end=False)
    spectrograms = tf.abs(stfts)


    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 10.0, 3000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))
    spect = mel_spectrograms
    print(np.shape(mel_spectrograms))
    return spect


#nfft =1024 # sr/20; # fs/freq_step = 800
#3: coed
#4: keyed
#49 keyed
#48 cooed
#104 keyed
#compute average spectrogram
for no in range(300):
    sno=0+no
    #print(filelist[sno])
    waveform=Xtrain_aud[sno][:,0]
    #print(np.shape(waveform))
    ## pre emphasis filtering
    if apply_pre_emp:
        pre = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    else:
        pre=waveform

    #waveform=waveform[2500:16384,:]
    
    if 1:
        stfts = tf.signal.stft(np.float32(np.transpose(pre)), frame_length, frame_step,fft_length=fft_len, pad_end=False)
        spectrograms = tf.abs(stfts)

        spec_pred_db=10*tf.math.log(spectrograms + _LOG_EPS)
        #print(np.shape(spec_pred_db))
        mx=tf.math.reduce_max(spec_pred_db)
        spec_pred_db= tf.clip_by_value(spec_pred_db, mx-dyn_range, mx)
        spec_pred_db = tf.nn.leaky_relu(spec_pred_db, alpha=alpha)
        #spec_pred_db=mel_spectrograms
        if 1:
            if no==0:
                avg_spec = tf.abs(spec_pred_db)
            else:
                avg_spec = avg_spec+tf.abs(spec_pred_db)
    else:
        spec_pred_db=get_mel_spec(waveform)
    if apply_reg==True:
        avg_spec=avg_spec/300
        val=np.shape(avg_spec)[1]



#step 4: Define Loss function
#spec loss
import keras.backend as K
from librosa import util
from scipy.signal import get_window
import librosa
tf_sr=11000
if (tf_sr==16000):
    Nval = 19#tf.shape(p)[0]
else:
    Nval = 14#tf.shape(p)[0]


ar_coeffs=tf.Variable(tf.zeros(shape=(batch_size,Nval), dtype=tf.float32))
ar_coeffs_prev=tf.Variable(tf.zeros(shape=(batch_size,Nval),dtype=tf.float32))
den=tf.Variable(tf.zeros_like(tf.range(batch_size),dtype=tf.float32))
fwd_pred_error = tf.Variable(tf.fill([batch_size,shp-1],0.0), dtype=tf.float32, shape=tf.TensorShape(None)  ) #tf.shape(y)[1]
bwd_pred_error = tf.Variable(np.zeros((batch_size,shp-1)), dtype=tf.float32, shape=tf.TensorShape(None)  )


                    
def tflpc(y, order):
    """
    y: batchsize x frames
    order: order of LPC
    """
    dtype = y.dtype
    #TODO: Problem here: can not retrieve shape
    bsize=batch_size#tf.shape(y)[0]
    #tf.print("BEGIN tflpc") 
    if 1:
        global ar_coeffs
        global ar_coeffs_prev
        global den
        global fwd_pred_error
        global bwd_pred_error
        ar_coeffs.assign(np.concatenate((np.ones(shape=(bsize,1), dtype=np.float32),np.zeros(shape=(bsize,order), dtype=np.float32)), axis=1))
        ar_coeffs_prev.assign(np.concatenate((np.ones(shape=(bsize,1), dtype=np.float32),np.zeros(shape=(bsize,order), dtype=np.float32)), axis=1))

        #TODO: Problem here: can not retrieve shape so using "shp", not ideal
        #print("y shape ",shp, tf.shape(y),y)
        #fwd_pred_error = tf.Variable(lambda : tf.fill([bsize,shp-1],0.0), dtype=dtype ) #tf.shape(y)[1]
        #bwd_pred_error = tf.Variable(lambda : np.zeros((bsize,shp-1)), dtype=dtype )
        fwd_pred_error.assign(y[:,1:])
        bwd_pred_error.assign(y[:,:-1])
        den.assign(tf.reduce_sum(tf.multiply(fwd_pred_error, fwd_pred_error), axis=-1) + tf.reduce_sum(tf.multiply(bwd_pred_error, bwd_pred_error), axis=-1)) 
        #tf.Variable(lambda : tf.reduce_sum(tf.multiply(fwd_pred_error, fwd_pred_error), axis=-1) + tf.reduce_sum(tf.multiply(bwd_pred_error, bwd_pred_error), axis=-1))

        den=tf.cast(den, dtype=dtype)
        for i in range(order):
            #print(den)
            #TODO: Problem here: can not use if condition with tensor
            #if tf.cond(tf.reduce_any(tf.less_equal(den,0)), lambda:1, lambda:0):
            #    raise FloatingPointError("numerical error, input ill-conditioned?")

            reflect_coeff = (tf.constant(-2, dtype=dtype) * tf.reduce_sum(tf.multiply(bwd_pred_error, fwd_pred_error), axis=1) / den)

            ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
            for j in range(1, i + 2):
                #ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]
                temp=tf.add(ar_coeffs_prev[:,j], tf.multiply(reflect_coeff, ar_coeffs_prev[:,i - j + 1] ))
                ar_coeffs[:,j].assign(temp)
            fwd_pred_error_tmp = tf.identity(fwd_pred_error)

            for k in range(bsize):
                fwd_pred_error[k,:].assign(fwd_pred_error[k,:] + reflect_coeff[k] * bwd_pred_error[k,:])
                bwd_pred_error[k,:].assign(bwd_pred_error[k,:] + reflect_coeff[k] * fwd_pred_error_tmp[k,:])

            q = tf.constant(1, dtype=dtype) - reflect_coeff ** 2
            den.assign(q * den - bwd_pred_error[:,-1] ** 2 - fwd_pred_error[:,0] ** 2)

            #fwd_pred_error = tf.Variable(lambda : fwd_pred_error[:,1:])
            fwd_pred_error.assign(fwd_pred_error[:,1:])
            bwd_pred_error.assign(bwd_pred_error[:,:-1])
            #print("fwd_pred_errr ", tf.shape(fwd_pred_error))
            #bwd_pred_error = tf.Variable(lambda : bwd_pred_error[:,:-1])
        #tf.print("END tflpc") 
        return ar_coeffs
    else:
        return tf.random.uniform(shape=(bsize,order+1))
    #tf.as_dtype() 

yfilt=tf.Variable(tf.zeros(shape=(batch_size, shp)))
#Amat=tf.Variable(np.diag(np.ones(17,), k=-1), dtype=tf.complex64)
def tflfilter(b,a,x_win):
    """
    b = filter num
    a = filter den
    x_win = batchsize x frames
    """
    #tf.print("BEGIN tflfilter") 
    if 1:
        b=1.
        a=[1., 0.63]
        i = tf.constant(1)
        yfilt.assign(tf.zeros(shape=(batch_size, shp)))
        yfilt[:,0].assign(x_win[:,0])

        dtype=yfilt.dtype
        while_condition = lambda i: tf.less(i, tf.shape(x_win)[1])
        def body(i):
            temp = tf.subtract(tf.multiply(b,tf.cast(x_win[:,i], dtype=dtype)), tf.multiply(a[1], tf.cast(yfilt[:,i-1], dtype=dtype) ))
            yfilt[:,i].assign(temp)
            return [tf.add(i, 1)]

        #for each frame
        if 1:
            # do the loop:
            r = tf.while_loop(while_condition, body, [i])
        else:
            for i in range(1,tf.shape(x_win)[1]):
                temp = tf.subtract(tf.multiply(b,tf.cast(x_win[:,i], dtype=dtype)), tf.multiply(a[1], tf.cast(y[:,i-1], dtype=dtype) ))
                yfilt[:,i].assign(temp)
        #tf.print("END tflfilter") 
        return yfilt
    #temp = tf.subtract(tf.multiply(b,x[i]), tf.multiply(a[1], tf.float32(y[ i-1])) ) / a[0]
                            

rootsbs=tf.Variable(tf.zeros(shape=(batch_size, Nval), dtype=tf.complex64))
Amat=tf.Variable(tf.zeros(tf.shape(np.diag(np.ones(Nval-2,), k=-1)), dtype=tf.float32))
def tfroots(pbs):
    #tf.print("BEGIN tfroots") 
    #print("pbs: ", tf.shape(pbs))
    if 1:
        global rootsbs
        global Amat
        #rootsbs.assign(tf.zeros(tf.shape(pbs)))
        for i in range(batch_size):
            p=pbs[i,:]
            #TODO: Problem here: can not retrieve shape
            if Nval > 1:
                # build companion matrix and find its eigenvalues (the roots)
                Amat.assign(np.diag(np.ones(Nval-2,), k=-1))
                #Amat=tf.cast(Amat,dtype=p.dtype)
                temp = -p[1:] / p[0]
                Amat[0,:].assign(temp)
                estimated_roots = tf.linalg.eigvals(Amat)
                roots=estimated_roots
                #TODO: Problem here: can not iterate over tensor
                #roots = [r for r in estimated_roots if tf.math.imag(r) >= 0]
                rootsbs.assign(tf.cast(rootsbs, dtype=estimated_roots.dtype))
                rootsbs[i,0:tf.shape(roots)[0]].assign(roots)
            else:
                roots = []
        #tf.print("END tfroots") 
        return rootsbs
    else:
        return tf.random.uniform(shape=(tf.shape(pbs)))

def tfformants(x, sample_rate):
    """
    x = batchsize x frames
    sample_rate = sampling rate of the audio signal
    """
    #tf.print("BEGIN tfformants") 
    waveform=x
    if 1:
        length=tf.shape(waveform)[1]
        hamming_win = tf.signal.hamming_window(length)
        # Apply window and high pass filter.
        x_win = tf.multiply(waveform, hamming_win)

        x_filt = tflfilter([1], [1.0, 0.63], x_win)
        lpc_rep = tflpc(x_filt, 2 + int(sample_rate / 1000))

        roots = tfroots(lpc_rep)

        #print("roots: ", tf.shape(roots))
        angles = tf.math.atan2(tf.math.imag(roots), tf.math.real(roots))
        indices=tf.argsort(angles,axis=-1,direction='ASCENDING',stable=False,name=None)
        ffreq=(angles * (sample_rate / (2 * np.math.pi)))
        bw = -1.0*(sample_rate/(2*np.math.pi))*tf.math.log(tf.abs(roots));
        #bw = -0.5*(sample_rate/(2*np.math.pi))*tf.math.log(tf.abs(roots));
        bwtf=tf.gather(bw,indices, axis=1)
        #frtf=tf.gather(frqs,indices)
        
        
        #[frqs,indices] = tf.sort(angles*(sample_rate/(2*np.math.pi)));
        #bw = -1/2*(sample_rate/(2*np.math.pi))*tf.log(tf.abs(roots[indices]));
        #print(frqs,bw)
        #tf.print("END tfformants") 
        return  ffreq, bw, indices
    else:
        length=tf.shape(waveform)[1]
        hamming_win = tf.signal.hamming_window(length)
        # Apply window and high pass filter.
        x_win = tf.multiply(waveform, hamming_win)
        x_filt = tflfilter([1], [1.0, 0.63], x_win)
        lpc_rep = tflpc(x_filt, 2 + int(sample_rate / 1000))
        return tf.random.uniform(shape=(batch_size,9))





_LOG_EPS = 1e-6
samp_rate=sr
nfft = fft_len

#frame_length=1024#np.int32(samp_rate*0.005) #window len
#frame_step=32#np.round(samp_rate*0.002) # time step
#nfft =1024 # sr/20; # fs/freq_step = 800

#Window len: 0.005s
#Max freq=5000Hz
#Time step= 0.002s
#Frequency step=20
#Window shape= gaussian

def normal_dist(x , mean , sd, binwidth, tbin,spec_true_db):
    n=tf.cast((mean/binwidth-0.5), dtype=tf.int32) # (n+1/2)*bw = fc
    amp=spec_true_db[0, tbin,n] #pick the amp at time=time/2

    sd=sd+1e-6
    prob_density = (3.14*sd) * tf.math.exp(-0.5*((x-mean)/sd)**2)
    return tf.math.sqrt(amp)*prob_density 


#def normal_dist(x , mean , sd):
#    sd=sd+1e-6
#    prob_density = (3.14*sd) * tf.math.exp(-0.5*((x-mean)/sd)**2)
#    return prob_density   


def tf_cov(ten):
    x = ten#tf.reshape(ten, [-1])
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

class changeAlpha(tf.keras.callbacks.Callback):
    beta=2
    def on_epoch_end(self, epoch, logs={}):
        self.beta = epoch
        #K.set_value(self.beta, K.get_value(epoch))

def mel_mse_loss(frame_step, fft_len):
    def mel_loss(y_true,y_pred):
        #y_pred shape (32, 16384, 1) y_true shape(32, 16384, 1)
        # make sure the input to the stft is batch_size x time points (2D) rather than bs x time points x 1
        mse = MeanSquaredError()
        #256,128
        waveform = y_pred[:,:,0]

        if apply_pre_emp:
            pre = tf.concat([tf.reshape(waveform[:,0],[batch_size,1]), waveform[:,1:] - 0.97 * waveform[:,:-1]], axis=1)
        else:
            pre=waveform
        stfts_pred = tf.signal.stft(pre, np.int32(frame_length), np.int32(frame_step), fft_length=np.int32(fft_len),window_fn=tf.signal.hamming_window,pad_end=False)


        #stfts_pred = tf.signal.stft(y_pred[:,:,0], np.int32(frame_length), np.int32(frame_step), fft_length=np.int32(nfft),window_fn=tf.signal.hamming_window,pad_end=False)
        spectrograms_pred = tf.math.abs(stfts_pred) 

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts_pred.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 7600.0, 100
        linear_to_mel_weight_matrix_pred = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
          upper_edge_hertz)
        spec_pred_db = tf.tensordot(
          spectrograms_pred, linear_to_mel_weight_matrix_pred, 1)
        spec_pred_db.set_shape(spectrograms_pred.shape[:-1].concatenate(
          linear_to_mel_weight_matrix_pred.shape[-1:]))

        log_mel_spectrograms = tf.math.log(spec_pred_db + 1e-6)
        
        #batchsize x #mfccs x #mel bins
        #(32, 128, 80)
        mfccs_pred = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mel_bins]


        
        waveform = y_true[:,:,0]
        if apply_pre_emp:
            pre = tf.concat([tf.reshape(waveform[:,0],[batch_size,1]), waveform[:,1:] - 0.97 * waveform[:,:-1]], axis=1)
        else:
            pre=waveform
        stfts_true = tf.signal.stft(pre, np.int32(frame_length), np.int32(frame_step), fft_length=np.int32(fft_len),window_fn=tf.signal.hamming_window,pad_end=False)


        #stfts_true = tf.signal.stft(y_true[:,:,0], np.int32(frame_length), np.int32(frame_step), fft_length=np.int32(nfft),window_fn=tf.signal.hamming_window,pad_end=False)
        spectrograms_true = tf.math.abs(stfts_true) 

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts_true.shape[-1]

        linear_to_mel_weight_matrix_true = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
          upper_edge_hertz)
        spec_true_db = tf.tensordot(
          spectrograms_true, linear_to_mel_weight_matrix_true, 1)
        spec_true_db.set_shape(spectrograms_true.shape[:-1].concatenate(
          linear_to_mel_weight_matrix_true.shape[-1:]))
        
        log_mel_spectrograms = tf.math.log(spec_true_db + 1e-6)
        
        mfccs_true = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mel_bins]
        print(np.shape(mfccs_true))
        #debug
        #print(np.shape(spec_pred_db))

        if (norm==2):
            loss=mse(mfccs_pred ,mfccs_true)
        else:
            loss=K.mean(K.sum(K.abs(spec_pred_db - spec_true_db), axis=1),axis=-1) 
        return loss
        #mse(mfccs_pred ,mfccs_true)
        #return K.mean(K.sum(K.square(K.abs(spec_pred_db - spec_true_db)), axis=1),axis=-1) 

    return mel_loss

def formant_loss(y_true, spec_true_db):
     ## FORMANT BASED LOSS TERM
    #ytrue = bs x 16384 x 1
    #resample audio pretending it is an image; height width channels
    #tf.print("BEGIN formant_loss") 
    modaud= tf.image.resize(y_true,[batch_size,tf_sr])

    f_temp, bw_temp, indices=tfformants(y_true[:,:,0], tf_sr)
    #tf.print("Vector temp: ", tf.shape(f_temp))
    #SORT the formants in ascending order
    ii, jj = tf.meshgrid(tf.range(tf.shape(f_temp)[0]),
                        tf.range(tf.shape(f_temp)[1]),
                            indexing='ij')
    # Stack complete index
    index = tf.stack([ii, indices], axis=-1)
    #tf.print(tf.shape(f_temp), tf.shape(index), tf.reduce_max(ii), tf.reduce_max(indices))
    #tf.print(tf.shape(bw_temp), tf.shape(index))
    f_gt=tf.gather_nd(f_temp,index, batch_dims=0)
    bw_gt=tf.gather_nd(bw_temp,index, batch_dims=0)

    #COMPUTE 1D profile curves from formants and BW
    # 32 (bs) x 509 (time) x 257 (freq)
    Nbins=tf.shape(spec_true_db)[2]
    tbin=tf.cast((tf.shape(spec_true_db)[1]/2), dtype=tf.int32)
    binwidth=tf.cast(samp_rate/(2*Nbins), dtype=f_gt.dtype)
    xspace = tf.linspace(1., 5500., num=12000)

    #replace < 50hz (margin in PRAAT) with a large number; idea is to pick the first positive number
    temp=tf.where(tf.math.less_equal(f_gt[0],tf.constant([50], dtype=f_gt.dtype)), sr*tf.ones_like(f_gt[0]), f_gt[0])
    ind=tf.argsort(tf.math.abs(temp),axis=-1,direction='ASCENDING',stable=False,name=None)
    stidx=ind[0] #skip 0 as 0 index has 0 value
    #tf.print("Formant Values: ",f_gt[0], ind, stidx)
    

    if 1:
        pdf_bs=tf.cond(stidx<Nval,lambda: normal_dist(xspace, f_gt[0,stidx], bw_gt[0,stidx], binwidth,tbin, spec_true_db),lambda: tf.zeros_like(xspace))
        pdf_bs+=tf.cond(stidx+1<Nval,lambda: normal_dist(xspace, f_gt[0,stidx+1], bw_gt[0,stidx+1], binwidth,tbin, spec_true_db),lambda: tf.zeros_like(xspace))
        pdf_bs+=tf.cond(stidx+2<Nval,lambda: normal_dist(xspace, f_gt[0,stidx+2], bw_gt[0,stidx+2], binwidth,tbin, spec_true_db),lambda: tf.zeros_like(xspace))
    else:
        n=tf.cast((f_gt[0,stidx]/binwidth-0.5), dtype=tf.int32) # (n+1/2)*bw = fc
        amp=spec_true_db[0, tbin,n] #pick the amp at time=time/2
        pdf_bs= tf.math.sqrt(amp)*normal_dist(xspace, f_gt[0,stidx], bw_gt[0,stidx]) #first formant
        #tf.print("stidx0: ", stidx, amp, f_gt[0,stidx], bw_gt[0,stidx])
    
        n=tf.cast((f_gt[0,stidx+1]/binwidth-0.5), dtype=tf.int32) # (n+1/2)*bw = fc
        amp=spec_true_db[0, tbin,n] #pick the amp at time=time/2
        pdf_bs+= tf.math.sqrt(amp)*normal_dist(xspace, f_gt[0,stidx+1], bw_gt[0,stidx+1]) #second formant
        #tf.print("stidx0: ", stidx, amp, f_gt[0,stidx+1], bw_gt[0,stidx+1])
    
        n=tf.cast((f_gt[0,stidx+2]/binwidth-0.5), dtype=tf.int32) # (n+1/2)*bw = fc
        amp=spec_true_db[0, tbin,n] #pick the amp at time=time/2
        pdf_bs+= tf.math.sqrt(amp)*normal_dist(xspace, f_gt[0,stidx+2], bw_gt[0,stidx+2]) #third formant
        #tf.print("stidx0: ", stidx, amp, f_gt[0,stidx+2], bw_gt[0,stidx+2])
    
    pdf_bs=tf.expand_dims(pdf_bs, axis=0)
    for bs in range(batch_size-1):
        temp=tf.where(tf.math.less_equal(f_gt[bs+1],tf.constant([0], dtype=f_gt.dtype)), sr*tf.ones_like(f_gt[bs+1]), f_gt[bs+1])
        ind=tf.argsort(tf.math.abs(temp),axis=-1,direction='ASCENDING',stable=False,name=None)
        #ind=tf.argsort(tf.math.abs(f_gt[bs+1]),axis=-1,direction='ASCENDING',stable=False,name=None)
        stidx=ind[0] #skip 0 as 0 index has 0 value
        #tf.print("BS: Formant Values: ",bs, ind, stidx)
        if 1:
            pdf=tf.cond(stidx+0<Nval,lambda: normal_dist(xspace, f_gt[bs+1,stidx], bw_gt[bs+1,stidx], binwidth,tbin, spec_true_db),lambda: tf.zeros_like(xspace))
            pdf+=tf.cond(stidx+1<Nval,lambda: normal_dist(xspace, f_gt[bs+1,stidx+1], bw_gt[bs+1,stidx+1], binwidth,tbin, spec_true_db),lambda: tf.zeros_like(xspace))
            pdf+=tf.cond(stidx+2<Nval,lambda: normal_dist(xspace, f_gt[bs+1,stidx+2], bw_gt[bs+1,stidx+2], binwidth,tbin, spec_true_db),lambda:  tf.zeros_like(xspace))
        else:
            n=tf.cast((f_gt[bs+1,stidx]/binwidth-0.5), dtype=tf.int32) # (n+1/2)*bw = fc
            amp=spec_true_db[bs+1, tbin,n] #pick the amp at time=time/2
            pdf = tf.math.sqrt(amp)*normal_dist(xspace, f_gt[bs+1,stidx], bw_gt[bs+1,stidx])
            #tf.print("stidx: ", bs, amp, f_gt[bs+1,stidx], bw_gt[bs+1,stidx])
            n=tf.cast((f_gt[bs+1,stidx+1]/binwidth-0.5), dtype=tf.int32) # (n+1/2)*bw = fc
            amp=spec_true_db[bs+1, tbin,n] #pick the amp at time=time/2
            pdf += tf.math.sqrt(amp)*normal_dist(xspace, f_gt[bs+1,stidx+1], bw_gt[bs+1,stidx+1])
            #tf.print("stidx: ", bs, amp, f_gt[bs+1,stidx+1], bw_gt[bs+1,stidx+1])
            n=tf.cast((f_gt[bs+1,stidx+2]/binwidth-0.5), dtype=tf.int32) # (n+1/2)*bw = fc
            amp=spec_true_db[bs+1, tbin,n] #pick the amp at time=time/2            
            pdf += tf.math.sqrt(amp)*normal_dist(xspace, f_gt[bs+1,stidx+2], bw_gt[bs+1,stidx+2])
            #tf.print("stidx: ", bs, amp, f_gt[bs+1,stidx+2], bw_gt[bs+1,stidx+2])
        pdf=tf.expand_dims(pdf, axis=0)
        pdf_bs=tf.concat([pdf_bs, pdf], axis=0)
    #tf.print("Vector: ", f_gt[0,stidx:])
    #tf.print("END formant_loss") 
    return pdf_bs
    ##FORMANT BASED LOSS END
        

# loss = w1 * low_res + w2 * mid_res + w3 * high_res
def spec_mse_loss(frame_step, fft_len):
    def spec_loss(y_true,y_pred):
        #y_pred shape (32, 16384, 1) y_true shape(32, 16384, 1)
        # make sure the input to the stft is batch_size x time points (2D) rather than bs x time points x 1

        #256,128
        waveform = y_pred[:,:,0]

        if apply_pre_emp:
            pre = tf.concat([tf.reshape(waveform[:,0],[batch_size,1]), waveform[:,1:] - 0.97 * waveform[:,:-1]], axis=1)
        else:
            pre=waveform
        spec_pred = tf.signal.stft(pre, np.int32(frame_length), np.int32(frame_step), fft_length=np.int32(fft_len),window_fn=tf.signal.hamming_window,pad_end=False)
        X_mag_pred = tf.math.abs(spec_pred)     
        spec_pred_db = 10*tf.math.log(X_mag_pred + _LOG_EPS)
        if apply_reg:
            spec_reg_term = tf.abs(spec_pred_db)
        mx=tf.math.reduce_max(spec_pred_db)
        spec_pred_db= tf.clip_by_value(spec_pred_db, mx-dyn_range, mx)
        spec_pred_db = tf.nn.leaky_relu(spec_pred_db, alpha=alpha) # set negative values close to zero as we want to match bright regions only


        waveform = y_true[:,:,0]

        if apply_pre_emp:
            pre = tf.concat([tf.reshape(waveform[:,0],[batch_size,1]), waveform[:,1:] - 0.97 * waveform[:,:-1]], axis=1)
        else:
            pre=waveform
        spec_true = tf.signal.stft(pre, np.int32(frame_length), np.int32(frame_step), fft_length=np.int32(fft_len),window_fn=tf.signal.hamming_window,pad_end=False)   
        X_mag_true = tf.math.abs(spec_true)
        spec_true_db = 10*tf.math.log(X_mag_true + _LOG_EPS)
        mx=tf.math.reduce_max(spec_true_db)
        spec_true_db= tf.clip_by_value(spec_true_db, mx-dyn_range, mx)
        spec_true_db = tf.nn.leaky_relu(spec_true_db, alpha=alpha)

        #debug
        #print(np.shape(spec_pred_db))
        #print(np.shape(spec_true_db))
        #print(np.shape(y_pred),np.shape(y_true))

        #if(np.shape(spec_true_db)[1] == 128):
            #plt.imshow(spec_true_db[1,:,:])
            #plt.imshow(spec_pred_db[1,:,:])

        #12 481 513
        #1,510,257 <-- with praat params
        if apply_reg:
            avg_spec_db = tf.nn.leaky_relu(avg_spec, alpha=alpha)
            X=tf.reshape(spec_reg_term,[batch_size,np.shape(spec_pred_db)[1]*np.shape(spec_pred_db)[2] ])
            Y=tf.reshape(avg_spec_db,[np.shape(spec_pred_db)[1]*np.shape(spec_pred_db)[2] ])
            #covY=tf_cov(Y)
            #temp=tf.matmul(tf.math.subtract(X,Y), tf.linalg.inv(covY))
            #term = 0.15*tf.matmul(temp, tf.math.subtract(X,Y))
            reg_term = 0.1*K.mean(K.sum(K.square(K.abs(spec_reg_term- avg_spec_db)), axis=1),axis=-1)
        else:
            reg_term=0
        mse = MeanSquaredError()
        
        spec_reshape=tf.expand_dims(spec_pred_db,axis=3)
        grad_image = tf.image.image_gradients(spec_reshape)
        grad_predict=grad_image[1] #along y axis
        #print(np.shape(grad_image), np.shape(grad_predict))   
        spec_reshape=tf.expand_dims(spec_true_db,axis=3)
        grad_image = tf.image.image_gradients(spec_reshape)
        grad_true=grad_image[1] #along y axis
 
        ## FORMANT BASED LOSS TERM
        #remove nans in the data; otherwise returns error
        #y_pred = tf.clip_by_value(y_pred, -1e7, 1e7)
        #X_mag_pred= tf.clip_by_value(X_mag_pred, -1e7, 1e7)
        y_pred = tf.where(tf.math.is_nan(y_pred), tf.ones_like(y_pred), y_pred)
        X_mag_pred = tf.where(tf.math.is_nan(X_mag_pred), tf.ones_like(X_mag_pred), X_mag_pred)
        
        if True:
            pdf_pred = formant_loss(y_pred, X_mag_pred)
            pdf_true = formant_loss(y_true, X_mag_true)
        #tf.print("Begin loss_formant: ", tf.shape(pdf_true), tf.shape(pdf_pred)) 
        #norm factor of 500..the values of loss are in thoussands
        loss_formant = mse(pdf_true, pdf_pred)/500  #1/500 is the weight
        #tf.print(pdf_true)
        #tf.print(pdf_pred)
        tf.print("Loss formant without normalization: ", tf.math.log(loss_formant))
        loss_formant = tf.clip_by_value(tf.math.log(loss_formant),0, 60)
        #print(np.shape(pdf_bs)," spec_pred ", print(tf.shape(spec_true_db)))
        
        #f_gt = 32 x 19, bw_gt=32x19
        
        #mse of spec is in 100s and mse of grad is in range 3
        #loss_mel = mel_mse_loss(np.int32(np.round(sr*0.002*4)), 128)
        loss_mel = mel_mse_loss(frame_step, fft_len)
        tf.print("formants: ", loss_formant," mean: ", mse(grad_predict, grad_true)/10, mse(spec_pred_db, spec_true_db)/20, tf.math.reduce_mean(loss_mel(y_true,y_pred))/5)
        

        return for_w*loss_formant + grad_w*mse(grad_predict, grad_true)/10+ spec_w* mse(spec_pred_db, spec_true_db)/20 +mel_w* tf.math.reduce_mean(loss_mel(y_true,y_pred))/5
        #return  K.mean(K.sum(K.square(K.abs(spec_pred_db - spec_true_db)), axis=1),axis=-1) +reg_term
    return spec_loss


#step 5: Define audio synthesis Network

## DC gans for sound
## create generator
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow import expand_dims
from tensorflow import shape, image
from tensorflow import reshape
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


##step 6: Start Traing network
#(1349, 16384, 1) (1349, 80, 58)
print(np.shape(Xtrain_aud), np.shape(Xtrain))

#859 starst 1107
#1023 starts 1109
# 1086 starts 1110

#get plain style indices


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


if data_augment:
    temp = np.concatenate((Xtrain[train_idx],Xtrain_aug[train_idx]), axis=0)
    Xtrain_vid_all = np.asarray(np.float32(temp))
    temp_aud = np.concatenate((Xtrain_aud[train_idx],Xtrain_aud[train_idx]), axis=0)
    Xtrain_aud_all =np.asarray(np.float32(temp_aud))
else:
    Xtrain_aud_all=np.asarray(np.stack(Xtrain_aud).astype(np.float32))[train_idx]
    Xtrain_vid_all=np.asarray(np.stack(Xtrain).astype(np.float32))[train_idx]
    #Xtrain_vid_all=np.asarray(np.float32(Xtrain))[train_idx]

#bs x frameCount x landmark_indices*2
#apply which coordinates/rois to use for training
print("Before: ", np.shape(Xtrain_aud_all), np.shape(Xtrain_vid_all), np.shape(val_idx))
Xtrain_vid_all = Xtrain_vid_all[0:875,:,np.concatenate((landmarks_indices,np.array(landmarks_indices)+68))]
Xtrain_aud_all=Xtrain_aud_all[0:875]


#configure these
Xtrain_aud_val=np.asarray(np.stack(Xtrain_aud).astype(np.float32))[val_idx]
Xtrain_vid_val=np.asarray(np.stack(Xtrain).astype(np.float32))[val_idx]
Xtrain_vid_val = Xtrain_vid_val[0:175,:,np.concatenate((landmarks_indices,np.array(landmarks_indices)+68))]
Xtrain_aud_val=Xtrain_aud_all[0:175]
print("After: ",np.shape(Xtrain_aud_all), np.shape(Xtrain_vid_all), np.shape(Xtrain_vid_val))

outfolder='./results/'
fname='best_model'
fname=fname+options


#without curriculum learning
C0=[] 
C0.append( tf.keras.callbacks.ModelCheckpoint(filepath='%s%s_fn_C0.h5'% (outfolder,fname),monitor='val_loss',save_best_only=True))


#with curriculum learning
C1=[] 
C1.append( tf.keras.callbacks.ModelCheckpoint(filepath='%s%s_fn_C1.h5'% (outfolder,fname),monitor='val_loss',save_best_only=True))

C2=[] 
C2.append( tf.keras.callbacks.ModelCheckpoint(filepath='%s%s_fn_C2.h5'% (outfolder,fname),monitor='val_loss',save_best_only=True))


C3=[] 
C3.append( tf.keras.callbacks.ModelCheckpoint(filepath='%s%s_fn_C3.h5'% (outfolder,fname),monitor='val_loss',save_best_only=True))


if cl_learning:
    #low resolution
    #mel_mse_loss or spec_mse_loss

    if spec:
        generator.compile(loss=spec_mse_loss(np.int32(np.round(sr*0.002*4)), 512/4), optimizer=gopt,metrics=['mse'])
        history1 = generator.fit(Xtrain_vid_all, Xtrain_aud_all,
                            batch_size=batch_size,
                            epochs=150,
                            validation_data=(Xtrain_vid_val, Xtrain_aud_val),
                            verbose=2,
                            callbacks=C1)
        generator.load_weights("./results/"+fname+'_fn_C1.h5')
    if mel:
        generator.compile(loss=mel_mse_loss(np.int32(np.round(sr*0.002*4)), 512/4), optimizer=gopt,metrics=['mse'])
        history1_1 = generator.fit(Xtrain_vid_all, Xtrain_aud_all,
                            batch_size=batch_size,
                            epochs=150,
                            validation_data=(Xtrain_vid_val, Xtrain_aud_val),
                            verbose=2,
                            callbacks=[])


    #med resolution
    if spec:
        generator.compile(loss=spec_mse_loss(np.int32(np.round(sr*0.002*2)), 512/2), optimizer=gopt,metrics=['mse'])
        history2 = generator.fit(Xtrain_vid_all, Xtrain_aud_all,
                            batch_size=batch_size,
                            epochs=150,
                            validation_data=(Xtrain_vid_val, Xtrain_aud_val),
                            verbose=2,
                            callbacks=C2)
        generator.load_weights("./results/"+fname+'_fn_C2.h5')

    if mel:
        generator.compile(loss=mel_mse_loss(np.int32(np.round(sr*0.002*4)), 512/4), optimizer=gopt,metrics=['mse'])
        history2_2 = generator.fit(Xtrain_vid_all, Xtrain_aud_all,
                            batch_size=batch_size,
                            epochs=150,
                            validation_data=(Xtrain_vid_val, Xtrain_aud_val),
                            verbose=2,
                            callbacks=[])

    #high resolution
    if spec:
        generator.compile(loss=spec_mse_loss(np.int32(np.round(sr*0.002)), 512), optimizer=gopt,metrics=['mse'])
        history3 = generator.fit(Xtrain_vid_all, Xtrain_aud_all,
                            batch_size=batch_size,
                            epochs=200,
                            validation_data=(Xtrain_vid_val, Xtrain_aud_val),
                            verbose=2,
                            callbacks=C3)

    if mel:
        generator.compile(loss=mel_mse_loss(np.int32(np.round(sr*0.002*4)), 512/4), optimizer=gopt,metrics=['mse'])
        history3_3 = generator.fit(Xtrain_vid_all, Xtrain_aud_all,
                            batch_size=batch_size,
                            epochs=200,
                            validation_data=(Xtrain_vid_val, Xtrain_aud_val),
                            verbose=2,
                            callbacks=[])

else:
    #high resolution
    generator.compile(loss=spec_mse_loss(np.int32(np.round(sr*0.002)), 512), optimizer=gopt,metrics=['mse'])
    history3 = generator.fit(Xtrain_vid_all, Xtrain_aud_all,
                        batch_size=batch_size,
                        epochs=500,
                        validation_data=(Xtrain_vid_val, Xtrain_aud_val),
                        verbose=2,
                        callbacks=C0)


#print('\nhistory dict:', history.history)
#mel loss - 832/832 - 8s - loss: 15.5748 - mae: 0.2019 - val_loss: 10.1861 - val_mae: 0.3856
#spec loss- 832/832 - 7s - loss: 177.5917 - mae: 0.1902 - val_loss: 156.8559 - val_mae: 0.3496


#
