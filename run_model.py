"""
Created on Tue May 28 12:54:43 2019

@author: Shreyas Seshadri
"""

from __future__ import print_function
import librosa
import scipy
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import tensorflow as tf
import time
import SylNet_model 
import math


############ CHANGE THIS TO YOUR CURRENT PATH #############
mainFile = '/l/seshads1/code/git/SylNet-tmp/'
modelFile = mainFile + '/trained_models/'
means_path = modelFile + 'means.npy'
std_path = modelFile + 'stds.npy'
model_path = modelFile + 'model_trained.ckpt'
res_path = mainFile + 'results.npy'

########### GET DATA #############

fileList = list(filter(bool,[line.rstrip('\n') for line in open('config_files_test.txt')]))

############### HARD CODED AS THIS IS WHAT THE MAIN MODEL IS TRAINED ON ##############
maxT = 91  
######################################################################################33

noUtt_main = len(fileList)
Fs = 16000
w_l = round(0.025*Fs)
w_h = round(0.01*Fs)
X = np.ndarray((noUtt_main,),dtype=object)
for i in range(noUtt_main):
    fs, y = scipy.io.wavfile.read(fileList[i])
    y = y/max(abs(y))
    y = librosa.core.resample(y=y, orig_sr=fs, target_sr=Fs)    
    X[i] = np.transpose(20*np.log10(librosa.feature.melspectrogram(y=y, sr=Fs, n_mels=24, n_fft=w_l, hop_length=w_h)))
    
MEAN = np.load(means_path)  
STD = np.load(std_path) 

for i in range(noUtt_main):
    X[i] = (X[i] - MEAN)/STD

print('DATA LOADING DONE')

########### TRAIN MODEL #############

def padarray(A, size):
    t = size - A.shape[0]
    if t==0:
        r = A        
    else:
        r =np.pad(A,[(0,t),(0,0)],'constant')           
    return r

tf.reset_default_graph()

## PARAMETERS
residual_channels = 128 
filter_width = 5
dilations = [1]*10
input_channels = X[0].shape[1]
output_channels = maxT
postnet_channels= 128
droupout_rate = 0.5

ids = tf.placeholder(shape=(None, 2), dtype=tf.int32)
ids_len = tf.placeholder(shape=(None), dtype=tf.int32)
is_train = tf.placeholder(dtype=tf.bool)
S = SylNet_model.CNET(name='S', 
                   input_channels=input_channels,
                   output_channels=output_channels,
                   residual_channels=residual_channels,
                   filter_width=filter_width,
                   dilations=dilations,
                   postnet_channels=postnet_channels,
                   cond_dim=None,
                   do_postproc=True,
                   do_GLU=True,
                   endList=ids,
                   seqLen=ids_len,
                   isTrain=is_train,
                   DRrate=droupout_rate)



# data placeholders of shape (batch_size, timesteps, feature_dim)
x = tf.placeholder(shape=(None, None, input_channels), dtype=tf.float32)
logits = S.forward_pass(x)

predictions = tf.nn.sigmoid(logits)
init=tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)

#saveFile1 = mainFile + 'res.mat'
saver = tf.train.Saver()


stpCrit_win = 10
stpCrit_min = 30
batchSize = 32
noBatches = math.floor(X.shape[0]/batchSize)

epoch=0
dontStop=1
with tf.Session(config=tfconfig) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 
    saver.restore(sess, model_path)
                    
    #test pre-trained model on val data
    no_utt = X.shape[0]
    PRED = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):
        #print(n_val)
        X_mini = X[n_val]
        #print(X_mini.shape)
        X_mini = X_mini[np.newaxis,:,:]
        l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)            
        E_list = np.asarray([[0,X_mini.shape[1]-1]])                    
        PRED[n_val] = sess.run([predictions], feed_dict={x: X_mini, ids:E_list, ids_len:l_mini,is_train:False})  #PRED[n_val]
Y = np.zeros(no_utt)
for n_val in range(no_utt):
    #Y[n_val] = np.argmax(PRED[n_val])
    Y[n_val] = sum(sum(np.asarray(PRED[n_val][0]>=0.5, dtype=np.float32)))
np.save(res_path, Y)        