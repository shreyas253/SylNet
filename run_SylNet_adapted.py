"""
Created on Tue May 28 12:54:43 2019 (SS)
Modified on Wed June 5 12:30:00 2019 (OR)

@author: Shreyas Seshadri, Okko Rasanen
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
import os
import sys
import glob



########### PATHS IN USED IN CODE ###########
mainFile = os.path.dirname(os.path.realpath(__file__))
modelFile = mainFile + '/trained_models/'
means_path = modelFile + 'means.npy'
std_path = modelFile + 'stds.npy'
model_path = modelFile + 'model_trained_adap.ckpt'
config_path = mainFile + '/config_files/'

# Parse input arguments


if(len(sys.argv) == 1): # No extra arguments, use default input and output files
    file_list =  config_path + 'config_files_run.txt'
    res_path = mainFile + '/results.txt'
elif(len(sys.argv) == 2):  # Custom list of files for input
    res_path = mainFile + '/results.txt'
    if(isinstance(sys.argv[1], str)):
        file_list = sys.argv[1]
    else:
        raise Exception('First input argument must be a string (file path)')
elif(len(sys.argv) == 3): # Custom list of input files + custom result file
    if(isinstance(sys.argv[1], str)):
        file_list = sys.argv[1]
    else:
        raise Exception('First input argument must be a string (file path)')
    if(isinstance(sys.argv[2], str)):
        res_path = sys.argv[2]
    else:
        raise Exception('Second argument must be a string (file path)')


########### GET DATA #############
wasdir = 0
# If input
if(os.path.isdir(file_list)):
    fileList = sorted(glob.glob(file_list + '*.wav'))
    wasdir = 1
    if(len(fileList) == 0):
        raise Exception('Provided directory contains no .wav files')
elif(file_list.endswith('.wav')):
    fileList = list()
    fileList.append(file_list)
else:
    if(os.path.isfile(file_list)):
        fileList = list(filter(bool,[line.rstrip('\n') for line in open(file_list)]))
    else:
        raise Exception('Provided input file list does not exist.')


maxT = 91   # HARD CODED AS THIS IS WHAT THE MAIN MODEL IS TRAINED ON

noUtt_main = len(fileList)
Fs = 16000
w_l = round(0.025*Fs)
w_h = round(0.01*Fs)
X = np.ndarray((noUtt_main,),dtype=object)
for i in range(noUtt_main):
    if(os.path.isfile(fileList[i]) == False):
        raise Exception('A file in the given file list does not exist.')
    #fs, y = scipy.io.wavfile.read(fileList[i])
    #y = y/max(abs(y))
    #y = librosa.core.resample(y=y, orig_sr=fs, target_sr=Fs)
    y, _ = librosa.core.load(fileList[i], sr = Fs, mono = True)
    y = y/max(abs(y))
    X[i] = np.transpose(20*np.log10(librosa.feature.melspectrogram(y=y, sr=Fs, n_mels=24, n_fft=w_l, hop_length=w_h)+0.00000000001))


MEAN = np.load(means_path)  # Z norm data based on training data mean and std
STD = np.load(std_path)
for i in range(noUtt_main):
    X[i] = (X[i] - MEAN)/STD

print('DATA LOADING DONE')

########### LOAD MODEL #############

def padarray(A, size):
    t = size - A.shape[0]
    if t==0:
        r = A
    else:
        r =np.pad(A,[(0,t),(0,0)],'constant')
    return r

tf.reset_default_graph()

# PARAMETERS
residual_channels = 128
filter_width = 5
dilations = [1]*10
input_channels = X[0].shape[1]
output_channels = maxT
postnet_channels= 128
droupout_rate = 0.5
stpCrit_win = 10
stpCrit_min = 30
batchSize = 32
noBatches = math.floor(X.shape[0]/batchSize)


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

saver = tf.train.Saver()

# GET OUTPUTS
epoch=0
dontStop=1
with tf.Session(config=tfconfig) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver.restore(sess, model_path)

    no_utt = X.shape[0]
    PRED = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):
        X_mini = X[n_val]
        X_mini = X_mini[np.newaxis,:,:]
        l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)
        E_list = np.asarray([[0,X_mini.shape[1]-1]])
        PRED[n_val] = sess.run([predictions], feed_dict={x: X_mini, ids:E_list, ids_len:l_mini,is_train:False})

# CONVERT OUTPUTS TO DESIRED INT FORMAT AND SAVE TXT FILE
Y = np.zeros(no_utt)
for n_val in range(no_utt):
    Y[n_val] = sum(sum(np.asarray(PRED[n_val][0]>=0.5, dtype=np.float32)))
print(res_path)
np.savetxt(res_path,Y,fmt='%i',delimiter='\n')
if(wasdir):
    newfile = res_path[0:-4] + '_files.txt'
    np.savetxt(newfile,fileList,fmt='%s',delimiter='\n')
