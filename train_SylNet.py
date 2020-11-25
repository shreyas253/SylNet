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
import os


mainFile = os.path.dirname(os.path.realpath(__file__)) 
modelFile = mainFile + '/trained_models/'
means_path = modelFile + 'means.npy'
std_path = modelFile + 'stds.npy'
model_path = modelFile + 'model_trained.ckpt'

########### GET DATA #############
fileList = list(filter(bool,[line.rstrip('\n') for line in open('config_files.txt')]))

noSyls =  list(map(int, list(filter(bool,[line.rstrip('\n') for line in open('config_sylls.txt')]))))
T_c = np.zeros((len(noSyls),max(noSyls)))
for i in range(len(noSyls)):
    if noSyls[i]>0:
        T_c[i,:noSyls[i]]=1
T = np.asarray(noSyls, dtype=np.int32)

if len(fileList) != len(noSyls):
    raise Exception('Check the config files. Lengths dont match!!!!')        

noUtt_main = len(noSyls)

Fs = 16000
w_l = round(0.025*Fs)
w_h = round(0.01*Fs)
X = np.ndarray((noUtt_main,),dtype=object)
for i in range(noUtt_main):
    fs, y = scipy.io.wavfile.read(fileList[i])
    y = y/max(abs(y))
    y = librosa.core.resample(y=y, orig_sr=fs, target_sr=Fs)    
    X[i] = np.transpose(20*np.log10(librosa.feature.melspectrogram(y=y, sr=Fs, n_mels=24, n_fft=w_l, hop_length=w_h)+0.00000000001))

MEAN = np.mean(np.concatenate(X),axis=0) # Z norm data
STD = np.std(np.concatenate(X),axis=0)  
np.save(means_path, MEAN)  
np.save(std_path, STD) 

for i in range(noUtt_main):
    X[i] = (X[i] - MEAN)/STD


idx = np.random.permutation(noUtt_main)
no_train = round(0.8*noUtt_main)

x_val = X[no_train+1:]
X = X[:no_train]

t_val_c = T_c[no_train+1:,]
T_c = T_c[:no_train,]  

t_val = T[no_train+1:]
T = T[:no_train]  
maxT = max(noSyls)  
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
adam_lr = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
num_epochs = 500

ids = tf.placeholder(shape=(None, 2), dtype=tf.int32)
ids_len = tf.placeholder(shape=(None), dtype=tf.int32)
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
y = tf.placeholder(shape=(None,), dtype=tf.float32)
y_c = tf.placeholder(shape=(None,maxT), dtype=tf.float32)
logits = S.forward_pass(x)

#loss_function
yd = tf.maximum(y,1)
predictions = tf.nn.sigmoid(logits)
individual_loss = tf.reduce_sum(tf.square(predictions-tf.cast(y_c,tf.float32)), axis=1)
loss_obj = tf.reduce_mean(tf.div(individual_loss,yd))
opt=tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(loss_obj)

init=tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)

saver = tf.train.Saver()


loss_val = np.ones((num_epochs,1)) * np.inf
stpCrit_win = 10
stpCrit_min = 30
batchSize = 32
noBatches = math.floor(X.shape[0]/batchSize)

epoch=0
dontStop=1
with tf.Session(config=tfconfig) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 
   
    
    while (epoch<num_epochs) & (dontStop):
                
        # train
        if X.shape[0]>batchSize: # adjust size of minibatch
            idx = np.random.permutation(X.shape[0])
            idx = idx[:batchSize*noBatches]
            idx = np.split(idx,noBatches)
        else:
            idx = np.random.permutation(X.shape[0])
            idx = np.split(idx,1)                    
        saver = tf.train.Saver(max_to_keep=0)
        t = time.time()
        for batch_i in range(len(idx)):            
            X_mini = X[idx[batch_i]]
            l_mini = np.asarray([xx.shape[0] for xx in X_mini],dtype=np.int32)
            X_mini = np.asarray([padarray(xx,max(l_mini)) for xx in X_mini])                    
            E_list = np.asarray([[i,xx-1] for i,xx in enumerate(l_mini)],dtype=np.int32)
            T_mini = T[idx[batch_i]]
            T_mini_c = T_c[idx[batch_i]]
            _, lossD = sess.run([opt,loss_obj], feed_dict={x: X_mini,y: T_mini,y_c: T_mini_c, ids: E_list, ids_len:l_mini,is_train:True})
            elapsed = time.time() - t       
            print("Errors for epoch %d, batch %d: %f, and took time: %f" % (epoch, batch_i,lossD, elapsed)) 
        
        # validate        
        no_utt = x_val.shape[0]
        loss_test = np.ones((no_utt,1))
        for n_val in range(no_utt):
            X_mini = x_val[n_val]
            X_mini = X_mini[np.newaxis,:,:]
            l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)            
            E_list = np.asarray([[0,X_mini.shape[1]-1]])                    
            loss_test[n_val] = sess.run([loss_obj], feed_dict={x: X_mini,y:t_val[[n_val]],y_c:t_val_c[[n_val]],ids:E_list, ids_len:l_mini,is_train:False})  
        loss_val[epoch] = np.mean(loss_test)   
        
        # save model if it reduces loss
        if loss_val[epoch] == np.min(loss_val):
            save_path = saver.save(sess, model_path)
        
        # stop training is condition is satisfied
        if epoch>stpCrit_min:
            tmp = loss_val[epoch:epoch-stpCrit_win-1:-1]
            tmp = tmp[1:]-tmp[0]
            if ((tmp < 0).sum() == tmp.size).astype(np.int):
                dontStop = 0
              
        print("Validation errors for epoch %d: %f , and took time: %f" % (epoch, loss_val[epoch], elapsed))
        epoch += 1    
       
