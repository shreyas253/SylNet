import numpy as np
import tensorflow as tf
import time
import hdf5storage

tf.reset_default_graph() # debugging, clear all tf variables
#tf.enable_eager_execution() # placeholders are not compatible

import model_SylNet 
import scipy.io
import math


def padarray(A, size):
    t = size - A.shape[0]
    if t==0:
        r = A        
    else:
        r =np.pad(A,[(0,t),(0,0)],'constant')           
    return r


## LOAD DATA
mainFile = '/l/seshads1/code/git/SylNet/'
datFile = '/l/seshads1/code/git/SylNet/adap_data/'
saveFile = '/l/seshads1/code/git/SylNet/saved_model/'
resFile = '/l/seshads1/code/git/SylNet/res_files/'

iMain = 0
iFold = 0

batchSizeMain = [32]
adam_lrMain = [1e-4]
stpCrit_minMain = [30]
stpCrit_winMain = [10]



doBefTest = 1

tf.reset_default_graph()
loadFile = datFile + 'pyDat_adap.mat'
loaddata = hdf5storage.loadmat(loadFile)
loadFile2 = datFile + 'pyDat_adap_CO.mat'
loaddata2 = hdf5storage.loadmat(loadFile2)
print('loading done')
X = loaddata['X_adap']
T = loaddata['T_adap']
T_c = loaddata2['T_adap_ord']
x_test = loaddata['X_test']
t_test = loaddata['T_test']
t_test_c = loaddata2['T_test_ord']
x_val = loaddata['X_val']
t_val = loaddata['T_val']
t_val_c = loaddata2['T_val_ord']
maxT = loaddata2['maxT']        

print('assignment to vars done')

X = np.concatenate(X)
x_test = np.concatenate(x_test)

## PARAMETERS
residual_channels = 128 
filter_width = 5
dilations = [1]*10
input_channels = X[0].shape[1]
output_channels = maxT
postnet_channels= 128
bn_rate = 0.5

ids = tf.placeholder(shape=(None, 2), dtype=tf.int32)
ids_len = tf.placeholder(shape=(None), dtype=tf.int32)
ids_len = tf.placeholder(shape=(None), dtype=tf.int32)
is_train = tf.placeholder(dtype=tf.bool)
S = model_SylNet.CNET(name='S', 
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
                   BNrate=bn_rate)


# optimizer parameters
adam_lr = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

num_epochs = 500#3#200

# data placeholders of shape (batch_size, timesteps, feature_dim)
x = tf.placeholder(shape=(None, None, input_channels), dtype=tf.float32)
y = tf.placeholder(shape=(None,1), dtype=tf.float32)
y_c = tf.placeholder(shape=(None,maxT), dtype=tf.float32)
logits = S.forward_pass(x)

#loss_function
#loss_obj = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction)
yd = tf.maximum(y,1)

predictions = tf.nn.sigmoid(logits)
individual_loss = tf.reduce_sum(tf.square(predictions-tf.cast(y_c,tf.float32)), axis=1)
loss_obj = tf.reduce_mean(tf.div(individual_loss,yd))
    
#    tf.maximum(predictions,1)
#    top =  tf.cast(tf.argmax(predictions,axis=-1), tf.float32)
#    loss=tf.reduce_mean(tf.div(tf.abs(y-top),yd))*100 
#optimization'
var_list_post =  S.get_variable_list_post()

opt=tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(loss_obj,var_list=[var_list_post])

#initialize variables
init=tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)
    
saveFile1 = resFile + 'res_befadapBN.mat'
saveFile2 = resFile + 'res_afteradapBN.mat'
model_path = saveFile + 'model.ckpt'

model_path_tr = saveFile + 'model_adap.ckpt'

saver = tf.train.Saver()


loss_val = np.ones((num_epochs,1)) * np.inf
stpCrit_win = stpCrit_winMain[iMain]
stpCrit_min = stpCrit_minMain[iMain]
batchSize = batchSizeMain[iMain]
noBatches = math.floor(X.shape[0]/batchSize)

epoch=0
dontStop=1
with tf.Session(config=tfconfig) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 
    saver.restore(sess, model_path)
    
    #### TEST MODEL ON NEW DATA
    #x
    if doBefTest:
        no_utt = x_test.shape[0]
        loss_test = np.ones((no_utt,1))
        PRED_VALS = np.ndarray((no_utt,),dtype=object)
        for n_val in range(no_utt):
            X_mini = x_test[n_val]
            X_mini = X_mini[np.newaxis,:,:]
            l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)            
            E_list = np.asarray([[0,X_mini.shape[1]-1]])
            loss_test[n_val],PRED_VALS[n_val] = sess.run([loss_obj,predictions], feed_dict={x: X_mini,y:t_test[[n_val]],y_c:t_test_c[[n_val]],ids:E_list, ids_len:l_mini,is_train:False})  
        print("RUN:- %d TEST ERRORS WITHOUT ADAPTATION ARE : %f" % ( iMain,np.mean(loss_test)))
        scipy.io.savemat(saveFile1,{"PRED_VALS":PRED_VALS})
        doBefTest = 0
    
    no_utt = x_val.shape[0]
    loss_test = np.ones((no_utt,1))
    for n_val in range(no_utt):
        #print(n_val)
        X_mini = x_val[n_val][0]
        #print(X_mini[0].shape)
        X_mini = X_mini[np.newaxis,:,:]
        l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)            
        E_list = np.asarray([[0,X_mini.shape[1]-1]])                    
        loss_test[n_val] = sess.run([loss_obj], feed_dict={x: X_mini,y:t_val[[n_val]],y_c:t_val_c[[n_val]],ids:E_list, ids_len:l_mini,is_train:False})  
    loss_val[0] = np.mean(loss_test)
    save_path = saver.save(sess, model_path_tr)
    print("RUN:- %d, Validation errors for epoch %d: %f " % (iMain,epoch, loss_val[0]))
    while (epoch<num_epochs) & (dontStop):
        
        # Train discriminator
        if X.shape[0]>batchSize:
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
            #X_mini = np.asarray([xx for xx in X_mini])
            l_mini = np.asarray([xx.shape[0] for xx in X_mini],dtype=np.int32)
            X_mini = np.asarray([padarray(xx,max(l_mini)) for xx in X_mini])                    
            E_list = np.asarray([[i,xx-1] for i,xx in enumerate(l_mini)],dtype=np.int32)
            T_mini = T[idx[batch_i]]
            T_mini_c = T_c[idx[batch_i]]
            _, lossD = sess.run([opt,loss_obj], feed_dict={x: X_mini,y: T_mini,y_c: T_mini_c, ids: E_list, ids_len:l_mini,is_train:True})
            elapsed = time.time() - t       
            print("RUN:- %d, Errors for epoch %d, batch %d: %f, and took time: %f" % (iMain,epoch, batch_i,lossD, elapsed)) 
        
        # validation        
        no_utt = x_val.shape[0]
        loss_test = np.ones((no_utt,1))
        for n_val in range(no_utt):
            #print(n_val)
            X_mini = x_val[n_val][0]
            #print(X_mini[0].shape)
            X_mini = X_mini[np.newaxis,:,:]
            l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)            
            E_list = np.asarray([[0,X_mini.shape[1]-1]])                    
            loss_test[n_val] = sess.run([loss_obj], feed_dict={x: X_mini,y:t_val[[n_val]],y_c:t_val_c[[n_val]],ids:E_list, ids_len:l_mini,is_train:False})  
        loss_val[epoch+1] = np.mean(loss_test)   
        
        if loss_val[epoch+1] == np.min(loss_val):
            save_path = saver.save(sess, model_path_tr)
        
        if epoch>stpCrit_min:
            tmp = loss_val[epoch:epoch-stpCrit_win-1:-1]
            tmp = tmp[1:]-tmp[0]
            if ((tmp < 0).sum() == tmp.size).astype(np.int):
                dontStop = 0
              
        print("RUN:- %d, Validation errors for epoch %d: %f , and took time: %f" % (iMain,epoch, loss_val[epoch+1], elapsed))
        epoch += 1    
    
    #x
    
    saver.restore(sess, model_path_tr)
    
    no_utt = x_test.shape[0]
    loss_test = np.ones((no_utt,1))
    PRED_VALS = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):
        X_mini = x_test[n_val]
        X_mini = X_mini[np.newaxis,:,:]
        l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)            
        E_list = np.asarray([[0,X_mini.shape[1]-1]])
        loss_test[n_val],PRED_VALS[n_val] = sess.run([loss_obj,predictions], feed_dict={x: X_mini,y:t_test[[n_val]],y_c:t_test_c[[n_val]],ids:E_list, ids_len:l_mini,is_train:False})  
    print("RUN:- %d, TEST ERRORS AFTER ADAPTATION ARE : %f" % ( iMain,np.mean(loss_test)))       
    scipy.io.savemat(saveFile2,{"PRED_VALS":PRED_VALS})                
#    iFold += 1
#    print('starting fold '+str(iFold+1))