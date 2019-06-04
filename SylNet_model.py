__author__ = "Lauri Juvela, lauri.juvela@aalto.fi; Shreyas Seshadri, shreyas.seshadri@aalto.fi"

import os
import sys
import math
import numpy as np
import tensorflow as tf

_FLOATX = tf.float32 

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    if shape is None:
        return tf.get_variable(name)
    else:  
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)): 
    if shape is None:
        return tf.get_variable(name) 
    else:     
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)
   

class CNET():

    def __init__(self,
                 name,
                 endList,
                 seqLen,
                 isTrain=True,
                 DRrate=0.3,
                 residual_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=123,
                 output_channels=1,
                 cond_dim = None,
                 cond_channels = 64,
                 postnet_channels=256,
                 do_postproc = True,
                 do_GLU = True):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
        self.do_postproc = do_postproc
        self.do_GLU = do_GLU
        self.seqLen = seqLen
        self.endList = endList
        self.isTrain = isTrain
        self.DRrate = DRrate

        if cond_dim is not None:
            self._use_cond = True
            self.cond_dim = cond_dim
            self.cond_channels = cond_channels
            
        else:
            self._use_cond = False

        self._name = name


    def get_variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)          
    def get_variable_list_post(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='S/postproc_module') 

    def _causal_conv(self,X, W, filter_width):
        """
        Assume X input shape (batch, timesteps, channels)
        """
        pad = (filter_width - 1) 
        # pad to causal
        Y = tf.pad(X, paddings=[[0,0], [pad,0], [0,0]]) # left pad to time-axis
        Y = tf.nn.convolution(Y, W, padding="VALID")
        return Y

    
    def _input_layer(self, main_input):
        fw = self.filter_width
        r = self.residual_channels
        i = self.input_channels
        
        with tf.variable_scope('input_layer'):            

            W = get_weight_variable('W', (fw, i, 2*r))
            b = get_bias_variable('b', (2*r))

            X = main_input
            Y = tf.nn.convolution(X, W, padding='SAME')
            #Y = self._causal_conv(X, W, fw)
            
            Y += b
            Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])
            Y = tf.layers.dropout(Y,rate=self.DRrate ,training=self.isTrain)

        return Y

    def _embed_cond(self, cond_input):
        cd = self.cond_dim
        c = self.cond_channels
        with tf.variable_scope('embed_cond'):            
            
            W = get_weight_variable('W',(1, cd, c))
            b = get_bias_variable('b',(c))

            Y = tf.nn.convolution(cond_input, W, padding='SAME') # 1x1 convolution
            Y += b

            return tf.tanh(Y)


    def _conv_module(self, main_input, residual_input, module_idx, dilation, cond_input=None):
        fw = self.filter_width
        r = self.residual_channels
        
        s = self.postnet_channels
        
        with tf.variable_scope('conv_modules'):
            with tf.variable_scope('module{}'.format(module_idx)):
                                
                W = get_weight_variable('filter_gate_W',(fw, r, 2*r)) 
                b = get_bias_variable('filter_gate_b',(2*r))                                                          

                X = main_input
                Y = tf.nn.convolution(X, W, padding='SAME', dilation_rate=[dilation])
                #Y = self._causal_conv(X, W, fw)
                
                Y += b
                
                if self._use_cond:
                    c = self.cond_channels
                    V_cond = get_weight_variable('cond_filter_gate_W',(1, c, 2*r)) 
                    b_cond = get_bias_variable('cond_filter_gate_b',(2*r)) 
                    
                    C = tf.nn.convolution(cond_input, V_cond, padding='SAME') # 1x1 convolution
                    #C = self._causal_conv(cond_input, V_cond, 1)
                    C += b_cond
                    C = tf.tanh(C)
                    Y += C

                # filter and gate
                Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])
                Y = tf.layers.dropout(Y,rate=self.DRrate ,training=self.isTrain)
                
                # add residual channel
                if self.do_postproc: 
                    W_s = get_weight_variable('skip_gate_W',(fw, r, s)) 
                    b_s = get_weight_variable('skip_gate_b',s)
                    
                    skip_out = tf.nn.convolution(Y, W_s, padding='SAME')
                    #skip_out = self._causal_conv(Y, W_s, fw)
                    skip_out += b_s
                else:
                    skip_out = []
                
                if self.do_GLU:
                    W_p = get_weight_variable('post_filter_gate_W',(1, r, r)) 
                    b_p = get_weight_variable('post_filter_gate_b',r)
                    
                    Y = tf.nn.convolution(Y, W_p, padding='SAME')
                    #Y = self._causal_conv(Y, W_p, 1)
                    Y += b_p
                    Y += X

        return Y, skip_out

    def _postproc_module(self, residual_module_outputs):
        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels
        op = self.output_channels
        with tf.variable_scope('postproc_module'):

            W1 = get_weight_variable('W1',(fw, r, s))
            b1 = get_bias_variable('b1',s)
            W2 = get_weight_variable('W2', (s, op))
            b2 = get_bias_variable('b2',op)
            num_units = self.postnet_channels

            # sum of residual module outputs
            X = tf.zeros_like(residual_module_outputs[0])
            for R in residual_module_outputs:
                X += R

            Y = tf.nn.convolution(X, W1, padding='SAME')    
            #Y = self._causal_conv(X, W1, fw)
            Y += b1
            
            Y = tf.nn.relu(X)
            Y = tf.layers.dropout(Y,rate=self.DRrate ,training=self.isTrain)
            
            lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE, name='lstm_layer')
            outputsM,_=tf.nn.dynamic_rnn(lstm_layer,Y,sequence_length=self.seqLen,dtype="float32")
            outputs = tf.gather_nd(outputsM,self.endList)
            prediction=tf.matmul(outputs,W2)+b2
            
                        
        return prediction
    
    def _last_layer(self, last_layer_ip):
        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels
        
        with tf.variable_scope('last_layer'):

            W1 = get_weight_variable('W1',(fw, r, s))
            b1 = get_bias_variable('b1',s)
            W2 = get_weight_variable('W2', (s, 1))
            b2 = get_bias_variable('b2',1)
            
            num_units = self.postnet_channels
 
            # sum of residual module outputs
            X = last_layer_ip

            Y = tf.nn.convolution(X, W1, padding='SAME')    
            Y += b1
            Y = tf.nn.relu(Y)

            
            lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE, name='lstm_layer')
            outputsM,_=tf.nn.dynamic_rnn(lstm_layer,Y,sequence_length=self.seqLen,dtype="float32")
            outputs = tf.gather_nd(outputsM,self.endList)
            prediction=tf.matmul(outputs,W2)+b2
            
            predictionALL = 1#tf.matmul(outputsM,W2)+b2
            
        return prediction,predictionALL

    def forward_pass(self, X_input, cond_input=None):
        
        skip_outputs = []
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):

            if self._use_cond:
                C = self._embed_cond(cond_input)
            else:
                C = None    

            R = self._input_layer(X_input)
            X = R
            for i, dilation in enumerate(self.dilations):
                X, skip = self._conv_module(X, R, i, dilation, cond_input=C)
                skip_outputs.append(skip)

            if self.do_postproc:    
                Y= self._postproc_module(skip_outputs)    
            else:
                Y = self._last_layer(X)                

        return Y
                                                 