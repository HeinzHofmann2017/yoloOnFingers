# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:27:04 2017

@author: hhofmann
"""

from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.platform import gfile

this_folder =  os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_folder+ '/../dataPreprocessing/OnILSVRCdata/')
import analyse_Dataset
import mailer

def batchnorm(input_tensor):
    with tf.name_scope("batch_norm"):
        input_depth = input_tensor.get_shape().as_list()[-1]#takes the last element which is in this case 64
    #make new mean and new Variance
        with tf.name_scope("beta"):
            beta = tf.Variable(tf.constant(0.0,shape=[input_depth],dtype=tf.float16), name="beta",trainable=True)
        with tf.name_scope("gamma"):
            gamma = tf.Variable(tf.constant(1.0,shape=[input_depth],dtype=tf.float16),name="gamma",trainable=True)
        batch_mean, batch_variance = tf.nn.moments(x=input_tensor,axes=[0,1,2])
        return tf.nn.batch_normalization(x=input_tensor,
                                             mean=batch_mean,
                                             variance=batch_variance,
                                             offset=beta,
                                             scale=gamma,
                                             variance_epsilon=1e-4,
                                             name=None)

def convLayer(tensor,layerNr,batchSize, filterwidth, inputdepth, outputdepth, strides, batchnorm_=True, dropout_=True,training=False):
    '''
    tensor:     "Tensor" Input-Tensor 4 dimensional [batches,width,height,depth] (Width and height could also be changed with each other)
    layerNr:    "Scalar" The number of the Layer in the whole context
    filterwidth:"Scalar" the width and the height of the convolutional filtermask
    inputdepth: "Scalar" The depth of the input
    outputdepth:"Scalar" The depth of the output (also the depth of the convolutional Filtermask)
    strides:    "Scalar" The number of Elements overhopped by learning in height and with
    
    returns:    "Tensor" 4 Dimensional Tensor with shapes []
    '''
    with tf.name_scope(str(layerNr)+"_conv_Layer") as scope:
        with tf.name_scope("W"):
            #calculate stdev for weights, to pretend vanishing Gradients
                        
            #weightdev = (2 / (filterwidth*(inputdepth+outputdepth))) + 1e-4#get shure, that stdev don't will be zero
            weightdev = 0.01
            W = tf.Variable(tf.truncated_normal(shape=[filterwidth,filterwidth,inputdepth,outputdepth], stddev=weightdev, dtype=tf.float16))
        with tf.name_scope("b"):
            b = tf.Variable(tf.truncated_normal(shape=[outputdepth],stddev=0.01,dtype=tf.float16))
        preactivate = tf.nn.conv2d(input=tensor,filter=W,strides=[1,strides,strides,1],padding='SAME')
        preactivate = tf.add(preactivate, b)
        if batchnorm_ == True:
            preactivate = batchnorm(input_tensor=preactivate)
        if dropout_ == True:
            #dropout only over all the feature-maps and batches.
            preactivate = tf.cond(training,
                                  lambda:tf.nn.dropout(x=preactivate, keep_prob=0.5,noise_shape=[batchSize,1,1,outputdepth]),
                                  lambda:preactivate)
        with tf.name_scope("relu"):
            tensor = tf.maximum(0*preactivate,preactivate)
#==============================================================================
#         with tf.name_scope("leaky_relu"):
#             tensor = tf.maximum(0.1*preactivate,preactivate)
#==============================================================================

        with tf.name_scope("summary"):
            variable_summaries(variable=W,name="W")
            variable_summaries(variable=b,name="b")
            variable_summaries(variable=preactivate, name="preactivate")
        return tensor
        
def variable_summaries(variable, name=" "):
    '''
    variable:   "Tensor" Tensor, which shall be displayed in the tensorboard
    name:       "String" is the name, with which the Graphics in the Board shall be named
    Reference:
    Jonas Schmid hand_detector.py
    '''
    mean = tf.reduce_mean(variable)
    tf.summary.scalar('mean_'+name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
    tf.summary.scalar('stddev_'+name, stddev)
    tf.summary.scalar('max_'+name, tf.reduce_max(variable))
    tf.summary.scalar('min_'+name, tf.reduce_min(variable))
    tf.summary.histogram('histogram_'+name, variable)

def normalize_pictures(tensor):
    mean,var = tf.nn.moments(x=tensor, axes=[1,2],keep_dims=True)
    tensor = tf.subtract(tensor , mean)
    var = var + 1e-4
    tensor = tf.div(tensor,var)
    return tensor
    