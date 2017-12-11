# A simple MNIST classifier (linear regression) with tensorflow
#

#
# Usage:
# python2.7 -m pip install tensorflow
# python3 board.py
# python3.5 board.py
#
# tensorboard --logdir=summary
# 
# JSCH 2017-04-23

from __future__ import print_function

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.platform import gfile

this_folder =  os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_folder+ '/../dataPreprocessing/OnTabeasData/Step4CollectTogether/')
import analyse_Fingerset
sys.path.insert(0,this_folder+"/../helperfunctions/")
import mailer
import heinzAPI as hAPI
import parserClassFingers as pC

#import matplotlib.pyplot as plt

#get default-Hyperparameters, btw. Parameters from shell-script:
parser_object = pC.make_parser()

name                            = parser_object.modelname
print(name)
batchSize                       = parser_object.batch_Size
learning_rate                   = parser_object.learning_rate
num_threads                     = parser_object.num_Threads
buffer_size                     = parser_object.buffer_Size
origin_path                     = parser_object.origin_Path
nr_of_epochs                    = parser_object.nr_of_epochs
nr_of_epochs_until_save_model   = parser_object.nr_of_epochs_until_save_model
dropout                         = parser_object.dropout_bool
batchnorm                       = parser_object.batchnorm_bool
test                            = parser_object.test_bool
random_seed                     = parser_object.rand_seed


def dataset_preprocessor(picname,label):
    content = tf.read_file(origin_path + picname)
    image = tf.image.decode_png(content,channels=1)
    image = tf.image.convert_image_dtype(image,tf.float32)
    #ToDo: random Crop here (is a kind of complicated because of the x and y labels.)
    return image,label
    
def main():
    print("TensorFlow version ", tf.__version__)
    tf.set_random_seed(random_seed)
    with tf.name_scope("Data") as scope:
        print("read in all Train Picture-Names & Labels and shuffle them")
        ReadData        = analyse_Fingerset.Dataset_Heinz()
        
        train_data      = ReadData.get_train_data(origin_path = origin_path)
        train_picnames  = [row[0] for row in train_data]
        train_labels    = np.float32([row[1] for row in train_data])#puts a 4 Dimensional Vector to train_labels
        train_data      = Dataset.from_tensor_slices((train_picnames,train_labels))
        train_data      = train_data.repeat()
        train_data      = train_data.shuffle(buffer_size=buffer_size)
        train_data      = train_data.map(map_func=dataset_preprocessor,
                                         num_threads=num_threads,
                                         output_buffer_size=10000)
        train_data      = train_data.batch(batchSize)

        print("read in all Valid Picture-Names & Labels and shuffle them")
        valid_data      = ReadData.get_valid_data(origin_path=origin_path)
        valid_picnames  = [row[0] for row in valid_data]
        valid_labels    = np.float32([row[1] for row in valid_data])
        valid_data      = Dataset.from_tensor_slices((valid_picnames,valid_labels))
        valid_data      = valid_data.repeat()
        valid_data      = valid_data.shuffle(buffer_size=buffer_size)
        valid_data      = valid_data.map(map_func=dataset_preprocessor,
                                         num_threads=num_threads,
                                         output_buffer_size=10000)
        valid_data      = valid_data.batch(batchSize)

        print("read in all Test Picture-Names & Labels and shuffle them")  
        test_data       = ReadData.get_valid_data(origin_path=origin_path)
        test_picnames   = [row[0] for row in test_data]
        test_labels    = np.float32([row[1] for row in test_data])
        test_data      = Dataset.from_tensor_slices((test_picnames,test_labels))
        test_data      = test_data.map(map_func=dataset_preprocessor,
                                         num_threads=num_threads,
                                         output_buffer_size=10000)
        test_data       = test_data.batch(batchSize)

    print("finished read all Pictures and Labels, start Data-iterator")
    with tf.name_scope("Data-Iterator") as scope:        
        iterator        = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        images_unnormalized, labels  = iterator.get_next()
        
        training_init_op    = iterator.make_initializer(train_data)
        validation_init_op  = iterator.make_initializer(valid_data)
        testing_init_op     = iterator.make_initializer(test_data)
        images_unnormalized = tf.image.resize_images(images_unnormalized,[448,448])#TODO:eventually reverse image resizing
        images_unnormalized = tf.cast(images_unnormalized,tf.float32)
        
        #To test, how the croped picters look like, when they are used to learn...
        tf.summary.image('images_after_crop',tensor = images_unnormalized , max_outputs=20)
            
    #is true,if the model is training right now, and is False, if the model is testing.
    training = tf.placeholder(tf.bool, name='training')
    learnrate = tf.placeholder(tf.float32, name='learnrate')
        
    with tf.name_scope("normalize_pictures") as scope:                            
        images = hAPI.normalize_pictures(tensor=images_unnormalized)
#==============================================================================
#                                                       
# HIer Graph aufbauen:                                                      
#                                                       
#==============================================================================
    #Conv. Layer 7x7x64-s-2                                                  
    output_1 = hAPI.convLayerPretrained(tensor=images,layerNr=1,batchSize=batchSize, filterwidth=7, inputdepth=1, outputdepth=64, strides=2, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("2_Maxpool_Layer") as scope:
        output_2 = tf.nn.max_pool(output_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_2")
    #Conv Layer 3x3x192
    output_3 = hAPI.convLayerPretrained(tensor=output_2,layerNr=3,batchSize=batchSize, filterwidth=3, inputdepth=64, outputdepth=192, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("4_Maxpool_Layer") as scope:
        output_4 = tf.nn.max_pool(output_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_4")
    #Conv Layer 1x1x128
    output_5 = hAPI.convLayerPretrained(tensor=output_4,layerNr=5,batchSize=batchSize, filterwidth=1, inputdepth=192, outputdepth=128, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x256 
    output_6 = hAPI.convLayerPretrained(tensor=output_5,layerNr=6,batchSize=batchSize, filterwidth=3, inputdepth=128, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_7 = hAPI.convLayerPretrained(tensor=output_6,layerNr=7,batchSize=batchSize, filterwidth=1, inputdepth=256, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_8 = hAPI.convLayerPretrained(tensor=output_7,layerNr=8,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("9_Maxpool_Layer") as scope:
        output_9 = tf.nn.max_pool(output_8,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_9")
    #Conv Layer 1x1x256
    output_10 = hAPI.convLayerPretrained(tensor=output_9,layerNr=10,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_11 = hAPI.convLayerPretrained(tensor=output_10,layerNr=11,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_12 = hAPI.convLayerPretrained(tensor=output_11,layerNr=12,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_13 = hAPI.convLayerPretrained(tensor=output_12,layerNr=13,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_14 = hAPI.convLayerPretrained(tensor=output_13,layerNr=14,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_15 = hAPI.convLayerPretrained(tensor=output_14,layerNr=15,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_16 = hAPI.convLayerPretrained(tensor=output_15,layerNr=16,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_17 = hAPI.convLayerPretrained(tensor=output_16,layerNr=17,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 1x1x512
    output_18 = hAPI.convLayerPretrained(tensor=output_17,layerNr=18,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_19 = hAPI.convLayerPretrained(tensor=output_18,layerNr=19,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("20_Maxpool_Layer") as scope:
        output_20 = tf.nn.max_pool(output_19,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_20")
    #Conv Layer 1x1x512
    output_21 = hAPI.convLayerPretrained(tensor=output_20,layerNr=21,batchSize=batchSize, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_22 = hAPI.convLayerPretrained(tensor=output_21,layerNr=22,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 1x1x512
    output_23 = hAPI.convLayerPretrained(tensor=output_22,layerNr=23,batchSize=batchSize, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_24 = hAPI.convLayerPretrained(tensor=output_23,layerNr=24,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=False,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_25 = hAPI.convLayer(tensor=output_24,layerNr=25,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x1024-s-2
    output_26 = hAPI.convLayer(tensor=output_25,layerNr=26,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=2, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x1024
    output_27 = hAPI.convLayer(tensor=output_26,layerNr=27,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x1024
    output_28 = hAPI.convLayer(tensor=output_27,layerNr=28,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
#TODO:eventually reverse image resizing
#==============================================================================
#     #Zero Padding        
#     with tf.name_scope("Layer29_ZeroPadding") as scope:
#         output_29=tf.pad(output_28, np.array([[0,0],[3,3],[1,0],[0,0]]))
#     #Maxpool 3x3 -s-3
#     with tf.name_scope("Layer30_maxpool") as scope:
#         output_30 = tf.nn.max_pool(output_29,ksize=[1,3,3,1],strides=[1,3,3,1], padding="SAME")
#==============================================================================
    #Fully-Connected Layer ==> make vector
    with tf.name_scope("Layer31_full") as scope:
        input_31 = tf.reshape(tensor=output_28,shape=[batchSize,7*7*1024])
        if dropout == True:
            with tf.name_scope("dropout"):        
                #dropout only over all the feature-maps and batches.
                input_31 = tf.cond(training,
                                      lambda:tf.nn.dropout(x=input_31, keep_prob=0.8,noise_shape=[batchSize,7*7*1024]),
                                      lambda:input_31)
        W31 = tf.Variable(tf.truncated_normal(shape=[7*7*1024,4096],stddev=0.01,dtype=tf.float32),name="W31")
        b31 = tf.Variable(tf.truncated_normal(shape=[4096],stddev=0.01,dtype=tf.float32),name="b31")
        preactivate_31 = tf.add(tf.matmul(input_31,W31),b31)
        with tf.name_scope("batch_norm"):
            input_depth_31 = preactivate_31.get_shape().as_list()[-1]#takes the last element which is in this case 64
            #make new weights and new bias
            with tf.name_scope("beta"):
                beta31 = tf.Variable(tf.constant(0.0,shape=[input_depth_31],dtype=tf.float32), name="beta",trainable=True)
            with tf.name_scope("gamma"):
                gamma31 = tf.Variable(tf.constant(1.0,shape=[input_depth_31],dtype=tf.float32),name="gamma",trainable=True)
            batch_mean31, batch_variance31 = tf.nn.moments(x=preactivate_31,axes=[0,1])
            
            preactivate_31 = tf.nn.batch_normalization(x=preactivate_31,mean=batch_mean31,variance=batch_variance31,offset=beta31,scale=gamma31,variance_epsilon=1e-4,name=None) 
        output_31 = tf.nn.relu(preactivate_31)
        with tf.name_scope("summary"):
            hAPI.variable_summaries(variable=W31,name="W31")
            hAPI.variable_summaries(variable=b31,name="b31")
            hAPI.variable_summaries(variable=preactivate_31, name="preactivate31")
            hAPI.variable_summaries(variable=output_31, name="output31")
    #Fully-Connected Layer ==> make tensor again.
    with tf.name_scope("Layer32_full") as scope:
        if dropout == True:
            with tf.name_scope("dropout"):        
                #dropout only over all the feature-maps and batches.
                output_31 = tf.cond(training,
                                      lambda:tf.nn.dropout(x=output_31, keep_prob=0.8,noise_shape=[batchSize,4096]),
                                      lambda:output_31)
        W32 = tf.Variable(tf.truncated_normal(shape=[4096,7*7*6],stddev=0.01,dtype=tf.float32),name="W32")
        b32 = tf.Variable(tf.truncated_normal(shape=[7*7*6],stddev=0.01,dtype=tf.float32),name="b32")
        preactivate_32 = tf.add(tf.matmul(output_31,W32),b32)
        with tf.name_scope("batch_norm"):
            input_depth_32 = preactivate_32.get_shape().as_list()[-1]#takes the last element which is in this case 64
            #make new weights and new bias
            with tf.name_scope("beta"):
                beta32 = tf.Variable(tf.constant(0.0,shape=[input_depth_32],dtype=tf.float32), name="beta",trainable=True)
            with tf.name_scope("gamma"):
                gamma32 = tf.Variable(tf.constant(1.0,shape=[input_depth_32],dtype=tf.float32),name="gamma",trainable=True)
            batch_mean32, batch_variance32 = tf.nn.moments(x=preactivate_32,axes=[0,1])
            preactivate_32 = tf.nn.batch_normalization(x=preactivate_32,mean=batch_mean32,variance=batch_variance32,offset=beta32,scale=gamma32,variance_epsilon=1e-4,name=None)   
        fully_32 = preactivate_32#tf.nn.relu(preactivate_32)
        output_32 = tf.sigmoid(tf.reshape(tensor=fully_32, shape=[batchSize,7,7,6]))
        output_32 = tf.cast(output_32,tf.float32)
        with tf.name_scope("summary"):                        
            hAPI.variable_summaries(variable=W32,name="W32")
            hAPI.variable_summaries(variable=b32,name="b32")
            hAPI.variable_summaries(variable=preactivate_32, name="preactivate32")
            hAPI.variable_summaries(variable=output_32, name="output32")
            
        
    with tf.name_scope("cost_function") as scope:
        x1_output = tf.squeeze(output_32[:,:,:,0])
        x_label   = tf.squeeze(labels[:,:,:,0])

        y1_output = tf.squeeze(output_32[:,:,:,1])
        y_label   = tf.squeeze(labels[:,:,:,1])

        h1_output = tf.squeeze(output_32[:,:,:,2])
        h_label  = tf.squeeze(labels[:,:,:,2])
        
        w1_output = tf.squeeze(output_32[:,:,:,3])
        w_label   = tf.squeeze(labels[:,:,:,3])

        c1_output = tf.squeeze(output_32[:,:,:,4])
        
        p_output  = tf.squeeze(output_32[:,:,:,5])
        p_label   = tf.squeeze(labels[:,:,:,4])
        
        with tf.name_scope("x-costs"):
            x1_difference = tf.subtract(x1_output,x_label)           
            x1_squared = tf.square(x1_difference)
            x1_costs = tf.multiply(p_label,x1_squared)
            x1_costs = tf.reduce_sum(x1_costs)
            x1_costs = tf.multiply(x1_costs,5)#like in the yolo-paper...
        with tf.name_scope("y-costs"):
            y1_difference = tf.subtract(y1_output,y_label)           
            y1_squared = tf.square(y1_difference)
            y1_costs = tf.multiply(p_label,y1_squared)
            y1_costs = tf.reduce_sum(y1_costs)
            y1_costs = tf.multiply(y1_costs,5)#like in the yolo-paper...
        with tf.name_scope("h-costs"):
            h1_output_root = tf.sqrt(h1_output)
            h_label_root = tf.sqrt(h_label)
            h1_difference = tf.subtract(h1_output_root,h_label_root)
            h1_squared = tf.square(h1_difference)
            h1_costs = tf.multiply(p_label,h1_squared)
            h1_costs = tf.reduce_sum(h1_costs)
            h1_costs = tf.multiply(h1_costs,5)#like in the yolo-paper...
        with tf.name_scope("w-costs"):
            w1_output_root = tf.sqrt(w1_output)
            w_label_root = tf.sqrt(w_label)
            w1_difference = tf.subtract(w1_output_root,w_label_root)
            w1_squared = tf.square(w1_difference)
            w1_costs = tf.multiply(p_label,w1_squared)
            w1_costs = tf.reduce_sum(w1_costs)
            w1_costs = tf.multiply(w1_costs,5)
        with tf.name_scope("C-costs"):
            w_label_half = tf.div(w_label,2)
            h_label_half = tf.div(h_label,2)

            w1_output_half= tf.div(w1_output,2)
            h1_output_half= tf.div(h1_output,2)
            
            label_left = tf.subtract(x_label,w_label_half)
            label_right= tf.add(x_label,w_label_half)
            label_top = tf.subtract(y_label,h_label_half)
            label_bottom=tf.add(y_label,h_label_half)
            
            output1_left = tf.subtract(x1_output,w1_output_half)
            output1_right= tf.add(x1_output,w1_output_half)
            output1_top = tf.subtract(y1_output,h1_output_half)
            output1_bottom= tf.add(y1_output,h1_output_half)

            overlap1_left = tf.maximum(label_left,output1_left)
            overlap1_right= tf.minimum(label_right,output1_right)
            overlap1_top = tf.maximum(label_top,output1_top)
            overlap1_bottom= tf.minimum(label_bottom,output1_bottom)
            
            overlap1_width = tf.subtract(overlap1_right,overlap1_left)
            overlap1_width = tf.maximum(np.float32(0),overlap1_width)
            overlap1_height = tf.subtract(overlap1_bottom,overlap1_top)
            overlap1_height = tf.maximum(np.float32(0),overlap1_height)
            area_of_overlap1 = tf.multiply(overlap1_width, overlap1_height)
            
            output1_area = tf.multiply(h1_output,w1_output)
            label_area = tf.multiply(h_label,w_label)
            
            area_of_union1 = tf.subtract(tf.add(output1_area,label_area),area_of_overlap1)
            
            IoU1 = tf.div(area_of_overlap1,area_of_union1)
            IoU1_scalar = tf.reduce_mean(IoU1)            
            
            c1_label = tf.multiply(IoU1,p_label)#with more than one finger use here p_label=max(p_label1,p_label2,...)
                                                
            c1_difference = tf.subtract(c1_output,c1_label)
            c1_squared = tf.square(c1_difference)
            
            with tf.name_scope("obj_present"):
                c1_obj = tf.multiply(c1_squared, p_label)
                c1_obj_costs = tf.reduce_sum(c1_obj)
            with tf.name_scope("noobj_present"):
                p_label_invers = tf.subtract(np.float32(1),p_label)# y=1-x   x=0|y=1  & x=1|y=0
                c1_noobj = tf.multiply(c1_squared, p_label_invers)
                c1_noobj_costs = tf.reduce_sum(c1_noobj)
                c1_noobj_costs = tf.multiply(c1_noobj_costs,0.5)#like in the yolo paper...
        with tf.name_scope("p_costs"):
            p_difference = tf.subtract(p_output,p_label)
            p_squared = tf.square(p_difference)
            p_costs = tf.multiply(p_label,p_squared)
            p_costs = tf.reduce_sum(p_costs)
        
        center_costs = tf.add(x1_costs,y1_costs)        
        box_costs = tf.add(h1_costs,w1_costs)
        conf_costs = tf.add(c1_obj_costs,c1_noobj_costs)
        spatial_costs = tf.add(center_costs,box_costs)
        prob_costs = tf.add(conf_costs,p_costs)
        costs = tf.add(spatial_costs,prob_costs)
        
        tf.summary.scalar("Costx",x1_costs)
        tf.summary.scalar("Costy",y1_costs)
        tf.summary.scalar("Costh",h1_costs)
        tf.summary.scalar("Costw",w1_costs)
        tf.summary.scalar("IoU_mean",IoU1_scalar)
        tf.summary.scalar("CostcObj",c1_obj_costs)
        tf.summary.scalar("CostcNoObj",c1_noobj_costs)
        tf.summary.scalar("Costp",p_costs)
        
        tf.summary.scalar("CostCenter",center_costs)
        tf.summary.scalar("CostBox",box_costs)
        tf.summary.scalar("CostConf",conf_costs)
        
        tf.summary.scalar("CostSpatial",spatial_costs)
        tf.summary.scalar("CostProb",prob_costs)
        
        tf.summary.scalar("Costs",costs)

        
    with tf.name_scope("optimizer") as scope:
        # Gradient descen
        #TODO: Gradient Decent durch ADAM ersetzen
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learnrate,epsilon=1e-04)#From 64lRate0_1_ to 114_01lRate098Ptest everything learned with the Adam default-learnrate of 0.001
        
        # grads_and_vars is a list of tuples (gradient, variable). Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        grads_and_vars = optimizer.compute_gradients(costs)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
        
        # Ask the optimizer to apply the capped gradients.
        train_step = optimizer.apply_gradients(capped_gvs)
#==============================================================================
#     for grad, var in grads_and_vars:
#         if grad is not None:
#             tf.summary.histogram(var.op.name +"gradients", grad)
#==============================================================================
    for grad, var in capped_gvs:
        if grad is not None:
            tf.summary.histogram(var.op.name +"capped_gradients",grad)
    with tf.name_scope("Validation_Numbers") as scope:
            true_probabilities = tf.div(tf.reduce_sum(tf.multiply(p_output,p_label)),tf.reduce_sum(p_label))
            max_probabilities = tf.reduce_mean(tf.reduce_max(p_output,[1,2]))
            mean_probabilities = tf.reduce_mean(p_output)
            true_confidence = tf.div(tf.reduce_sum(tf.multiply(c1_output,p_label)),tf.reduce_sum(p_label))
            max_confidence = tf.reduce_mean(tf.reduce_max(c1_output,[1,2]))
            mean_confidence = tf.reduce_mean(c1_output)
            tf.summary.scalar("true_probabilities",true_probabilities)
            tf.summary.scalar("max_probabilities",max_probabilities)
            tf.summary.scalar("mean_probabilities",mean_probabilities)
            tf.summary.scalar("true_confidence",true_confidence)
            tf.summary.scalar("max_confidence",max_confidence)
            tf.summary.scalar("mean_confidence",mean_confidence)
        
    with tf.name_scope("Test") as scope:
        total_nr_of_Gridcells = tf.reduce_sum(tf.maximum(tf.add(p_label,2),1))
        with tf.name_scope("true_positives"):     
            true_probs = tf.multiply(p_output,p_label)
            normed_probs_tp = tf.subtract(true_probs,0.98)
            deleted_probs_tp = tf.maximum(normed_probs_tp,np.float32(0))
            residual_probs_tp = tf.multiply(deleted_probs_tp,1000.0)
            true_positives = tf.minimum(residual_probs_tp,np.float32(1))
            true_positives = tf.reduce_sum(true_positives)
            true_positives_normed = tf.div(true_positives,total_nr_of_Gridcells)
            tf.summary.scalar("true_positives",true_positives_normed)
        with tf.name_scope("false_positives"):   
            false_probs = tf.multiply(p_output,p_label_invers)
            normed_probs_fp = tf.subtract(false_probs,0.98)
            deleted_probs_fp = tf.maximum(normed_probs_fp,np.float32(0))
            residual_probs_fp = tf.multiply(deleted_probs_fp,1000.0)
            false_positives = tf.minimum(residual_probs_fp,np.float32(1))
            false_positives = tf.reduce_sum(false_positives)            
            false_positives_normed = tf.div(false_positives,total_nr_of_Gridcells)
            tf.summary.scalar("false_positives",false_positives_normed)
        with tf.name_scope("true_negatives"):
            p_label_invers_1000 = tf.add(tf.multiply(p_label,999),1)
            false_probs = tf.multiply(p_label_invers_1000,p_output)            
            normed_probs_tn = tf.subtract(false_probs,0.98)
            deleted_probs_tn = tf.minimum(normed_probs_tn,np.float32(0))
            residual_probs_tn = tf.multiply(deleted_probs_tn,-1000.0)
            true_negatives = tf.minimum(residual_probs_tn,np.float32(1))
            true_negatives = tf.reduce_sum(true_negatives)
            true_negatives_normed = tf.div(true_negatives,total_nr_of_Gridcells)
            tf.summary.scalar("true_negatives",true_negatives_normed)
        with tf.name_scope("false_negatives"):
            p_label_1000 = tf.add(tf.multiply(p_label_invers,999),1)
            true_probs = tf.multiply(p_label_1000,p_output)            
            normed_probs_fn = tf.subtract(true_probs,0.98)
            deleted_probs_fn = tf.minimum(normed_probs_fn,np.float32(0))
            residual_probs_fn = tf.multiply(deleted_probs_fn,-1000.0)
            false_negatives = tf.minimum(residual_probs_fn,np.float32(1))
            false_negatives = tf.reduce_sum(false_negatives)
            false_negatives_normed = tf.div(false_negatives,total_nr_of_Gridcells)
            tf.summary.scalar("false_negatives",false_negatives_normed)
        with tf.name_scope("IOU"):
            y_label_index = tf.argmax(tf.reduce_max(p_label,axis=[2]),axis=1)
            x_label_index  = tf.argmax(tf.reduce_max(p_label,axis=[1]),axis=1)
            y_label_global = tf.add((y_index/7) , (y_label/7))
            x_label_global = tf.add((x_index/7) , (x_label/7))
            
            #Do something here
            
         
        
        

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver(
        max_to_keep=5,
        keep_checkpoint_every_n_hours=4.0, 
        pad_step_number=True,
        save_relative_paths=True,)
    if not os.path.exists(origin_path + "../../../../weights/"+name+"/"):
        os.makedirs(origin_path + "../../../../weights/"+name+"/")

    
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(testing_init_op)
        sess.run(validation_init_op)
        sess.run(training_init_op)
        sess.run(init_op)
#==============================================================================
#         for i in range(5):
#             label, hoehenvector, hoehenindex, breitenvector, breitenindex = sess.run([p_label,height_vector,height_index,width_vector,width_index])
#   
#             print("p_label")
#             print(label)
#             print("height-vector")
#             print(hoehenvector)
#             print("height-index")
#             print(hoehenindex)
#             print("width_vector")
#             print(breitenvector)
#             print("width-index")
#             print(breitenindex)
#         while True :
#             x=2
#==============================================================================
                
        if(test==False):
            train_writer=tf.summary.FileWriter(origin_path + "../../../../summarys/training/summary_" + name + "_train")
            train_writer.add_graph(sess.graph) 
            valid_writer=tf.summary.FileWriter(origin_path + "../../../../summarys/training/summary_" + name + "_valid")
            valid_writer.add_graph(sess.graph) 
            


            #saver.restore(sess=sess, save_path=origin_path + "../../../../weights/7BnormBeforeRelu2.ckpt-00103000")
            print("start training....\n")
            for i in range(nr_of_epochs/nr_of_epochs_until_save_model):
                if nr_of_epochs < 80000:
                    for j in range(nr_of_epochs_until_save_model):
                        _ = sess.run([train_step],feed_dict={training: True, learnrate : (learning_rate)})
                elif nr_of_epochs < 120000:
                    for j in range(nr_of_epochs_until_save_model):
                        _ = sess.run([train_step],feed_dict={training: True, learnrate : (learning_rate/10)})  
                else:
                    for j in range(nr_of_epochs_until_save_model):
                        _ = sess.run([train_step],feed_dict={training: True, learnrate : (learning_rate/100)})                     
    
                numbers_of_iterations_until_now = i*nr_of_epochs_until_save_model+j+1            
                #testing on traindata
                train_writer.add_summary(sess.run(merged_summary_op,feed_dict={training: False}),(numbers_of_iterations_until_now))
                tp,fp,tn,fn,pt,mp,meanp,tc,mc,meanc = sess.run([true_positives,         
                                                                false_positives, 
                                                                true_negatives, 
                                                                false_negatives,
                                                                true_probabilities,
                                                                max_probabilities,
                                                                mean_probabilities,
                                                                true_confidence,
                                                                max_confidence,
                                                                mean_confidence],feed_dict={training: False})
                print("\n\n\n\nTraining "+name+"\ntrue-positives ="+str(tp)+"\nfalse-positives ="+str(fp)+"\ntrue-negatives ="+str(tn)+"\nfalse-negatives ="+str(fn)+" . \nDone in "+ str(numbers_of_iterations_until_now)+ " Steps")
                print("mean of the true training Probability = " + str(pt))
                print("mean of the max training Probabilitys = " + str(mp))
                print("mean of all training Probabilitys = " + str(meanp))
                print("mean of the true training Confidences = " + str(tc))
                print("mean of the max training Confidences = " + str(mc))
                print("mean of all training Confidences = " + str(meanc))
                #testing on validationdata:
                sess.run(validation_init_op)
                valid_writer.add_summary(sess.run(merged_summary_op,feed_dict={training: False}),(numbers_of_iterations_until_now))
                tp,fp,tn,fn,pt,mp,meanp,tc,mc,meanc = sess.run([true_positives,         
                                                                false_positives, 
                                                                true_negatives, 
                                                                false_negatives,
                                                                true_probabilities,
                                                                max_probabilities,
                                                                mean_probabilities,
                                                                true_confidence,
                                                                max_confidence,
                                                                mean_confidence],feed_dict={training: False})
                print("\nValidation "+name+"\ntrue-positives ="+str(tp)+"\nfalse-positives ="+str(fp)+"\ntrue-negatives ="+str(tn)+"\nfalse-negatives ="+str(fn)+" . \nDone in "+ str(numbers_of_iterations_until_now)+ " Steps")
                print("mean of the true training Probability = " + str(pt))
                print("mean of the max training Probabilitys = " + str(mp))
                print("mean of all training Probabilitys = " + str(meanp))
                print("mean of the true training Confidences = " + str(tc))
                print("mean of the max training Confidences = " + str(mc))
                print("mean of all training Confidences = " + str(meanc))              
                sess.run(training_init_op)
                
                #save Model
                saver.save(sess=sess, save_path=origin_path + "../../../../weights/"+name+"/"+name+".ckpt", global_step=(numbers_of_iterations_until_now))
                print("model updatet\n")

                #training:


        else:
            sess.run(testing_init_op)
            print("Try to restore")
            saver.restore(sess,origin_path + "../../../../weights/105withDropout.ckpt-00040000")                
            print("Restored")
            test_writer=tf.summary.FileWriter(origin_path + "../../../../summarys/training/summary_" + name + "_test")
            test_writer.add_graph(sess.graph)   
            

            for i in range(len(test_picnames)/batchSize):
                testimages,output = sess.run([images_unnormalized,tf.squeeze(output_32)],        feed_dict={training: False})
                #print(output) #output[batchelements, [x_coords, y_coords, probs]]
                #test_writer.add_summary(sess.run(merged_summary_op,feed_dict={training: False}),(0))
                #print("made summary")
                for b in range(batchSize):
                    print(str(i*batchSize+b))
                    testimage = testimages[b]*200
                    conf_max = 0
                    prob_max = 0
                    probconf_max = 0
                    conf_x_offset = 0
                    conf_x_fine = 0
                    conf_y_offset = 0
                    conf_y_fine = 0
                    conf_h = 0
                    conf_w = 0
                    prob_x_offset = 0
                    prob_x_fine = 0
                    prob_y_offset = 0
                    prob_y_fine = 0
                    prob_h = 0
                    prob_w = 0
                    probconf_x_offset = 0
                    probconf_x_fine = 0
                    probconf_y_offset = 0
                    probconf_y_fine = 0
                    probconf_h = 0
                    probconf_w = 0
                    for h in range(7):
                        for w in range(7):
                            conf_pred = output[b,h,w,4]
                            prob_pred = output[b,h,w,5]
                            if(conf_pred>conf_max):
                                conf_max = conf_pred
                                conf_x_offset   = w
                                conf_x_fine     = output[b,h,w,0]
                                conf_y_offset   = h
                                conf_y_fine     = output[b,h,w,1]
                                conf_h          = output[b,h,w,2]
                                conf_w          = output[b,h,w,3]                       
                            if(prob_pred>prob_max):
                                prob_max = prob_pred
                                prob_x_offset   = w
                                prob_x_fine     = output[b,h,w,0]
                                prob_y_offset   = h
                                prob_y_fine     = output[b,h,w,1]
                                prob_h          = output[b,h,w,2]
                                prob_w          = output[b,h,w,3]
                            if((conf_pred*prob_pred)>probconf_max):
                                probconf_max = conf_pred*prob_pred
                                probconf_x_offset   = w
                                probconf_x_fine     = output[b,h,w,0]
                                probconf_y_offset   = h
                                probconf_y_fine     = output[b,h,w,1]
                                probconf_h          = output[b,h,w,2]
                                probconf_w          = output[b,h,w,3]
                                
                    conf_x_offset = float(conf_x_offset)/7
                    conf_y_offset = float(conf_y_offset)/7
                    conf_x_fine = conf_x_fine/7
                    conf_y_fine = conf_y_fine/7
                    conf_x = int((conf_x_offset + conf_x_fine)*1280)
                    conf_y = int((conf_y_offset + conf_y_fine)*960)
                    conf_h = int(conf_h * 960)
                    conf_w = int(conf_w *1280)
                    
                    prob_x_offset = float(prob_x_offset)/7
                    prob_y_offset = float(prob_y_offset)/7
                    prob_x_fine = float(prob_x_fine)/7
                    prob_y_fine = float(prob_y_fine)/7
                    prob_x = int((prob_x_offset + prob_x_fine)*1280)
                    prob_y = int((prob_y_offset + prob_y_fine)*960)
                    prob_h = int(prob_h * 960)
                    prob_w = int(prob_w * 1280)
                    
                    probconf_x_offset = float(probconf_x_offset)/7
                    probconf_y_offset = float(probconf_y_offset)/7
                    probconf_x_fine = probconf_x_fine/7
                    probconf_y_fine = probconf_y_fine/7
                    probconf_x = int((probconf_x_offset + probconf_x_fine)*1280)
                    probconf_y = int((probconf_y_offset + probconf_y_fine)*960)
                    probconf_h = int(probconf_h * 960)
                    probconf_w = int(probconf_w *1280)
                            
                    #vertical lines for box with the highest confidence
#==============================================================================
#                     for x in range((conf_x-conf_w),(conf_x+conf_w)):
#                         for y in range((conf_y-conf_h),(conf_y+conf_h)):
#                             if(x<0):
#                                 x=0
#                             if(x>1279):
#                                 x=1279
#                             if(y<0):
#                                 y=0
#                             if(y>959):
#                                 y=959
#                             if(x%2 == 0):                    
#                                 testimage[y,x,0]=255
#                             else:
#                                 testimage[y,x,0]=0
#==============================================================================
                            
                    #horizontal lines for box with the highest probability
#==============================================================================
#                     for y in range((prob_y-prob_h),(prob_y+prob_h)):
#                         for x in range((prob_x-prob_w),(prob_x+prob_w)):                        
#                             if(x<0):
#                                 x=0
#                             if(x>1279):
#                                 x=1279
#                             if(y<0):
#                                 y=0
#                             if(y>959):
#                                 y=959
#                             if(x%2 == 0):                    
#                                 testimage[y,x,0]=255
#                             else:
#                                 testimage[y,x,0]=0                    
#==============================================================================
                    #Lines for box with the highest probability*confidence
                    for y in range((probconf_y-probconf_h),(probconf_y+probconf_h)):
                        for x in range((probconf_x-probconf_w),(probconf_x+probconf_w)):                        
                            if(x<0):
                                x=0
                            if(x>1279):
                                x=1279
                            if(y<0):
                                y=0
                            if(y>959):
                                y=959
                            if(x%2 == 0):                    
                                testimage[y,x,0]=255
                            else:
                                testimage[y,x,0]=0     
                    #save picture in own folder for recognized fingers 
                    sess.run(tf.write_file(origin_path+"picsRecognized/pic" + str(batchSize*i+b)+"conf%.3f"%conf_max +"prob%.3f"%prob_max+".png",tf.image.encode_png(testimage)))
                       

            
            
            #testimage[y,x,0]
            #print(testimage)
    # plt.show()
    print("finished")

    


if __name__ == "__main__":
   
    main()
