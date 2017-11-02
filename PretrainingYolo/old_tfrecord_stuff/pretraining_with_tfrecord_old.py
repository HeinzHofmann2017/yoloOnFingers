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
import tensorflow as tf
import numpy as np
import mailer
import os
import analyse_Dataset as aD
#import matplotlib.pyplot as plt


def main():
    print("TensorFlow version ", tf.__version__)
    
    #Environment = "dgx"
    Environment = "Desktop"
    print("Enviroment =", Environment)
    
    if (Environment         == "Desktop"):    
        batchSize           = 1
        learning_rate       = 0.005
        capacity            = 10
        num_threads         = 2
        min_after_dequeue   = 5
        data_path           ="/media/hhofmann/deeplearning/ilsvrc2012/train_tf_records/"#Desktop-path
        valid_data_path     ="/media/hhofmann/deeplearning/ilsvrc2012/valid_tf_records/"#Desktop-path
        weights_path        ="/media/hhofmann/deeplearning/getfingers_heinz/pretrained_weights/pretrained_model.ckpt"#Desktop-path       
        mailtext            ="training on Desktop"
        #nr_of_epochs       = 3  
        
    elif (Environment       == "dgx"):
        batchSize           = 16
        learning_rate       = 0.005 
        capacity            = 2000
        num_threads         = 8
        min_after_dequeue   = 1000
        data_path           ="/mnt/data/ilsvrc2012/train_tf_records/"#dgx-path
        valid_data_path     ="/mnt/data/ilsvrc2012/valid_tf_records/"#dgx-path
        weights_path        ="/mnt/data/getfingers_heinz/pretrained_weights/pretrained_model.ckpt"#dgx-path
        mailtext            ="training on DGX"
        #nr_of_epochs       = 1000  
    

    #data_path="/home/hhofmann/Schreibtisch/Daten/indexfinger_right/3000_readyTOlearn/trainData/trainData.tfrecords"
    with tf.name_scope("training_Data") as scope:
        list_of_tfrecords = os.listdir(data_path)
        data = []
        for l in list_of_tfrecords:
            data.append(l)
        filename_queue = tf.train.string_input_producer(data)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,features={
                                    'picName':      tf.FixedLenFeature([],tf.string),
                                    'height':       tf.FixedLenFeature([],tf.int64),
                                    'width':        tf.FixedLenFeature([],tf.int64),
                                    'channels':     tf.FixedLenFeature([],tf.int64),
                                    'image_raw':    tf.FixedLenFeature([],tf.string),  })
        #get back the "2D"-Image out of the bitstream
        image       =   tf.decode_raw(features["image_raw"],out_type=tf.uint8)
        image_float =   tf.cast(image, tf.float16)        
        height      =   tf.cast(features['height']  ,tf.int32)
        width       =   tf.cast(features['width']   ,tf.int32)
        channels    =   tf.cast(features['channels'],tf.int32)
        image_shape =   tf.stack([height,width,channels])        
        #image_shape =   tf.stack([HEIGHT, WIDTH, CHANNELS])
        reshaped_image       = tf.reshape(image_float, image_shape)
        size=[224,224,3]
        croped_image = tf.random_crop(value=reshaped_image, size=size) 
    
        name = tf.cast(features['picName'], tf.string)

        
        #Make automatic shuffled batches out of the dataset('s)
        
        images, names =tf.train.shuffle_batch([croped_image,name], 
                                              batch_size=batchSize,#Number of Pictures&Labels per Batch
                                              capacity=capacity,#max Number of Elements in the queue 
                                              num_threads=num_threads, #Nr. of Threads, which enqueueing tensor-list.
                                              min_after_dequeue=min_after_dequeue #minimum Number of Pictures in the queue after give the data to the GPU
                                              )
                                              
    with tf.name_scope("validation_Data") as scope:
        list_of_valid_tfrecords = os.listdir(valid_data_path)
        valid_data = []
        for l in list_of_valid_tfrecords:
            valid_data.append(l)
        valid_filename_queue = tf.train.string_input_producer(valid_data)
        valid_reader = tf.TFRecordReader()
        _,serialized_valid_example = valid_reader.read(valid_filename_queue)
        valid_features = tf.parse_single_example(serialized_valid_example,features={
                                    'picName':      tf.FixedLenFeature([],tf.string),
                                    'height':       tf.FixedLenFeature([],tf.int64),
                                    'width':        tf.FixedLenFeature([],tf.int64),
                                    'channels':     tf.FixedLenFeature([],tf.int64),
                                    'image_raw':    tf.FixedLenFeature([],tf.string),  })
        #get back the "2D"-Image out of the bitstream
        valid_image     =   tf.decode_raw(valid_features["image_raw"],out_type=tf.uint8)
        valid_image_float   =   tf.cast(valid_image, tf.float16)
        valid_height        =   tf.cast(valid_features['height'],tf.int32)
        valid_width         =   tf.cast(valid_features['width'],tf.int32)
        valid_channels      =   tf.cast(valid_features['channels'], tf.int32)
        valid_image_shape   =   tf.stack([valid_height,valid_width, valid_channels])
        reshaped_valid_image=   tf.reshape(valid_image_float,valid_image_shape)
        size=[224,224,3]
        croped_valid_image  =   tf.random_crop(value=reshaped_valid_image, size=size)
        valid_name          =   tf.cast(valid_features['picName'],tf.string)
        
        valid_images,valid_names        =   tf.train.shuffle_batch([croped_valid_image,valid_name], 
                                              batch_size=batchSize,#Number of Pictures&Labels per Batch
                                              capacity=capacity,#max Number of Elements in the queue 
                                              num_threads=num_threads, #Nr. of Threads, which enqueueing tensor-list.
                                              min_after_dequeue=min_after_dequeue #minimum Number of Pictures in the queue after give the data to the GPU
                                              )


    input_pictures  = tf.placeholder(dtype=tf.float16,shape=(batchSize, 224, 224, 3))
    input_labels    = tf.placeholder(dtype=tf.float16, shape=(batchSize, 4096)) #ToDo: Replace 4096 with the correct number of Labels
#==============================================================================
#                                                       
# HIer Graph aufbauen:                                                      
#                                                       
#==============================================================================
                            
#==============================================================================
# Layer 1:
#     Conv. Layer 7x7x64-s-2                                                  
#==============================================================================
    with tf.name_scope("Layer1_Conv") as scope:
        W1 = tf.Variable(tf.truncated_normal(shape=[7,7,3,64], stddev=0.01, dtype=tf.float16),name='W1')
        b1 = tf.Variable(tf.truncated_normal(shape=[64],stddev=0.01,dtype=tf.float16),name='b1')
        conv_1_unbiased=tf.nn.conv2d(input=input_pictures,filter=W1,strides=[1,2,2,1],padding='SAME',name="conv_1_unbiased")
        conv_1_linear = tf.add(conv_1_unbiased, b1,name="conv_1_linear")
        conv_1 = tf.maximum(0.1*conv_1_linear,conv_1_linear,name="leaky_relu_1")
#==============================================================================
# Layer 2:
#     Maxpool Layer 2x2 -s-2
#==============================================================================
    with tf.name_scope("Layer2_maxpool") as scope:
        mpool_2 = tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_2")
#==============================================================================
# Layer 3:
#     Conv Layer 3x3x192
#==============================================================================
    with tf.name_scope("Layer3_Conv") as scope:
        W3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,192],stddev=0.01,dtype=tf.float16), name='W3')
        b3 = tf.Variable(tf.truncated_normal(shape=[192],stddev=0.01,dtype=tf.float16),name='b3')
        conv_3_unbiased = tf.nn.conv2d(input=mpool_2,filter=W3,strides=[1,1,1,1], padding='SAME',name="conv_3_unbiased")
        conv_3_linear = tf.add(conv_3_unbiased, b3, name="conv_3_linear")
        conv_3 = tf.maximum(0.1*conv_3_linear, conv_3_linear, name="leaky_relu_3")
#==============================================================================
# Layer 4:
#     Maxpool Layer 2x2 -s-2
#==============================================================================
    with tf.name_scope("Layer4_maxpool") as scope:
        mpool_4 = tf.nn.max_pool(conv_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_4")
#==============================================================================
# Layer 5:
#     Conv Layer 1x1x128
#==============================================================================
    with tf.name_scope("Layer5_Conv") as scope:
        W5 = tf.Variable(tf.truncated_normal(shape=[1,1,192,128],stddev=0.01,dtype=tf.float16),name="W5")
        b5 = tf.Variable(tf.truncated_normal(shape=[128],stddev=0.01,dtype=tf.float16),name="b5")
        conv_5_unbiased = tf.nn.conv2d(input=mpool_4, filter=W5,strides=[1,1,1,1], padding="SAME", name="conv_5_unbiased")
        conv_5_linear = tf.add(conv_5_unbiased, b5, name = "conv_5_linear")
        conv_5 = tf.maximum(0.1*conv_5_linear, conv_5_linear, name="leaky_relu_5")
#==============================================================================
# Layer 6:
#     Conv Layer 3x3x256        
#==============================================================================
    with tf.name_scope("Layer6_Conv") as scope:
        W6 = tf.Variable(tf.truncated_normal(shape=[3,3,128,256],stddev=0.01, dtype=tf.float16),name="W6")
        b6 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b6")
        conv_6_unbiased = tf.nn.conv2d(input=conv_5,filter=W6,strides=[1,1,1,1], padding="SAME", name="conv_6_unbiased")
        conv_6_linear = tf.add(conv_6_unbiased, b6, name="conv_6_linear")
        conv_6 = tf.maximum(0.1*conv_6_linear, conv_6_linear, name="leaky_relu_6")
#==============================================================================
# Layer 7:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer7_Conv") as scope:
        W7 = tf.Variable(tf.truncated_normal(shape=[1,1,256,256],stddev=0.01,dtype=tf.float16),name="W7")
        b7 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b7")
        conv_7_unbiased = tf.nn.conv2d(input=conv_6,filter=W7,strides=[1,1,1,1],padding="SAME", name="conv_7_unbiased")
        conv_7_linear = tf.add(conv_7_unbiased, b7, name="conv_7_linear")
        conv_7 = tf.maximum(0.1*conv_7_linear, conv_7_linear, name="leaky_relu_7")
#==============================================================================
# Layer 8:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer8_Conv") as scope:
        W8 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W8")
        b8 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b8")
        conv_8_unbiased = tf.nn.conv2d(input=conv_7,filter=W8,strides=[1,1,1,1],padding="SAME",name="conv_8_unbiased")
        conv_8_linear = tf.add(conv_8_unbiased, b8, name="conv_8_linear")
        conv_8 = tf.maximum(0.1*conv_8_linear, conv_8_linear, name="leaky_relu_8")
#==============================================================================
# Layer 9:
#     Maxpool Layer 2x2 -s-2
#==============================================================================
    with tf.name_scope("Layer9_maxpool") as scope:
        mpool_9 = tf.nn.max_pool(conv_8,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_9")
#==============================================================================
# Layer 10:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer10_Conv") as scope:    #Todo: ????
        W10 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],stddev=0.01,dtype=tf.float16),name="W10")
        b10 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b10")
        conv_10_unbiased = tf.nn.conv2d(input=mpool_9,filter=W10,strides=[1,1,1,1],padding="SAME", name="conv_10_unbiased")
        conv_10_linear = tf.add(conv_10_unbiased, b10, name="conv_10_linear")
        conv_10 = tf.maximum(0.1*conv_10_linear, conv_10_linear, name="leaky_relu_10")
#==============================================================================
# Layer 11:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer11_Conv") as scope:
        W11 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W11")
        b11 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b11")
        conv_11_unbiased = tf.nn.conv2d(input=conv_10,filter=W11,strides=[1,1,1,1],padding="SAME",name="conv_11_unbiased")
        conv_11_linear = tf.add(conv_11_unbiased,b11,name="conv_11_linear")
        conv_11 = tf.maximum(0.1*conv_11_linear,conv_11_linear, name="leaky_relu_11")
#==============================================================================
# Layer 12:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer12_Conv") as scope:    #Todo: ????
        W12 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],stddev=0.01,dtype=tf.float16),name="W12")
        b12 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b12")
        conv_12_unbiased = tf.nn.conv2d(input=conv_11,filter=W12,strides=[1,1,1,1],padding="SAME", name="conv_12_unbiased")
        conv_12_linear = tf.add(conv_12_unbiased, b12, name="conv_12_linear")
        conv_12 = tf.maximum(0.1*conv_12_linear, conv_12_linear, name="leaky_relu_12")
#==============================================================================
# Layer 13:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer13_Conv") as scope:
        W13 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W13")
        b13 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b13")
        conv_13_unbiased = tf.nn.conv2d(input=conv_12,filter=W13,strides=[1,1,1,1],padding="SAME",name="conv_13_unbiased")
        conv_13_linear = tf.add(conv_13_unbiased,b13,name="conv_13_linear")
        conv_13 = tf.maximum(0.1*conv_13_linear,conv_13_linear, name="leaky_relu_13")
#==============================================================================
# Layer 14:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer10_Conv") as scope:    #Todo: ????
        W14 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],stddev=0.01,dtype=tf.float16),name="W14")
        b14 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b14")
        conv_14_unbiased = tf.nn.conv2d(input=conv_13,filter=W14,strides=[1,1,1,1],padding="SAME", name="conv_14_unbiased")
        conv_14_linear = tf.add(conv_14_unbiased, b14, name="conv_14_linear")
        conv_14 = tf.maximum(0.1*conv_14_linear, conv_14_linear, name="leaky_relu_14")
#==============================================================================
# Layer 15:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer15_Conv") as scope:
        W15 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W15")
        b15 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b15")
        conv_15_unbiased = tf.nn.conv2d(input=conv_14,filter=W15,strides=[1,1,1,1],padding="SAME",name="conv_15_unbiased")
        conv_15_linear = tf.add(conv_15_unbiased,b15,name="conv_15_linear")
        conv_15 = tf.maximum(0.1*conv_15_linear,conv_15_linear, name="leaky_relu_15")
#==============================================================================
# Layer 16:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer16_Conv") as scope:    #Todo: ????
        W16 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],stddev=0.01,dtype=tf.float16),name="W16")
        b16 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b16")
        conv_16_unbiased = tf.nn.conv2d(input=conv_15,filter=W16,strides=[1,1,1,1],padding="SAME", name="conv_16_unbiased")
        conv_16_linear = tf.add(conv_16_unbiased, b16, name="conv_16_linear")
        conv_16 = tf.maximum(0.1*conv_16_linear, conv_16_linear, name="leaky_relu_16")
#==============================================================================
# Layer 17:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer17_Conv") as scope:
        W17 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W17")
        b17 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b17")
        conv_17_unbiased = tf.nn.conv2d(input=conv_16,filter=W17,strides=[1,1,1,1],padding="SAME",name="conv_17_unbiased")
        conv_17_linear = tf.add(conv_17_unbiased,b17,name="conv_17_linear")
        conv_17 = tf.maximum(0.1*conv_17_linear,conv_17_linear, name="leaky_relu_17")   
#==============================================================================
# Layer 18:
#     Conv Layer 1x1x512
#==============================================================================
    with tf.name_scope("Layer18_Conv") as scope:
        W18 = tf.Variable(tf.truncated_normal(shape=[1,1,512,512],stddev=0.01,dtype=tf.float16),name="W18")
        b18 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b18")
        conv_18_unbiased = tf.nn.conv2d(input=conv_17,filter=W18,strides=[1,1,1,1],padding="SAME", name="conv_18_unbiased")
        conv_18_linear = tf.add(conv_18_unbiased, b18, name="conv_18_linear")
        conv_18 = tf.maximum(0.1*conv_18_linear, conv_18_linear, name="leaky_relu_18")
#==============================================================================
# Layer 19:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer19_Conv") as scope:
        W19 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],stddev=0.01,dtype=tf.float16),name="W19")
        b19 = tf.Variable(tf.truncated_normal(shape=[1024],stddev=0.01,dtype=tf.float16),name="b19")
        conv_19_unbiased = tf.nn.conv2d(input=conv_18,filter=W19,strides=[1,1,1,1],padding="SAME",name="conv_19_unbiased")
        conv_19_linear = tf.add(conv_19_unbiased,b19,name="conv_19_linear")
        conv_19 = tf.maximum(0.1*conv_19_linear,conv_19_linear, name="leaky_relu_19")
#==============================================================================
# Layer 20:
#     Maxpool Layer 2x2 -s-2
#==============================================================================
    with tf.name_scope("Layer20_maxpool") as scope:
        mpool_20 = tf.nn.max_pool(conv_19,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_20")
#==============================================================================
# Layer 21:
#     Conv Layer 1x1x512
#==============================================================================
    with tf.name_scope("Layer21_Conv") as scope:    #Todo: ????
        W21 = tf.Variable(tf.truncated_normal(shape=[1,1,1024,512],stddev=0.01,dtype=tf.float16),name="W21")
        b21 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b21")
        conv_21_unbiased = tf.nn.conv2d(input=mpool_20,filter=W21,strides=[1,1,1,1],padding="SAME", name="conv_21_unbiased")
        conv_21_linear = tf.add(conv_21_unbiased, b21, name="conv_21_linear")
        conv_21 = tf.maximum(0.1*conv_21_linear, conv_21_linear, name="leaky_relu_21")
#==============================================================================
# Layer 22:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer22_Conv") as scope:    
        W22 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],stddev=0.01,dtype=tf.float16),name="W22")
        b22 = tf.Variable(tf.truncated_normal(shape=[1024],stddev=0.01,dtype=tf.float16),name="b22")
        conv_22_unbiased = tf.nn.conv2d(input=conv_21,filter=W22,strides=[1,1,1,1],padding="SAME", name="conv_22_unbiased")
        conv_22_linear = tf.add(conv_22_unbiased, b22, name="conv_22_linear")
        conv_22 = tf.maximum(0.1*conv_22_linear, conv_22_linear, name="leaky_relu_22")
#==============================================================================
# Layer 23:
#     Conv Layer 1x1x512
#==============================================================================
    with tf.name_scope("Layer23_Conv") as scope:    #Todo: ????
        W23 = tf.Variable(tf.truncated_normal(shape=[1,1,1024,512],stddev=0.01,dtype=tf.float16),name="W23")
        b23 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b23")
        conv_23_unbiased = tf.nn.conv2d(input=conv_22,filter=W23,strides=[1,1,1,1],padding="SAME", name="conv_23_unbiased")
        conv_23_linear = tf.add(conv_23_unbiased, b23, name="conv_23_linear")
        conv_23 = tf.maximum(0.1*conv_23_linear, conv_23_linear, name="leaky_relu_23")
#==============================================================================
# Layer 24:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer24_Conv") as scope:    
        W24 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],stddev=0.01,dtype=tf.float16),name="W24")
        b24 = tf.Variable(tf.truncated_normal(shape=[1024],stddev=0.01,dtype=tf.float16),name="b24")
        conv_24_unbiased = tf.nn.conv2d(input=conv_23,filter=W24,strides=[1,1,1,1],padding="SAME", name="conv_24_unbiased")
        conv_24_linear = tf.add(conv_24_unbiased, b24, name="conv_24_linear")
        conv_24 = tf.maximum(0.1*conv_24_linear, conv_24_linear, name="leaky_relu_24")









#==============================================================================
# Layer 25:
#     Averagepool 3x3 -s-3
#==============================================================================
    with tf.name_scope("Layer26_AvgPool") as scope:
        avgpool_25 = tf.nn.avg_pool(conv_24,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME", name="avgpool_25")
#==============================================================================
# Layer 26:
#     Fully connected Layer
#==============================================================================
    with tf.name_scope("Layer26_full") as scope:
        input_26 = tf.reshape(tensor=avgpool_25, shape=[batchSize,4*4*1024])
        W26 = tf.Variable(tf.truncated_normal(shape=[4*4*1024,4096],stddev=0.01,dtype=tf.float16),name="W26")#ToDo: Replace 4096 with the correct number of Labels
        b26 = tf.Variable(tf.truncated_normal(shape=[4096],stddev=0.01,dtype=tf.float16),name="b26")#ToDo: Replace 4096 with the correct number of Labels
        fully_26 = tf.nn.relu(tf.matmul(input_26,W26)+b26,name="fully_31")



        
    with tf.name_scope("cost_function") as scope:
        #make data ready for using with 
        #cost_function = output_32[batchElement,hoehe,breite,tiefe]
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=fully_26)
        cost_x =tf.reduce_mean(tf.square(tf.subtract(output_32[:,:,:,0],x_coords)))
        cost_y =tf.reduce_mean(tf.square(tf.subtract(output_32[:,:,:,1],y_coords)))
        cost_p =tf.reduce_mean(tf.square(tf.subtract(output_32[:,:,:,2],probs)))
        cost = tf.add(tf.add(cost_x,cost_y),cost_p)
        
    with tf.name_scope("optimizer") as scope:
        # Gradient descen
        #TODO: Gradient Decent durch ADAM ersetzen
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescentOptimizer')
        
        # grads_and_vars is a list of tuples (gradient, variable). Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        grads_and_vars = optimizer.compute_gradients(cost)
        
        # Ask the optimizer to apply the capped gradients.
        train_step = optimizer.apply_gradients(grads_and_vars)

    
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver(
        max_to_keep=5,
        keep_checkpoint_every_n_hours=4.0, 
        pad_step_number=True,
        save_relative_paths=True,)


    c_tp                    =np.zeros(64)
    for i in range(64):
        c_tp[i] = 1
    x                       =64
    i                       =0
    lowest_cost             =100#has to be higher than 2 else it wouldn't start training
    count_of_improvement    =0
    


    with tf.Session() as sess:
        sess.run(init_op)
        
        writer=tf.summary.FileWriter("summary") 
        writer.add_graph(sess.graph)     
        
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        
        while(x>=6.4):
            
#==============================================================================
#             img, xcor, ycor, prb = sess.run([images,x_coords,y_coords,probs])
#             print(img.shape)
#             print(img[0,:,:,:].shape)
#             print(xcor.shape)
#             print(ycor.shape)
#            
#             _,cx,cy,cp,c = sess.run([train_step,cost_x,cost_y,cost_p,cost])
#             print(i, " Kosten x=",cx," Kosten y=",cy," Kosten p=",cp," Kosten=",c)
#             out = sess.run(padded_29[0,:,4,0])#gibt eine Bildzeile aus, wobei rechts und links zerogepadded wurde
#             out = sess.run(padded_29[0,4,:,0])#gibt eine Bildkolonne aus, wobei oben zerogepadded wurde...
#             out = sess.run(fully_31)
#             out = sess.run(mpool_30[0,:,4,1]) # 100x7x7x1024
#             print(out)
            
#             plt.figure(i*3)
#             plt.imshow(img[0, :, :, :])
#             plt.figure(i*3+1)
#             plt.imshow(img[1, :, :, :])
#             plt.figure(i*3+2)
#             plt.imshow(img[2, :, :, :])
#==============================================================================


            _,c=sess.run([train_step,cost], feed_dict={input_pictures: images, input_labels: names})#Todo: replace "names" with the correct input for the labels...
            c_tp[i%64]=c
            x=0
            for j in range(64):
                x += c_tp[j]
            print("Kosten im Mittel = ",x/64)
            if(i%64==0):
                cx,cy,cp,c = sess.run([cost_x,cost_y,cost_p,cost])
                if((x/64)<lowest_cost):
                    lowest_cost             = x/64
                    count_of_improvement   += 1
                    saver.save(sess=sess, save_path=weights_path, global_step=i)
                    print("model updatet")
                    text = "updatet model with average cost = " + str(lowest_cost) + "\n actual cx="+str(cx)+"\n actual cy="+str(cy)+"\n actual cp="+str(cp)+"\n actual c="+str(c)                    
                    mailer.mailto(text)
                
                
            

            i+=1
        coord.request_stop()
        coord.join(threads)
        #print("Accuracy:", accuracy.eval())
    
    # plt.show()
    print("finished")
    mailtext += " finished \n"
    mailtext += "\n cost                = " + str(x)
    mailtext += "\n batchSize           = " + str(batchSize)
    mailtext += "\n learningrate        = " + str(learning_rate)
    mailtext += "\n num_threads         = " + str(num_threads)
    mailtext += "\n min_after_dequeue   = " + str(min_after_dequeue)
    mailtext += "\n trained on          = " + data_path
    mailtext += "\n number of iterations= " + str(i)

    mailer.mailto(mailtext)
    print("mailed")
    


if __name__ == "__main__":
   
    main()
