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

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.platform import gfile

this_folder =  os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_folder+ '/../dataPreprocessing/OnILSVRCdata/')
import analyse_Dataset
sys.path.insert(0,this_folder+"/../helperfunctions/")
import mailer

#import matplotlib.pyplot as plt

Environment = "dgx"
#Environment = "Desktop"
print("Enviroment =", Environment)

if (Environment         == "Desktop"):    
    batchSize           = 2
    learning_rate       = 0.01
    capacity            = 10
    num_threads         = 2
    min_after_dequeue   = 5
    buffer_size         = 20
    origin_path         ="/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"#Desktop-path  
    image_path          ="/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"#Desktop-path 
    mailtext            ="training on Desktop"
    nr_of_epochs       = 3
    nr_of_epochs_until_save_model = 1
    
elif (Environment       == "dgx"):
    batchSize           = 128
    learning_rate       = 1 
    capacity            = 2000
    num_threads         = 8
    min_after_dequeue   = 1000
    buffer_size         = 100000
    origin_path         ="/mnt/data/ilsvrc2012/LabelList_Heinz/"#dgx-path
    image_path          ="/mnt/fast/ilsvrc2012/ILSVRC2012_img_train_t12/"
    mailtext            ="training on DGX"
    nr_of_epochs        = 10000000 
    nr_of_epochs_until_save_model = 1000#all 10minutes

def dataset_preprocessor(picname,labels):
    content = tf.read_file(origin_path + "../ILSVRC2012_img_train_t12_grayscale/" + picname)
    #content = tf.read_file(image_path+"../ILSVRC2012_img_train_t12_grayscale/"+picname)
    image = tf.image.decode_jpeg(content,channels=1)
    image = tf.image.convert_image_dtype(image,tf.float16)
    image = tf.random_crop(image,[224,224,1])
    
    #When everything with preprocessing outside of dgx works fine, the following commented code can be deletet!!
    #image = tf.image.rgb_to_grayscale(image)
#==============================================================================
#     image = tf.cond(tf.logical_and(tf.greater_equal(tf.shape(image)[0],224),
#                                    tf.greater_equal(tf.shape(image)[1],224)),
#                     lambda: tf.random_crop(image,[224,224,1]),
#                     lambda: tf.image.resize_image_with_crop_or_pad(image,224,224))    
#==============================================================================
    
    #if(tf.shape(image)[1]>224 and tf.shape(image)[2]>224):
    #image = tf.random_crop(image,[224,224,1])
    #else:
    #    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    return image,labels
def main():
    print("TensorFlow version ", tf.__version__)
    

    
    print("read in all Picture-Names & Labels and shuffle them")
    ReadData        = analyse_Dataset.Dataset_Heinz()
    train_picnames  = ReadData.get_train_picnames(origin_path=origin_path)
    train_labels    = ReadData.get_train_labels(origin_path=origin_path)#labels between 0 & 999
    train_data      = Dataset.from_tensor_slices((train_picnames,train_labels))
    train_data      = train_data.repeat()
    train_data      = train_data.shuffle(buffer_size=buffer_size)
    train_data      = train_data.map(map_func=dataset_preprocessor,
                                     num_threads=16,
                                     output_buffer_size=10000)
    train_data      = train_data.batch(batchSize)
    
    valid_picnames  = ReadData.get_valid_picnames(origin_path=origin_path)
    valid_labels    = ReadData.get_valid_labels(origin_path=origin_path)
    valid_data      = Dataset.from_tensor_slices((valid_picnames,valid_labels))
    valid_data      = valid_data.repeat()
    valid_data      = valid_data.shuffle(buffer_size=buffer_size)
    valid_data      = valid_data.map(map_func=dataset_preprocessor,
                                     num_threads=16,
                                     output_buffer_size=10000)
    valid_data      = valid_data.batch(batchSize)
    
    iterator        = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    images, labels  = iterator.get_next()
    
    training_init_op    = iterator.make_initializer(train_data)
    validation_init_op  = iterator.make_initializer(valid_data)

    labels = tf.one_hot(indices     =   labels,
                        depth       =   1000,
                        on_value    =   1.0,
                        off_value   =   0.0,
                        axis        =   -1,
                        dtype       =   tf.float32)    
    
    #input_pictures  = tf.placeholder(dtype=tf.float16,shape=(batchSize, 224, 224, 3))
    #input_labels    = tf.placeholder(dtype=tf.float16, shape=(batchSize, 1000))
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
        W1 = tf.Variable(tf.truncated_normal(shape=[7,7,1,64], stddev=0.01, dtype=tf.float16),name='W1')
        b1 = tf.Variable(tf.truncated_normal(shape=[64],stddev=0.01,dtype=tf.float16),name='b1')
        W1_h = tf.summary.histogram("weights1",W1)
        b1_h = tf.summary.histogram("biases1",b1)
        conv_1_unbiased=tf.nn.conv2d(input=images,filter=W1,strides=[1,2,2,1],padding='SAME',name="conv_1_unbiased")
        conv_1_linear = tf.add(conv_1_unbiased, b1,name="conv_1_linear")
        prerelu1_h = tf.summary.histogram("prerelu1",conv_1_linear)
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
        W3_h = tf.summary.histogram("weights3",W3)
        b3_h = tf.summary.histogram("biases3",b3)
        conv_3_unbiased = tf.nn.conv2d(input=mpool_2,filter=W3,strides=[1,1,1,1], padding='SAME',name="conv_3_unbiased")
        conv_3_linear = tf.add(conv_3_unbiased, b3, name="conv_3_linear")
        prerelu3_h = tf.summary.histogram("prerelu3",conv_3_linear)
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
        W5_h = tf.summary.histogram("weights5",W5)
        b5_h = tf.summary.histogram("biases5",b5)
        conv_5_unbiased = tf.nn.conv2d(input=mpool_4, filter=W5,strides=[1,1,1,1], padding="SAME", name="conv_5_unbiased")
        conv_5_linear = tf.add(conv_5_unbiased, b5, name = "conv_5_linear")
        prerelu5_h = tf.summary.histogram("prerelu5",conv_5_linear)
        conv_5 = tf.maximum(0.1*conv_5_linear, conv_5_linear, name="leaky_relu_5")
#==============================================================================
# Layer 6:
#     Conv Layer 3x3x256        
#==============================================================================
    with tf.name_scope("Layer6_Conv") as scope:
        W6 = tf.Variable(tf.truncated_normal(shape=[3,3,128,256],stddev=0.01, dtype=tf.float16),name="W6")
        b6 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b6")
        W6_h = tf.summary.histogram("weights6",W6)
        b6_h = tf.summary.histogram("biases6",b6)
        conv_6_unbiased = tf.nn.conv2d(input=conv_5,filter=W6,strides=[1,1,1,1], padding="SAME", name="conv_6_unbiased")
        conv_6_linear = tf.add(conv_6_unbiased, b6)
        prerelu6_h = tf.summary.histogram("prerelu6",conv_6_linear)
        conv_6 = tf.maximum(0.1*conv_6_linear, conv_6_linear, name="leaky_relu_6")
#==============================================================================
# Layer 7:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer7_Conv") as scope:
        W7 = tf.Variable(tf.truncated_normal(shape=[1,1,256,256],stddev=0.01,dtype=tf.float16),name="W7")
        b7 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b7")
        W7_h = tf.summary.histogram("weights7",W7)
        b7_h = tf.summary.histogram("biases7",b7)
        conv_7_unbiased = tf.nn.conv2d(input=conv_6,filter=W7,strides=[1,1,1,1],padding="SAME", name="conv_7_unbiased")
        conv_7_linear = tf.add(conv_7_unbiased, b7)
        prerelu7_h = tf.summary.histogram("prerelu7",conv_7_linear)
        conv_7 = tf.maximum(0.1*conv_7_linear, conv_7_linear, name="leaky_relu_7")
#==============================================================================
# Layer 8:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer8_Conv") as scope:
        W8 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W8")
        b8 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b8")
        W8_h = tf.summary.histogram("weights8",W8)
        b8_h = tf.summary.histogram("biases8",b8)
        conv_8_unbiased = tf.nn.conv2d(input=conv_7,filter=W8,strides=[1,1,1,1],padding="SAME",name="conv_8_unbiased")
        conv_8_linear = tf.add(conv_8_unbiased, b8, name="conv_8_linear")
        prerelu8_h = tf.summary.histogram("prerelu8",conv_8_linear)
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
        W10_h = tf.summary.histogram("weights10",W10)
        b10_h = tf.summary.histogram("biases10",b10)
        conv_10_unbiased = tf.nn.conv2d(input=mpool_9,filter=W10,strides=[1,1,1,1],padding="SAME", name="conv_10_unbiased")
        conv_10_linear = tf.add(conv_10_unbiased, b10, name="conv_10_linear")
        prerelu10_h = tf.summary.histogram("prerelu10",conv_10_linear)
        conv_10 = tf.maximum(0.1*conv_10_linear, conv_10_linear, name="leaky_relu_10")
#==============================================================================
# Layer 11:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer11_Conv") as scope:
        W11 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W11")
        b11 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b11")
        W11_h = tf.summary.histogram("weights11",W11)
        b11_h = tf.summary.histogram("biases11",b11)
        conv_11_unbiased = tf.nn.conv2d(input=conv_10,filter=W11,strides=[1,1,1,1],padding="SAME",name="conv_11_unbiased")
        conv_11_linear = tf.add(conv_11_unbiased,b11,name="conv_11_linear")
        prerelu11_h = tf.summary.histogram("prerelu11",conv_11_linear)
        conv_11 = tf.maximum(0.1*conv_11_linear,conv_11_linear, name="leaky_relu_11")
#==============================================================================
# Layer 12:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer12_Conv") as scope:    #Todo: ????
        W12 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],stddev=0.01,dtype=tf.float16),name="W12")
        b12 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b12")
        W12_h = tf.summary.histogram("weights12",W12)
        b12_h = tf.summary.histogram("biases12",b12)
        conv_12_unbiased = tf.nn.conv2d(input=conv_11,filter=W12,strides=[1,1,1,1],padding="SAME", name="conv_12_unbiased")
        conv_12_linear = tf.add(conv_12_unbiased, b12, name="conv_12_linear")
        prerelu12_h = tf.summary.histogram("prerelu12",conv_12_linear)
        conv_12 = tf.maximum(0.1*conv_12_linear, conv_12_linear, name="leaky_relu_12")
#==============================================================================
# Layer 13:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer13_Conv") as scope:
        W13 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W13")
        b13 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b13")
        W13_h = tf.summary.histogram("weights13",W13)
        b13_h = tf.summary.histogram("biases13",b13)
        conv_13_unbiased = tf.nn.conv2d(input=conv_12,filter=W13,strides=[1,1,1,1],padding="SAME",name="conv_13_unbiased")
        conv_13_linear = tf.add(conv_13_unbiased,b13,name="conv_13_linear")
        prerelu13_h = tf.summary.histogram("prerelu13",conv_13_linear)
        conv_13 = tf.maximum(0.1*conv_13_linear,conv_13_linear, name="leaky_relu_13")
#==============================================================================
# Layer 14:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer10_Conv") as scope:    #Todo: ????
        W14 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],stddev=0.01,dtype=tf.float16),name="W14")
        b14 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b14")
        W14_h = tf.summary.histogram("weights14",W14)
        b14_h = tf.summary.histogram("biases14",b14)
        conv_14_unbiased = tf.nn.conv2d(input=conv_13,filter=W14,strides=[1,1,1,1],padding="SAME", name="conv_14_unbiased")
        conv_14_linear = tf.add(conv_14_unbiased, b14, name="conv_14_linear")
        prerelu14_h = tf.summary.histogram("prerelu14",conv_14_linear)
        conv_14 = tf.maximum(0.1*conv_14_linear, conv_14_linear, name="leaky_relu_14")
#==============================================================================
# Layer 15:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer15_Conv") as scope:
        W15 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W15")
        b15 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b15")
        W15_h = tf.summary.histogram("weights15",W15)
        b15_h = tf.summary.histogram("biases15",b15)
        conv_15_unbiased = tf.nn.conv2d(input=conv_14,filter=W15,strides=[1,1,1,1],padding="SAME",name="conv_15_unbiased")
        conv_15_linear = tf.add(conv_15_unbiased,b15,name="conv_15_linear")
        prerelu15_h = tf.summary.histogram("prerelu15",conv_15_linear)
        conv_15 = tf.maximum(0.1*conv_15_linear,conv_15_linear, name="leaky_relu_15")
#==============================================================================
# Layer 16:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer16_Conv") as scope:    #Todo: ????
        W16 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],stddev=0.01,dtype=tf.float16),name="W16")
        b16 = tf.Variable(tf.truncated_normal(shape=[256],stddev=0.01,dtype=tf.float16),name="b16")
        W16_h = tf.summary.histogram("weights16",W16)
        b16_h = tf.summary.histogram("biases16",b16)
        conv_16_unbiased = tf.nn.conv2d(input=conv_15,filter=W16,strides=[1,1,1,1],padding="SAME", name="conv_16_unbiased")
        conv_16_linear = tf.add(conv_16_unbiased, b16, name="conv_16_linear")
        prerelu16_h = tf.summary.histogram("prerelu16",conv_16_linear)
        conv_16 = tf.maximum(0.1*conv_16_linear, conv_16_linear, name="leaky_relu_16")
#==============================================================================
# Layer 17:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer17_Conv") as scope:
        W17 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],stddev=0.01,dtype=tf.float16),name="W17")
        b17 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b17")
        W17_h = tf.summary.histogram("weights17",W17)
        b17_h = tf.summary.histogram("biases17",b17)
        conv_17_unbiased = tf.nn.conv2d(input=conv_16,filter=W17,strides=[1,1,1,1],padding="SAME",name="conv_17_unbiased")
        conv_17_linear = tf.add(conv_17_unbiased,b17,name="conv_17_linear")
        prerelu17_h = tf.summary.histogram("prerelu17",conv_17_linear)
        conv_17 = tf.maximum(0.1*conv_17_linear,conv_17_linear, name="leaky_relu_17")   
#==============================================================================
# Layer 18:
#     Conv Layer 1x1x512
#==============================================================================
    with tf.name_scope("Layer18_Conv") as scope:
        W18 = tf.Variable(tf.truncated_normal(shape=[1,1,512,512],stddev=0.01,dtype=tf.float16),name="W18")
        b18 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b18")
        W18_h = tf.summary.histogram("weights18",W18)
        b18_h = tf.summary.histogram("biases18",b18)
        conv_18_unbiased = tf.nn.conv2d(input=conv_17,filter=W18,strides=[1,1,1,1],padding="SAME", name="conv_18_unbiased")
        conv_18_linear = tf.add(conv_18_unbiased, b18, name="conv_18_linear")
        prerelu18_h = tf.summary.histogram("prerelu18",conv_18_linear)
        conv_18 = tf.maximum(0.1*conv_18_linear, conv_18_linear, name="leaky_relu_18")
#==============================================================================
# Layer 19:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer19_Conv") as scope:
        W19 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],stddev=0.01,dtype=tf.float16),name="W19")
        b19 = tf.Variable(tf.truncated_normal(shape=[1024],stddev=0.01,dtype=tf.float16),name="b19")
        W19_h = tf.summary.histogram("weights19",W19)
        b19_h = tf.summary.histogram("biases19",b19)
        conv_19_unbiased = tf.nn.conv2d(input=conv_18,filter=W19,strides=[1,1,1,1],padding="SAME",name="conv_19_unbiased")
        conv_19_linear = tf.add(conv_19_unbiased,b19,name="conv_19_linear")
        prerelu19_h = tf.summary.histogram("prerelu19",conv_19_linear)
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
        W21_h = tf.summary.histogram("weights21",W21)
        b21_h = tf.summary.histogram("biases21",b21)
        conv_21_unbiased = tf.nn.conv2d(input=mpool_20,filter=W21,strides=[1,1,1,1],padding="SAME", name="conv_21_unbiased")
        conv_21_linear = tf.add(conv_21_unbiased, b21, name="conv_21_linear")
        prerelu21_h = tf.summary.histogram("prerelu21",conv_21_linear)
        conv_21 = tf.maximum(0.1*conv_21_linear, conv_21_linear, name="leaky_relu_21")
#==============================================================================
# Layer 22:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer22_Conv") as scope:    
        W22 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],stddev=0.01,dtype=tf.float16),name="W22")
        b22 = tf.Variable(tf.truncated_normal(shape=[1024],stddev=0.01,dtype=tf.float16),name="b22")
        W22_h = tf.summary.histogram("weights22",W22)
        b22_h = tf.summary.histogram("biases22",b22)
        conv_22_unbiased = tf.nn.conv2d(input=conv_21,filter=W22,strides=[1,1,1,1],padding="SAME", name="conv_22_unbiased")
        conv_22_linear = tf.add(conv_22_unbiased, b22, name="conv_22_linear")
        prerelu22_h = tf.summary.histogram("prerelu22",conv_22_linear)
        conv_22 = tf.maximum(0.1*conv_22_linear, conv_22_linear, name="leaky_relu_22")
#==============================================================================
# Layer 23:
#     Conv Layer 1x1x512
#==============================================================================
    with tf.name_scope("Layer23_Conv") as scope:    #Todo: ????
        W23 = tf.Variable(tf.truncated_normal(shape=[1,1,1024,512],stddev=0.01,dtype=tf.float16),name="W23")
        b23 = tf.Variable(tf.truncated_normal(shape=[512],stddev=0.01,dtype=tf.float16),name="b23")
        W23_h = tf.summary.histogram("weights23",W23)
        b23_h = tf.summary.histogram("biases23",b23)
        conv_23_unbiased = tf.nn.conv2d(input=conv_22,filter=W23,strides=[1,1,1,1],padding="SAME", name="conv_23_unbiased")
        conv_23_linear = tf.add(conv_23_unbiased, b23, name="conv_23_linear")
        prerelu23_h = tf.summary.histogram("prerelu23",conv_23_linear)
        conv_23 = tf.maximum(0.1*conv_23_linear, conv_23_linear, name="leaky_relu_23")
#==============================================================================
# Layer 24:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer24_Conv") as scope:    
        W24 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],stddev=0.01,dtype=tf.float16),name="W24")
        b24 = tf.Variable(tf.truncated_normal(shape=[1024],stddev=0.01,dtype=tf.float16),name="b24")
        W24_h = tf.summary.histogram("weights24",W24)
        b24_h = tf.summary.histogram("biases24",b24)
        conv_24_unbiased = tf.nn.conv2d(input=conv_23,filter=W24,strides=[1,1,1,1],padding="SAME", name="conv_24_unbiased")
        conv_24_linear = tf.add(conv_24_unbiased, b24, name="conv_24_linear")
        prerelu24_h = tf.summary.histogram("prerelu24",conv_24_linear)
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
        W26 = tf.Variable(tf.truncated_normal(shape=[4*4*1024,1000],stddev=0.01,dtype=tf.float16),name="W26")
        b26 = tf.Variable(tf.truncated_normal(shape=[1000],stddev=0.01,dtype=tf.float16),name="b26")
        W26_h = tf.summary.histogram("weights26",W26)
        b26_h = tf.summary.histogram("biases26",b26)
        prerelu26 = tf.matmul(input_26,W26)+b26
        prerelu26_h = tf.summary.histogram("prerelu26",prerelu26)
        fully_26 = prerelu26#tf.nn.relu(prerelu26)



        
    with tf.name_scope("cost_function") as scope:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fully_26))
        cost_h = tf.summary.scalar("Costs",cost)

        
    with tf.name_scope("optimizer") as scope:
        # Gradient descen
        #TODO: Gradient Decent durch ADAM ersetzen
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescentOptimizer')
        
        # grads_and_vars is a list of tuples (gradient, variable). Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        grads_and_vars = optimizer.compute_gradients(cost)

        
        # Ask the optimizer to apply the capped gradients.
        train_step = optimizer.apply_gradients(grads_and_vars)
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name +"gradients", 1000*grad)
        
    with tf.name_scope("onehot_from_prediction") as scope:
        test_vectors = tf.one_hot(tf.nn.top_k(fully_26).indices,tf.shape(fully_26)[1])

    
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver(
        max_to_keep=5,
        keep_checkpoint_every_n_hours=4.0, 
        pad_step_number=True,
        save_relative_paths=True,)

    
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        
        sess.run(validation_init_op)
        sess.run(training_init_op)
        sess.run(init_op)
        
        writer=tf.summary.FileWriter(origin_path + "../../getfingers_heinz/summarys/summary")
        writer.add_graph(sess.graph) 
        
        training_matches = 0
        validation_matches = 0
        print("start training....\n")
        for i in range(nr_of_epochs):
            
            #training:
            _ = sess.run([train_step])
            writer.add_summary(sess.run(merged_summary_op),i)
            
            #testing (on traindata and on validationdata)
            if(i%nr_of_epochs_until_save_model ==0):
                c,predicted_tensor,label_tensor = sess.run([cost,test_vectors,labels])
                #print("full predicted tensor: " + str(predicted_tensor))
                print("one shot training-cost: " +str(c))
                nr_of_matches = 0                 
                for k in range(batchSize):
                    for number_of_Elements in range(1000):                                      
                        if label_tensor[k][number_of_Elements] == 1 and predicted_tensor[k][0][number_of_Elements] == 1:
                            nr_of_matches += 1
                match_probability = 100 * nr_of_matches / batchSize
                print("How well does the training-Prediction match on the Labels: " + str(match_probability)+" %")
                if(match_probability > training_matches):
                    training_matches+=3
                    mailer.mailto("Number of Matches in the training Set reached (with higher learnrat(0.05)) "+str(match_probability)+" %. Done in "+ str(i)+ " Steps")
                
                #validation:
                sess.run(validation_init_op)
                c,predicted_tensor,label_tensor,predicted_fully = sess.run([cost,test_vectors,labels,fully_26])
                #print("full predicted tensor: " + str(predicted_tensor))
                print("validation cost: " +str(c))
                nr_of_matches = 0                 
                for k in range(batchSize):
                    for number_of_Elements in range(1000):                                      
                        if label_tensor[k][number_of_Elements] == 1 and predicted_tensor[k][0][number_of_Elements] == 1:
                            nr_of_matches += 1
                match_probability = 100 * nr_of_matches / batchSize
                print("How well does the validation-Prediction match on the Labels: " + str(match_probability)+" %")
                if(match_probability > validation_matches):
                    validation_matches+=3
                    mailer.mailto("Number of Matches in the validation Set reached (with higher learnrat(0.05)) "+str(match_probability)+" %. Done in "+ str(i)+ " Steps")
                sess.run(training_init_op)

                saver.save(sess=sess, save_path=origin_path + "../../getfingers_heinz/weights_normal/pretrain_model.ckpt", global_step=i)
                print("model updatet\n")

                
    
    # plt.show()
    print("finished")

    


if __name__ == "__main__":
   
    main()
