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
import heinzAPI as hAPI

#import matplotlib.pyplot as plt

Environment = "dgx"
#Environment = "Desktop"
print("Enviroment =", Environment)

if (Environment         == "Desktop"):    
    batchSize           = 2
    learning_rate       = 0.001
    capacity            = 10
    num_threads         = 2
    min_after_dequeue   = 5
    buffer_size         = 20
    origin_path         ="/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"#Desktop-path  
    image_path          ="/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"#Desktop-path 
    summarypath         = origin_path + "../../getfingers_heinz/summarys/summarydesktop"
    mailtext            ="training on Desktop"
    nr_of_epochs       = 3
    nr_of_epochs_until_save_model = 1
    
elif (Environment       == "dgx"):
    batchSize           = 64
    learning_rate       = 1 
    capacity            = 2000
    num_threads         = 8
    min_after_dequeue   = 1000
    buffer_size         = 100000
    origin_path         ="/mnt/data/ilsvrc2012/LabelList_Heinz/"#dgx-path
    image_path          ="/mnt/fast/ilsvrc2012/ILSVRC2012_img_train_t12/"
    summarypath         = origin_path + "../../getfingers_heinz/summarys/summary_batchnorm"
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
#==============================================================================
#                                                       
# HIer Graph aufbauen:                                                      
#                                                       
#==============================================================================
    #Conv. Layer 7x7x64-s-2                                                  
    output_1 = hAPI.convLayer(tensor=images,layerNr=1, filterwidth=7, inputdepth=1, outputdepth=64, strides=2)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("2_Maxpool_Layer") as scope:
        output_2 = tf.nn.max_pool(output_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_2")
    #Conv Layer 3x3x192
    output_3 = hAPI.convLayer(tensor=output_2,layerNr=3, filterwidth=3, inputdepth=64, outputdepth=192, strides=1)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("4_Maxpool_Layer") as scope:
        output_4 = tf.nn.max_pool(output_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_4")
    #Conv Layer 1x1x128
    output_5 = hAPI.convLayer(tensor=output_4,layerNr=5, filterwidth=1, inputdepth=192, outputdepth=128, strides=1)
    #Conv Layer 3x3x256 
    output_6 = hAPI.convLayer(tensor=output_5,layerNr=6, filterwidth=3, inputdepth=128, outputdepth=256, strides=1)
    #Conv Layer 1x1x256
    output_7 = hAPI.convLayer(tensor=output_6,layerNr=7, filterwidth=1, inputdepth=256, outputdepth=256, strides=1)
    #Conv Layer 3x3x512
    output_8 = hAPI.convLayer(tensor=output_7,layerNr=8, filterwidth=3, inputdepth=256, outputdepth=512, strides=1)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("9_Maxpool_Layer") as scope:
        output_9 = tf.nn.max_pool(output_8,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_9")
    #Conv Layer 1x1x256
    output_10 = hAPI.convLayer(tensor=output_9,layerNr=10, filterwidth=1, inputdepth=512, outputdepth=256, strides=1)
    #Conv Layer 3x3x512
    output_11 = hAPI.convLayer(tensor=output_10,layerNr=11, filterwidth=3, inputdepth=256, outputdepth=512, strides=1)
    #Conv Layer 1x1x256
    output_12 = hAPI.convLayer(tensor=output_11,layerNr=12, filterwidth=1, inputdepth=512, outputdepth=256, strides=1)
    #Conv Layer 3x3x512
    output_13 = hAPI.convLayer(tensor=output_12,layerNr=13, filterwidth=3, inputdepth=256, outputdepth=512, strides=1)
    #Conv Layer 1x1x256
    output_14 = hAPI.convLayer(tensor=output_13,layerNr=14, filterwidth=1, inputdepth=512, outputdepth=256, strides=1)
    #Conv Layer 3x3x512
    output_15 = hAPI.convLayer(tensor=output_14,layerNr=15, filterwidth=3, inputdepth=256, outputdepth=512, strides=1)
    #Conv Layer 1x1x256
    output_16 = hAPI.convLayer(tensor=output_15,layerNr=16, filterwidth=1, inputdepth=512, outputdepth=256, strides=1)
    #Conv Layer 3x3x512
    output_17 = hAPI.convLayer(tensor=output_16,layerNr=17, filterwidth=3, inputdepth=256, outputdepth=512, strides=1)
    #Conv Layer 1x1x512
    output_18 = hAPI.convLayer(tensor=output_17,layerNr=18, filterwidth=1, inputdepth=512, outputdepth=512, strides=1)
    #Conv Layer 3x3x1024
    output_19 = hAPI.convLayer(tensor=output_18,layerNr=19, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("20_Maxpool_Layer") as scope:
        output_20 = tf.nn.max_pool(output_19,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_20")
    #Conv Layer 1x1x512
    output_21 = hAPI.convLayer(tensor=output_20,layerNr=21, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1)
    #Conv Layer 3x3x1024
    output_22 = hAPI.convLayer(tensor=output_21,layerNr=22, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1)
    #Conv Layer 1x1x512
    output_23 = hAPI.convLayer(tensor=output_22,layerNr=23, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1)
    #Conv Layer 3x3x1024
    output_24 = hAPI.convLayer(tensor=output_23,layerNr=24, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1)
#==============================================================================
# Layer 25:
#     Averagepool 3x3 -s-3
#==============================================================================
    with tf.name_scope("25_AvgPool_Layer") as scope:
        avgpool_25 = tf.nn.avg_pool(output_24,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME", name="avgpool_25")
#==============================================================================
# Layer 26:
#     Fully connected Layer
#==============================================================================
    with tf.name_scope("26_full_Layer") as scope:
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
            tf.summary.histogram(var.op.name +"gradients", grad)
        
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
        
        writer=tf.summary.FileWriter(summarypath)
        writer.add_graph(sess.graph) 
        
        training_matches = 0
        validation_matches = 0
        print("start training....\n")
        for i in range(nr_of_epochs):
            
            #training:
            _ = sess.run([train_step])
            writer.add_summary(sess.run(merged_summary_op),i)
            print(str(i))
            
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
