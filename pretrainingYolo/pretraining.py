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
import time

this_folder =  os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_folder+ '/../dataPreprocessing/OnILSVRCdata/')
import analyse_Dataset
sys.path.insert(0,this_folder+"/../helperfunctions/")
import mailer
import heinzAPI as hAPI
import parserClass as pC

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



def dataset_preprocessor(picname,labels):
    content = tf.read_file(origin_path + "../ILSVRC2012_img_train_t12_grayscale/" + picname)
    #content = tf.read_file(image_path+"../ILSVRC2012_img_train_t12_grayscale/"+picname)
    image = tf.image.decode_jpeg(content,channels=1)
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.random_crop(image,[224,224,1])
    return image,labels
    
def main():
    print("TensorFlow version ", tf.__version__)    
    with tf.name_scope("Data") as scope:
        print("read in all Picture-Names & Labels and shuffle them")
        ReadData        = analyse_Dataset.Dataset_Heinz()
        train_picnames  = ReadData.get_train_picnames(origin_path=origin_path)
        train_labels    = ReadData.get_train_labels(origin_path=origin_path)#labels between 0 & 999
        train_data      = Dataset.from_tensor_slices((train_picnames,train_labels))
        train_data      = train_data.repeat()
        train_data      = train_data.shuffle(buffer_size=buffer_size)#buffersize must be minimum size of the whole dataset
        train_data      = train_data.map(map_func=dataset_preprocessor,
                                         num_parallel_calls=num_threads)
        train_data      = train_data.batch(batchSize)
        
        valid_picnames  = ReadData.get_valid_picnames(origin_path=origin_path)
        valid_labels    = ReadData.get_valid_labels(origin_path=origin_path)
        valid_data      = Dataset.from_tensor_slices((valid_picnames,valid_labels))
        valid_data      = valid_data.repeat()
        valid_data      = valid_data.shuffle(buffer_size=buffer_size)#buffersize must be minimum size of the whole dataset
        valid_data      = valid_data.map(map_func=dataset_preprocessor,
                                         num_parallel_calls=num_threads)
        valid_data      = valid_data.batch(batchSize)
    with tf.name_scope("Data-Iterator") as scope:        
        iterator        = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        images, labels  = iterator.get_next()
        
        training_init_op    = iterator.make_initializer(train_data)
        validation_init_op  = iterator.make_initializer(valid_data)
        
        #To test, how the croped picters look like, when they are used to learn...
        tf.summary.image('images_after_crop',tensor = images , max_outputs=12)
            
    with tf.name_scope("make_labels") as scope:
        labels = tf.one_hot(indices     =   labels,
                            depth       =   1000,
                            on_value    =   1.0,
                            off_value   =   0.0,
                            axis        =   -1,
                            dtype       =   tf.float32)
    #is true,if the model is training right now, and is False, if the model is testing.
    training = tf.placeholder(tf.bool, name='training')
    with tf.name_scope("normalize_pictures") as scope:                            
        images = hAPI.normalize_pictures(tensor=images)
#==============================================================================
#                                                       
# HIer Graph aufbauen:                                                      
#                                                       
#==============================================================================
    #Conv. Layer 7x7x64-s-2                                                  
    output_1 = hAPI.convLayer(tensor=images,layerNr=1,batchSize=batchSize, filterwidth=7, inputdepth=1, outputdepth=64, strides=2, batchnorm_=batchnorm, dropout_=False,training=training)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("2_Maxpool_Layer") as scope:
        output_2 = tf.nn.max_pool(output_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_2")
    #Conv Layer 3x3x192
    output_3 = hAPI.convLayer(tensor=output_2,layerNr=3,batchSize=batchSize, filterwidth=3, inputdepth=64, outputdepth=192, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("4_Maxpool_Layer") as scope:
        output_4 = tf.nn.max_pool(output_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_4")
    #Conv Layer 1x1x128
    output_5 = hAPI.convLayer(tensor=output_4,layerNr=5,batchSize=batchSize, filterwidth=1, inputdepth=192, outputdepth=128, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x256 
    output_6 = hAPI.convLayer(tensor=output_5,layerNr=6,batchSize=batchSize, filterwidth=3, inputdepth=128, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 1x1x256
    output_7 = hAPI.convLayer(tensor=output_6,layerNr=7,batchSize=batchSize, filterwidth=1, inputdepth=256, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x512
    output_8 = hAPI.convLayer(tensor=output_7,layerNr=8,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("9_Maxpool_Layer") as scope:
        output_9 = tf.nn.max_pool(output_8,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_9")
    #Conv Layer 1x1x256
    output_10 = hAPI.convLayer(tensor=output_9,layerNr=10,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x512
    output_11 = hAPI.convLayer(tensor=output_10,layerNr=11,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 1x1x256
    output_12 = hAPI.convLayer(tensor=output_11,layerNr=12,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x512
    output_13 = hAPI.convLayer(tensor=output_12,layerNr=13,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 1x1x256
    output_14 = hAPI.convLayer(tensor=output_13,layerNr=14,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x512
    output_15 = hAPI.convLayer(tensor=output_14,layerNr=15,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 1x1x256
    output_16 = hAPI.convLayer(tensor=output_15,layerNr=16,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x512
    output_17 = hAPI.convLayer(tensor=output_16,layerNr=17,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 1x1x512
    output_18 = hAPI.convLayer(tensor=output_17,layerNr=18,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Conv Layer 3x3x1024
    output_19 = hAPI.convLayer(tensor=output_18,layerNr=19,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=False,training=training)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("20_Maxpool_Layer") as scope:
        output_20 = tf.nn.max_pool(output_19,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_20")
    #Conv Layer 1x1x512
    output_21 = hAPI.convLayer(tensor=output_20,layerNr=21,batchSize=batchSize, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training)
    #Conv Layer 3x3x1024
    output_22 = hAPI.convLayer(tensor=output_21,layerNr=22,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training)
    #Conv Layer 1x1x512
    output_23 = hAPI.convLayer(tensor=output_22,layerNr=23,batchSize=batchSize, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training)
    #Conv Layer 3x3x1024
    output_24 = hAPI.convLayer(tensor=output_23,layerNr=24,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training)
#==============================================================================
# Layer 25:
#     Averagepool 3x3 -s-3
#==============================================================================
    with tf.name_scope("25_AvgPool_Layer") as scope:
        avgpool_25 = tf.nn.avg_pool(output_24,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME", name="avgpool_25")#Todo: set the 7 back to 2
#==============================================================================
# Layer 26:
#     Fully connected Layer
#==============================================================================
    with tf.name_scope("26_full_Layer") as scope:
        input_26 = tf.reshape(tensor=avgpool_25, shape=[batchSize,4*4*1024]) #Todo: Set the 1 back to 4
        if dropout == True:
            with tf.name_scope("dropout"):        
                #dropout only over all the feature-maps and batches.
                input_26 = tf.cond(training,
                                      lambda:tf.nn.dropout(x=input_26, keep_prob=0.8,noise_shape=[batchSize,4*4*1024]),
                                      lambda:input_26)
        W26 = tf.Variable(tf.truncated_normal(shape=[4*4*1024,1000],stddev=0.01,dtype=tf.float32),name="W26") #Todo: Set the 1 back to 4
        b26 = tf.Variable(tf.truncated_normal(shape=[1000],stddev=0.01,dtype=tf.float32),name="b26")
        W26_h = tf.summary.histogram("weights26",W26)
        b26_h = tf.summary.histogram("biases26",b26)
        prerelu26 = tf.matmul(input_26,W26)+b26
        with tf.name_scope("batch_norm"):
            input_depth_26 = prerelu26.get_shape().as_list()[-1]#takes the last element which is in this case 64
            #make new weights and new bias
            with tf.name_scope("beta"):
                beta26 = tf.Variable(tf.constant(0.0,shape=[input_depth_26],dtype=tf.float32), name="beta",trainable=True)
            with tf.name_scope("gamma"):
                gamma26 = tf.Variable(tf.constant(1.0,shape=[input_depth_26],dtype=tf.float32),name="gamma",trainable=True)
            batch_mean26, batch_variance26 = tf.nn.moments(x=prerelu26,axes=[0,1])
            prerelu26 = tf.nn.batch_normalization(x=prerelu26,mean=batch_mean26,variance=batch_variance26,offset=beta26,scale=gamma26,variance_epsilon=1e-4,name=None)
        prerelu26_h = tf.summary.histogram("prerelu26",prerelu26)
        fully_26 = prerelu26#tf.nn.relu(prerelu26)


        
    with tf.name_scope("cost_function") as scope:
        fully_26 = tf.cast(fully_26, tf.float32)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fully_26))
        cost_h = tf.summary.scalar("Costs",cost)

        
    with tf.name_scope("optimizer") as scope:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1e-04)##From 4Adam to 13dropoutLastFewLayers0001lRate everything learned with the Adam default-learnrate of 0.001
        
        # grads_and_vars is a list of tuples (gradient, variable). Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        grads_and_vars = optimizer.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
        
        # Ask the optimizer to apply the capped gradients.
        train_step = optimizer.apply_gradients(capped_gvs)
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name +"gradients", grad)
    for grad, var in capped_gvs:
        if grad is not None:
            tf.summary.histogram(var.op.name +"capped_gradients",grad)
    
    with tf.name_scope("Test") as scope:
        #squeeze is important to remove not needed dimensions. such dimensions would affect the result
        test_vectors = tf.squeeze(tf.one_hot(tf.nn.top_k(fully_26).indices,tf.shape(fully_26)[1]))
        number_of_matches = tf.reduce_sum(tf.multiply(x=test_vectors,y=labels))    
        matches_in_percent= tf.div(x=tf.multiply(x=number_of_matches,y=100),y=batchSize)
        test_h = tf.summary.scalar("Test",matches_in_percent)
        
    with tf.name_scope("Test_top5") as scope:
        #make one-hot-vector for every top-5 Probability  and reduce them together (squeezes are for get rid of unused dimensionalities)
        top5_test_vectors = tf.squeeze(tf.reduce_sum(tf.squeeze(tf.one_hot(tf.nn.top_k(fully_26,k=5).indices,tf.shape(fully_26)[1])),axis=1))
        top5_number_of_matches = tf.reduce_sum(tf.multiply(x=top5_test_vectors,y=labels))    
        top5_matches_in_percent= tf.div(x=tf.multiply(x=top5_number_of_matches,y=100),y=batchSize)
        top5_test_h = tf.summary.scalar("Top5_Test",top5_matches_in_percent)
        


    
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver(
        max_to_keep=5,
        keep_checkpoint_every_n_hours=4.0, 
        pad_step_number=True,
        save_relative_paths=True,)
    if not os.path.exists(origin_path + "../../data_hhofmann/weights/"+name+"/"):
        os.makedirs(origin_path + "../../data_hhofmann/weights/"+name+"/")
    
    merged_summary_op = tf.summary.merge_all()
    
    with tf.Session() as sess:
        
        sess.run(validation_init_op)
        sess.run(training_init_op)
        sess.run(init_op)
        
        
        train_train_writer = tf.summary.FileWriter(origin_path + "../../data_hhofmann/summarys/pretraining/summary_" + name + "_traintrain")
        train_train_writer.add_graph(sess.graph)  
        train_train2_writer = tf.summary.FileWriter(origin_path + "../../data_hhofmann/summarys/pretraining/summary_" + name + "_traintrain2")
        train_train2_writer.add_graph(sess.graph) 
        train_writer=tf.summary.FileWriter(origin_path + "../../data_hhofmann/summarys/pretraining/summary_" + name + "_train")
        train_writer.add_graph(sess.graph) 
        valid_writer=tf.summary.FileWriter(origin_path + "../../data_hhofmann/summarys/pretraining/summary_" + name + "_valid")
        valid_writer.add_graph(sess.graph) 
        
        training_matches = 0.1#%
        validation_matches = 0.1#%
        print("start training....\n")
        for i in range(nr_of_epochs/nr_of_epochs_until_save_model):
            #training:
            for j in range(nr_of_epochs_until_save_model):
                _ = sess.run([train_step],feed_dict={training: True})



            #testing while training on traindata
            
            numbers_of_iterations_until_now = i*nr_of_epochs_until_save_model+j+1
            sumsum,_,sumsum2 = sess.run([merged_summary_op,train_step,merged_summary_op],feed_dict={training: True})
            train_train_writer.add_summary(sumsum,(numbers_of_iterations_until_now)) 
            train_train2_writer.add_summary(sumsum,(numbers_of_iterations_until_now))
               
                
            #testing on traindata
            numbers_of_iterations_until_now = i*nr_of_epochs_until_save_model+j+1
            train_writer.add_summary(sess.run(merged_summary_op,feed_dict={training: False}),(numbers_of_iterations_until_now))
            matches = sess.run(matches_in_percent,feed_dict={training: False})
            matches = sess.run(top5_matches_in_percent,feed_dict={training: False})
            if(matches > training_matches):
                training_matches=matches
                mailer.mailto("\n\n"+name+"\n\n top5-training \n\n Reached: "+str(matches)+" %. \n\n Done in "+ str(numbers_of_iterations_until_now)+ " Steps")
            
            #testing on validationdata:
            sess.run(validation_init_op)
            valid_writer.add_summary(sess.run(merged_summary_op,feed_dict={training: False}),(numbers_of_iterations_until_now))
            matches = sess.run(matches_in_percent,feed_dict={training: False})
            matches = sess.run(top5_matches_in_percent,feed_dict={training: False})
            if(matches > validation_matches):
                validation_matches=matches
                mailer.mailto("\n\n"+name+"\n\n top5-validation \n\n Reached: "+str(matches)+" %. \n\n Done in "+ str(numbers_of_iterations_until_now)+" Steps")
            sess.run(training_init_op)
            
            #save Model
            saver.save(sess=sess, save_path=origin_path + "../../data_hhofmann/weights/"+name+"/"+name+".ckpt", global_step=(numbers_of_iterations_until_now))
            print("model "+name+" updatet\n")

                
    
    # plt.show()
    print("finished")

    


if __name__ == "__main__":
   
    main()
