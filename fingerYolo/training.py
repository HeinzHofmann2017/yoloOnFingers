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



def dataset_preprocessor(picname,x_coord,y_coord,prob):
    content = tf.read_file(origin_path + picname)
    image = tf.image.decode_png(content,channels=1)
    image = tf.image.convert_image_dtype(image,tf.float16)
    #ToDo: random Crop here (is a kind of complicated because of the x and y labels.)
    return image,x_coord,y_coord,prob
    
def main():
    print("TensorFlow version ", tf.__version__)
    

    with tf.name_scope("Data") as scope:
        print("read in all Train Picture-Names & Labels and shuffle them")
        ReadData        = analyse_Fingerset.Dataset_Heinz()
        
        
        train_data  = ReadData.get_train_data(origin_path=origin_path)
        train_picnames  = [row[0] for row in train_data]
        train_x_coords  = np.float16([row[1] for row in train_data])
        train_y_coords  = np.float16([row[2] for row in train_data])
        train_probs     = np.float16([row[3] for row in train_data])
        train_data      = Dataset.from_tensor_slices((train_picnames,train_x_coords, train_y_coords, train_probs))
        train_data      = train_data.repeat()
        train_data      = train_data.shuffle(buffer_size=buffer_size)
        train_data      = train_data.map(map_func=dataset_preprocessor,
                                         num_threads=num_threads,
                                         output_buffer_size=10000)
        train_data      = train_data.batch(batchSize)
        print("read in all Valid Picture-Names & Labels and shuffle them")  
        valid_data  = ReadData.get_valid_data(origin_path=origin_path)
        valid_picnames  = [row[0] for row in valid_data]
        valid_x_coords  = np.float16([row[1] for row in valid_data])
        valid_y_coords  = np.float16([row[2] for row in valid_data])
        valid_probs     = np.float16([row[3] for row in valid_data]) 
        valid_data      = Dataset.from_tensor_slices((valid_picnames,valid_x_coords, valid_y_coords, valid_probs))
        valid_data      = valid_data.repeat()
        valid_data      = valid_data.shuffle(buffer_size=buffer_size)
        valid_data      = valid_data.map(map_func=dataset_preprocessor,
                                         num_threads=num_threads,
                                         output_buffer_size=10000)
        valid_data      = valid_data.batch(batchSize)

    with tf.name_scope("Data-Iterator") as scope:        
        iterator        = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        images, x_coords, y_coords, probs  = iterator.get_next()
        
        training_init_op    = iterator.make_initializer(train_data)
        validation_init_op  = iterator.make_initializer(valid_data)
        
        #To test, how the croped picters look like, when they are used to learn...
        tf.summary.image('images_after_crop',tensor = images , max_outputs=20)
            
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
    output_1 = hAPI.convLayerPretrained(tensor=images,layerNr=1,batchSize=batchSize, filterwidth=7, inputdepth=1, outputdepth=64, strides=2, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("2_Maxpool_Layer") as scope:
        output_2 = tf.nn.max_pool(output_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_2")
    #Conv Layer 3x3x192
    output_3 = hAPI.convLayerPretrained(tensor=output_2,layerNr=3,batchSize=batchSize, filterwidth=3, inputdepth=64, outputdepth=192, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("4_Maxpool_Layer") as scope:
        output_4 = tf.nn.max_pool(output_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name="maxpool_4")
    #Conv Layer 1x1x128
    output_5 = hAPI.convLayerPretrained(tensor=output_4,layerNr=5,batchSize=batchSize, filterwidth=1, inputdepth=192, outputdepth=128, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x256 
    output_6 = hAPI.convLayerPretrained(tensor=output_5,layerNr=6,batchSize=batchSize, filterwidth=3, inputdepth=128, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_7 = hAPI.convLayerPretrained(tensor=output_6,layerNr=7,batchSize=batchSize, filterwidth=1, inputdepth=256, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_8 = hAPI.convLayerPretrained(tensor=output_7,layerNr=8,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("9_Maxpool_Layer") as scope:
        output_9 = tf.nn.max_pool(output_8,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_9")
    #Conv Layer 1x1x256
    output_10 = hAPI.convLayerPretrained(tensor=output_9,layerNr=10,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_11 = hAPI.convLayerPretrained(tensor=output_10,layerNr=11,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_12 = hAPI.convLayerPretrained(tensor=output_11,layerNr=12,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_13 = hAPI.convLayerPretrained(tensor=output_12,layerNr=13,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_14 = hAPI.convLayerPretrained(tensor=output_13,layerNr=14,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_15 = hAPI.convLayerPretrained(tensor=output_14,layerNr=15,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 1x1x256
    output_16 = hAPI.convLayerPretrained(tensor=output_15,layerNr=16,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=256, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x512
    output_17 = hAPI.convLayerPretrained(tensor=output_16,layerNr=17,batchSize=batchSize, filterwidth=3, inputdepth=256, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 1x1x512
    output_18 = hAPI.convLayerPretrained(tensor=output_17,layerNr=18,batchSize=batchSize, filterwidth=1, inputdepth=512, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_19 = hAPI.convLayerPretrained(tensor=output_18,layerNr=19,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Maxpool Layer 2x2 -s-2
    with tf.name_scope("20_Maxpool_Layer") as scope:
        output_20 = tf.nn.max_pool(output_19,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name="maxpool_20")
    #Conv Layer 1x1x512
    output_21 = hAPI.convLayerPretrained(tensor=output_20,layerNr=21,batchSize=batchSize, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_22 = hAPI.convLayerPretrained(tensor=output_21,layerNr=22,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 1x1x512
    output_23 = hAPI.convLayerPretrained(tensor=output_22,layerNr=23,batchSize=batchSize, filterwidth=1, inputdepth=1024, outputdepth=512, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_24 = hAPI.convLayerPretrained(tensor=output_23,layerNr=24,batchSize=batchSize, filterwidth=3, inputdepth=512, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training,origin_path=origin_path)
    #Conv Layer 3x3x1024
    output_25 = hAPI.convLayer(tensor=output_24,layerNr=25,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training)
    #Conv Layer 3x3x1024-s-2
    output_26 = hAPI.convLayer(tensor=output_25,layerNr=26,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=2, batchnorm_=batchnorm, dropout_=dropout,training=training)
    #Conv Layer 3x3x1024
    output_27 = hAPI.convLayer(tensor=output_26,layerNr=27,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training)
    #Conv Layer 3x3x1024
    output_28 = hAPI.convLayer(tensor=output_27,layerNr=28,batchSize=batchSize, filterwidth=3, inputdepth=1024, outputdepth=1024, strides=1, batchnorm_=batchnorm, dropout_=dropout,training=training)
    #Zero Padding        
    with tf.name_scope("Layer29_ZeroPadding") as scope:
        output_29=tf.pad(output_28, np.array([[0,0],[3,3],[1,0],[0,0]]))
    #Maxpool 3x3 -s-3
    with tf.name_scope("Layer30_maxpool") as scope:
        output_30 = tf.nn.max_pool(output_29,ksize=[1,3,3,1],strides=[1,3,3,1], padding="SAME")
    #Fully-Connected Layer ==> make vector
    with tf.name_scope("Layer31_full") as scope:
        input_31 = tf.reshape(tensor=output_30,shape=[batchSize,7*7*1024])
        W31 = tf.Variable(tf.truncated_normal(shape=[7*7*1024,4096],stddev=0.01,dtype=tf.float16),name="W31")
        b31 = tf.Variable(tf.truncated_normal(shape=[4096],stddev=0.01,dtype=tf.float16),name="b31")
        preactivate_31 = tf.add(tf.matmul(input_31,W31),b31)        
        output_31 = tf.nn.relu(preactivate_31)
        with tf.name_scope("summary"):
            hAPI.variable_summaries(variable=W31,name="W31")
            hAPI.variable_summaries(variable=b31,name="b31")
            hAPI.variable_summaries(variable=preactivate_31, name="preactivate31")
            hAPI.variable_summaries(variable=output31, name="output31")
    #Fully-Connected Layer ==> make tensor again.
    with tf.name_scope("Layer32_full") as scope:
        W32 = tf.Variable(tf.truncated_normal(shape=[4096,1*1*3],stddev=0.01,dtype=tf.float16),name="W32")
        b32 = tf.Variable(tf.truncated_normal(shape=[1*1*3],stddev=0.01,dtype=tf.float16),name="b32")
        preactivate_32 = tf.add(tf.matmul(output_31,W32),b32)    
        fully_32 = tf.nn.relu(preactivate_32)
        output_32 = tf.reshape(tensor=fully_32, shape=[batchSize,1,1,3])
        with tf.name_scope("summary"):
            hAPI.variable_summaries(variable=W32,name="W32")
            hAPI.variable_summaries(variable=b32,name="b32")
            hAPI.variable_summaries(variable=preactivate_32, name="preactivate32")
            hAPI.variable_summaries(variable=output32, name="output32")
        
    with tf.name_scope("cost_function") as scope:
        cost_x = tf.reduce_mean(tf.multiply(tf.square(tf.subtract(output_32[:,:,:,0],x_coords)),probs))
        cost_y = tf.reduce_mean(tf.multiply(tf.square(tf.subtract(output_32[:,:,:,1],y_coords)),probs))
        cost_p = tf.reduce_mean(tf.square(tf.subtract(output_32[:,:,:,2],probs)))
        cost = tf.add(tf.add(cost_x,cost_y),cost_p)
        cost_h = tf.summary.scalar("Costs",cost)

        
    with tf.name_scope("optimizer") as scope:
        # Gradient descen
        #TODO: Gradient Decent durch ADAM ersetzen
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(epsilon=1e-04)
        
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
        test_vector = tf.ones([batchSize],dtype=tf.float16)
        probability_isnt_correct = tf.greater(tf.sqrt(tf.square(tf.subtract(tf.squeeze(output_32[:,:,:,2]),probs))),0.5)
        with tf.name_scope("Test_Is_Finger_visible"):#-------------------------------------------------------------------------------------------------
            bool_vector_visible = tf.logical_not(probability_isnt_correct)
            result_in_percent_visible = tf.div(tf.multiply(tf.reduce_sum(tf.boolean_mask(test_vector,bool_vector_visible)),100),batchSize)
            visible_test_h = tf.summary.scalar("IsFingerThere_Test",result_in_percent_visible)
        with tf.name_scope("Test_InsideCircleOf0.5Picturesize"):#--------------------------------------------------------------------------------------
            probability_is_higher_0_5 = tf.greater(tf.squeeze(output_32[:,:,:,2]),0.5)
            #                       =sqrt((x-x)^2+(y-y)^2)>0.25
            distance_is_greater_0_5 = tf.greater(tf.sqrt(tf.add(tf.square(tf.subtract(tf.squeeze(output_32[:,:,:,0]),x_coords)),   
                                                                tf.square(tf.subtract(tf.squeeze(output_32[:,:,:,1]),y_coords)))) ,0.25)
            #           = not(probabilityIsntCorrect OR [probabilityIsHigherThan0.5 AND distanceIsGreaterThan0.1])
            bool_vector_0_5 = tf.logical_not(tf.logical_or(probability_isnt_correct,tf.logical_and(probability_is_higher_0_5,distance_is_greater_0_5)))
            result_in_percent_05 = tf.div(tf.multiply(tf.reduce_sum(tf.boolean_mask(test_vector,bool_vector_0_5)),100),batchSize)
            in0_5_test_h = tf.summary.scalar("inCircle_0_5_Test",result_in_percent_05)
        with tf.name_scope("Test_InsideCircleOf0.3Picturesize"):#--------------------------------------------------------------------------------------
            probability_is_higher_0_5 = tf.greater(tf.squeeze(output_32[:,:,:,2]),0.5)
            #                       =sqrt((x-x)^2+(y-y)^2)>0.15
            distance_is_greater_0_3 = tf.greater(tf.sqrt(tf.add(tf.square(tf.subtract(tf.squeeze(output_32[:,:,:,0]),x_coords)),   
                                                                tf.square(tf.subtract(tf.squeeze(output_32[:,:,:,1]),y_coords)))) ,0.15)
            #           = not(probabilityIsntCorrect OR [probabilityIsHigherThan0.5 AND distanceIsGreaterThan0.1])
            bool_vector_0_3 = tf.logical_not(tf.logical_or(probability_isnt_correct,tf.logical_and(probability_is_higher_0_5,distance_is_greater_0_3)))
            result_in_percent_03 = tf.div(tf.multiply(tf.reduce_sum(tf.boolean_mask(test_vector,bool_vector_0_3)),100),batchSize)
            in0_3_test_h = tf.summary.scalar("inCircle_0_3_Test",result_in_percent_03)
        with tf.name_scope("Test_InsideCircleOf0.1Picturesize"):#--------------------------------------------------------------------------------------
            probability_is_higher_0_5 = tf.greater(tf.squeeze(output_32[:,:,:,2]),0.5)
            #                       =sqrt((x-x)^2+(y-y)^2)>0.05
            distance_is_greater_0_1 = tf.greater(tf.sqrt(tf.add(tf.square(tf.subtract(tf.squeeze(output_32[:,:,:,0]),x_coords)),   
                                                                tf.square(tf.subtract(tf.squeeze(output_32[:,:,:,1]),y_coords)))) ,0.05)
            #           = not(probabilityIsntCorrect OR [probabilityIsHigherThan0.5 AND distanceIsGreaterThan0.1])
            bool_vector_0_1 = tf.logical_not(tf.logical_or(probability_isnt_correct,tf.logical_and(probability_is_higher_0_5,distance_is_greater_0_1)))
            result_in_percent_01 = tf.div(tf.multiply(tf.reduce_sum(tf.boolean_mask(test_vector,bool_vector_0_1)),100),batchSize)
            in0_1_test_h = tf.summary.scalar("inCircle_0_1_Test",result_in_percent_01)
            

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
        
        train_writer=tf.summary.FileWriter(origin_path + "../../../../summarys/summary_" + name + "_train")
        train_writer.add_graph(sess.graph) 
        valid_writer=tf.summary.FileWriter(origin_path + "../../../../summarys/summary_" + name + "_valid")
        valid_writer.add_graph(sess.graph) 
        
        training_matches = 0.1#%
        validation_matches = 0.1#%
        #saver.restore(sess=sess, save_path=origin_path + "../../../../weights/7BnormBeforeRelu2.ckpt-00103000")
        print("start training....\n")
        for i in range(nr_of_epochs/nr_of_epochs_until_save_model):
            #training:
            for j in range(nr_of_epochs_until_save_model):
                _ = sess.run([train_step],feed_dict={training: True})

            numbers_of_iterations_until_now = i*nr_of_epochs_until_save_model+j+1            
            #testing on traindata
            train_writer.add_summary(sess.run(merged_summary_op,feed_dict={training: False}),(numbers_of_iterations_until_now))
            matches = sess.run(result_in_percent_visible,   feed_dict={training: False})      
            matches = sess.run(result_in_percent_05,        feed_dict={training: False}) 
            matches = sess.run(result_in_percent_03,        feed_dict={training: False})
            matches = sess.run(result_in_percent_01,        feed_dict={training: False})
            if(matches > training_matches):
                training_matches=matches
                mailer.mailto("\n\n"+name+"\n\n top5-training \n\n Reached: "+str(matches)+" %. \n\n Done in "+ str(numbers_of_iterations_until_now)+ " Steps")
            #testing on validationdata:
            sess.run(validation_init_op)
            valid_writer.add_summary(sess.run(merged_summary_op,feed_dict={training: False}),(numbers_of_iterations_until_now))
            matches = sess.run(result_in_percent_visible,   feed_dict={training: False}) 
            matches = sess.run(result_in_percent_05,        feed_dict={training: False})         
            matches = sess.run(result_in_percent_03,        feed_dict={training: False})
            matches = sess.run(result_in_percent_01,        feed_dict={training: False})
            if(matches > validation_matches):
                validation_matches=matches
                mailer.mailto("\n\n"+name+"\n\n top5-validation \n\n Reached: "+str(matches)+" %. \n\n Done in "+ str(numbers_of_iterations_until_now)+" Steps")
            sess.run(training_init_op)
            
            #save Model
            saver.save(sess=sess, save_path=origin_path + "../../../../weights/"+name+".ckpt", global_step=(numbers_of_iterations_until_now))
            print("model updatet\n")

                
    
    # plt.show()
    print("finished")

    


if __name__ == "__main__":
   
    main()
