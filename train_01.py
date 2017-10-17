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


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("TensorFlow version ", tf.__version__)


              
    origin_path="/home/hhofmann/Schreibtisch/Daten/indexfinger_right/3000_readyTOlearn/trainData/trainData.tfrecords"    
    filename_queue = tf.train.string_input_producer([origin_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
                                'height':       tf.FixedLenFeature([],tf.int64),
                                'width':        tf.FixedLenFeature([],tf.int64),
                                'channels':     tf.FixedLenFeature([],tf.int64),
                                'image_raw':    tf.FixedLenFeature([],tf.string),
                                'x_coord':      tf.FixedLenFeature([],tf.float32),
                                'y_coord':      tf.FixedLenFeature([],tf.float32),
                                'box_width':    tf.FixedLenFeature([],tf.float32),
                                'box_height':   tf.FixedLenFeature([],tf.float32),
                                'C':            tf.FixedLenFeature([],tf.float32),
                                'P':            tf.FixedLenFeature([],tf.float32)                            
                                })
    #get back the "2D"-Image out of the bitstream
    HEIGHT=960
    WIDTH=1280
    CHANNELS=3
    image=tf.decode_raw(features["image_raw"],out_type=tf.uint8) #TODO: verschiedene Fehler, je nach Datentyp im Beispiel wars uint8   
    image_float = tf.cast(image, tf.float32)
    image_shape = tf.stack([HEIGHT, WIDTH, CHANNELS])
    reshaped_image       = tf.reshape(image_float, image_shape)
    #get the coordinates
    x_coord = features['x_coord']
    y_coord = features['y_coord']
    #Make automatic shuffled batches out of the dataset('s)
    images, x_coords, y_coords=tf.train.shuffle_batch([reshaped_image,x_coord,y_coord], 
                                                      batch_size=100,#Number of Pictures&Labels per Batch
                                                      capacity=50000,#max Number of Elements in the queue 
                                                      num_threads=2, #Nr. of Threads, which enqueueing tensor-list.
                                                      min_after_dequeue=50
                                                      )
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
        W1 = tf.Variable(tf.truncated_normal(shape=[7,7,3,64], dtype=tf.float32),name='W1')
        b1 = tf.Variable(tf.truncated_normal(shape=[64],dtype=tf.float32),name='b1')
        conv_1_unbiased=tf.nn.conv2d(input=images,filter=W1,strides=[1,2,2,1],padding='SAME',name="conv_1_unbiased")
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
        W3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,192],dtype=tf.float32), name='W3')
        b3 = tf.Variable(tf.truncated_normal(shape=[192],dtype=tf.float32),name='b3')
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
    with tf.name_scope("Layer5_Conv") as scope:     #TODO: ???
        W5 = tf.Variable(tf.truncated_normal(shape=[1,1,192,128],dtype=tf.float32),name="W5")
        b5 = tf.Variable(tf.truncated_normal(shape=[128],dtype=tf.float32),name="b5")
        conv_5_unbiased = tf.nn.conv2d(input=mpool_4, filter=W5,strides=[1,1,1,1], padding="SAME", name="conv_5_unbiased")
        conv_5_linear = tf.add(conv_5_unbiased, b5, name = "conv_5_linear")
        conv_5 = tf.maximum(0.1*conv_5_linear, conv_5_linear, name="leaky_relu_5")
#==============================================================================
# Layer 6:
#     Conv Layer 3x3x256        
#==============================================================================
    with tf.name_scope("Layer6_Conv") as scope:
        W6 = tf.Variable(tf.truncated_normal(shape=[3,3,128,256], dtype=tf.float32),name="W6")
        b6 = tf.Variable(tf.truncated_normal(shape=[256],dtype=tf.float32),name="b6")
        conv_6_unbiased = tf.nn.conv2d(input=conv_5,filter=W6,strides=[1,1,1,1], padding="SAME", name="conv_6_unbiased")
        conv_6_linear = tf.add(conv_6_unbiased, b6, name="conv_6_linear")
        conv_6 = tf.maximum(0.1*conv_6_linear, conv_6_linear, name="leaky_relu_6")
#==============================================================================
# Layer 7:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer7_Conv") as scope:
        W7 = tf.Variable(tf.truncated_normal(shape=[1,1,256,256],dtype=tf.float32),name="W7")
        b7 = tf.Variable(tf.truncated_normal(shape=[256],dtype=tf.float32),name="b7")
        conv_7_unbiased = tf.nn.conv2d(input=conv_6,filter=W7,strides=[1,1,1,1],padding="SAME", name="conv_7_unbiased")
        conv_7_linear = tf.add(conv_7_unbiased, b7, name="conv_7_linear")
        conv_7 = tf.maximum(0.1*conv_7_linear, conv_7_linear, name="leaky_relu_7")
#==============================================================================
# Layer 8:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer8_Conv") as scope:
        W8 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],dtype=tf.float32),name="W8")
        b8 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b8")
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
        W10 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],dtype=tf.float32),name="W10")
        b10 = tf.Variable(tf.truncated_normal(shape=[256],dtype=tf.float32),name="b10")
        conv_10_unbiased = tf.nn.conv2d(input=mpool_9,filter=W10,strides=[1,1,1,1],padding="SAME", name="conv_10_unbiased")
        conv_10_linear = tf.add(conv_10_unbiased, b10, name="conv_10_linear")
        conv_10 = tf.maximum(0.1*conv_10_linear, conv_10_linear, name="leaky_relu_10")
#==============================================================================
# Layer 11:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer11_Conv") as scope:
        W11 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],dtype=tf.float32),name="W11")
        b11 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b11")
        conv_11_unbiased = tf.nn.conv2d(input=conv_10,filter=W11,strides=[1,1,1,1],padding="SAME",name="conv_11_unbiased")
        conv_11_linear = tf.add(conv_11_unbiased,b11,name="conv_11_linear")
        conv_11 = tf.maximum(0.1*conv_11_linear,conv_11_linear, name="leaky_relu_11")
#==============================================================================
# Layer 12:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer12_Conv") as scope:    #Todo: ????
        W12 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],dtype=tf.float32),name="W12")
        b12 = tf.Variable(tf.truncated_normal(shape=[256],dtype=tf.float32),name="b12")
        conv_12_unbiased = tf.nn.conv2d(input=conv_11,filter=W12,strides=[1,1,1,1],padding="SAME", name="conv_12_unbiased")
        conv_12_linear = tf.add(conv_12_unbiased, b12, name="conv_12_linear")
        conv_12 = tf.maximum(0.1*conv_12_linear, conv_12_linear, name="leaky_relu_12")
#==============================================================================
# Layer 13:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer13_Conv") as scope:
        W13 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],dtype=tf.float32),name="W13")
        b13 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b13")
        conv_13_unbiased = tf.nn.conv2d(input=conv_12,filter=W13,strides=[1,1,1,1],padding="SAME",name="conv_13_unbiased")
        conv_13_linear = tf.add(conv_13_unbiased,b13,name="conv_13_linear")
        conv_13 = tf.maximum(0.1*conv_13_linear,conv_13_linear, name="leaky_relu_13")
#==============================================================================
# Layer 14:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer10_Conv") as scope:    #Todo: ????
        W14 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],dtype=tf.float32),name="W14")
        b14 = tf.Variable(tf.truncated_normal(shape=[256],dtype=tf.float32),name="b14")
        conv_14_unbiased = tf.nn.conv2d(input=conv_13,filter=W14,strides=[1,1,1,1],padding="SAME", name="conv_14_unbiased")
        conv_14_linear = tf.add(conv_14_unbiased, b14, name="conv_14_linear")
        conv_14 = tf.maximum(0.1*conv_14_linear, conv_14_linear, name="leaky_relu_14")
#==============================================================================
# Layer 15:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer15_Conv") as scope:
        W15 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],dtype=tf.float32),name="W15")
        b15 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b15")
        conv_15_unbiased = tf.nn.conv2d(input=conv_14,filter=W15,strides=[1,1,1,1],padding="SAME",name="conv_15_unbiased")
        conv_15_linear = tf.add(conv_15_unbiased,b15,name="conv_15_linear")
        conv_15 = tf.maximum(0.1*conv_15_linear,conv_15_linear, name="leaky_relu_15")
#==============================================================================
# Layer 16:
#     Conv Layer 1x1x256
#==============================================================================
    with tf.name_scope("Layer16_Conv") as scope:    #Todo: ????
        W16 = tf.Variable(tf.truncated_normal(shape=[1,1,512,256],dtype=tf.float32),name="W16")
        b16 = tf.Variable(tf.truncated_normal(shape=[256],dtype=tf.float32),name="b16")
        conv_16_unbiased = tf.nn.conv2d(input=conv_15,filter=W16,strides=[1,1,1,1],padding="SAME", name="conv_16_unbiased")
        conv_16_linear = tf.add(conv_16_unbiased, b16, name="conv_16_linear")
        conv_16 = tf.maximum(0.1*conv_16_linear, conv_16_linear, name="leaky_relu_16")
#==============================================================================
# Layer 17:
#     Conv Layer 3x3x512
#==============================================================================
    with tf.name_scope("Layer17_Conv") as scope:
        W17 = tf.Variable(tf.truncated_normal(shape=[3,3,256,512],dtype=tf.float32),name="W17")
        b17 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b17")
        conv_17_unbiased = tf.nn.conv2d(input=conv_16,filter=W17,strides=[1,1,1,1],padding="SAME",name="conv_17_unbiased")
        conv_17_linear = tf.add(conv_17_unbiased,b17,name="conv_17_linear")
        conv_17 = tf.maximum(0.1*conv_17_linear,conv_17_linear, name="leaky_relu_17")   
#==============================================================================
# Layer 18:
#     Conv Layer 1x1x512
#==============================================================================
    with tf.name_scope("Layer18_Conv") as scope:
        W18 = tf.Variable(tf.truncated_normal(shape=[1,1,512,512],dtype=tf.float32),name="W18")
        b18 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b18")
        conv_18_unbiased = tf.nn.conv2d(input=conv_17,filter=W18,strides=[1,1,1,1],padding="SAME", name="conv_18_unbiased")
        conv_18_linear = tf.add(conv_18_unbiased, b18, name="conv_18_linear")
        conv_18 = tf.maximum(0.1*conv_18_linear, conv_18_linear, name="leaky_relu_18")
#==============================================================================
# Layer 19:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer19_Conv") as scope:
        W19 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],dtype=tf.float32),name="W19")
        b19 = tf.Variable(tf.truncated_normal(shape=[1024],dtype=tf.float32),name="b19")
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
        W21 = tf.Variable(tf.truncated_normal(shape=[1,1,1024,512],dtype=tf.float32),name="W21")
        b21 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b21")
        conv_21_unbiased = tf.nn.conv2d(input=mpool_20,filter=W21,strides=[1,1,1,1],padding="SAME", name="conv_21_unbiased")
        conv_21_linear = tf.add(conv_21_unbiased, b21, name="conv_21_linear")
        conv_21 = tf.maximum(0.1*conv_21_linear, conv_21_linear, name="leaky_relu_21")
#==============================================================================
# Layer 22:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer22_Conv") as scope:    
        W22 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],dtype=tf.float32),name="W22")
        b22 = tf.Variable(tf.truncated_normal(shape=[1024],dtype=tf.float32),name="b22")
        conv_22_unbiased = tf.nn.conv2d(input=conv_21,filter=W22,strides=[1,1,1,1],padding="SAME", name="conv_22_unbiased")
        conv_22_linear = tf.add(conv_22_unbiased, b22, name="conv_22_linear")
        conv_22 = tf.maximum(0.1*conv_22_linear, conv_22_linear, name="leaky_relu_22")
#==============================================================================
# Layer 23:
#     Conv Layer 1x1x512
#==============================================================================
    with tf.name_scope("Layer23_Conv") as scope:    #Todo: ????
        W23 = tf.Variable(tf.truncated_normal(shape=[1,1,1024,512],dtype=tf.float32),name="W23")
        b23 = tf.Variable(tf.truncated_normal(shape=[512],dtype=tf.float32),name="b23")
        conv_23_unbiased = tf.nn.conv2d(input=conv_22,filter=W23,strides=[1,1,1,1],padding="SAME", name="conv_23_unbiased")
        conv_23_linear = tf.add(conv_23_unbiased, b23, name="conv_23_linear")
        conv_23 = tf.maximum(0.1*conv_23_linear, conv_23_linear, name="leaky_relu_23")
#==============================================================================
# Layer 24:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer24_Conv") as scope:    
        W24 = tf.Variable(tf.truncated_normal(shape=[3,3,512,1024],dtype=tf.float32),name="W24")
        b24 = tf.Variable(tf.truncated_normal(shape=[1024],dtype=tf.float32),name="b24")
        conv_24_unbiased = tf.nn.conv2d(input=conv_23,filter=W24,strides=[1,1,1,1],padding="SAME", name="conv_24_unbiased")
        conv_24_linear = tf.add(conv_24_unbiased, b24, name="conv_24_linear")
        conv_24 = tf.maximum(0.1*conv_24_linear, conv_24_linear, name="leaky_relu_24")
#==============================================================================
# Layer 25:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer25_Conv") as scope:    
        W25 = tf.Variable(tf.truncated_normal(shape=[3,3,1024,1024],dtype=tf.float32),name="W25")
        b25 = tf.Variable(tf.truncated_normal(shape=[1024],dtype=tf.float32),name="b25")
        conv_25_unbiased = tf.nn.conv2d(input=conv_24,filter=W25,strides=[1,1,1,1],padding="SAME", name="conv_25_unbiased")
        conv_25_linear = tf.add(conv_25_unbiased, b25, name="conv_25_linear")
        conv_25 = tf.maximum(0.1*conv_25_linear, conv_25_linear, name="leaky_relu_25")
#==============================================================================
# Layer 26:
#     Conv Layer 3x3x1024-s-2
#==============================================================================
    with tf.name_scope("Layer26_Conv") as scope:    
        W26 = tf.Variable(tf.truncated_normal(shape=[3,3,1024,1024],dtype=tf.float32),name="W26")
        b26 = tf.Variable(tf.truncated_normal(shape=[1024],dtype=tf.float32),name="b26")
        conv_26_unbiased = tf.nn.conv2d(input=conv_25,filter=W26,strides=[1,2,2,1],padding="SAME", name="conv_26_unbiased")
        conv_26_linear = tf.add(conv_26_unbiased, b26, name="conv_26_linear")
        conv_26 = tf.maximum(0.1*conv_26_linear, conv_26_linear, name="leaky_relu_26")
#==============================================================================
# Layer 27:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer27_Conv") as scope:    
        W27 = tf.Variable(tf.truncated_normal(shape=[3,3,1024,1024],dtype=tf.float32),name="W27")
        b27 = tf.Variable(tf.truncated_normal(shape=[1024],dtype=tf.float32),name="b27")
        conv_27_unbiased = tf.nn.conv2d(input=conv_26,filter=W27,strides=[1,1,1,1],padding="SAME", name="conv_27_unbiased")
        conv_27_linear = tf.add(conv_27_unbiased, b27, name="conv_27_linear")
        conv_27 = tf.maximum(0.1*conv_27_linear, conv_27_linear, name="leaky_relu_27")
#==============================================================================
# Layer 28:
#     Conv Layer 3x3x1024
#==============================================================================
    with tf.name_scope("Layer28_Conv") as scope:    
        W28 = tf.Variable(tf.truncated_normal(shape=[3,3,1024,1024],dtype=tf.float32),name="W28")
        b28 = tf.Variable(tf.truncated_normal(shape=[1024],dtype=tf.float32),name="b28")
        conv_28_unbiased = tf.nn.conv2d(input=conv_27,filter=W28,strides=[1,1,1,1],padding="SAME", name="conv_28_unbiased")
        conv_28_linear = tf.add(conv_28_unbiased, b28, name="conv_28_linear")
        conv_28 = tf.maximum(0.1*conv_28_linear, conv_28_linear, name="leaky_relu_28")
#==============================================================================
# Layer 29:
#     Zero Padding        
#==============================================================================
    with tf.name_scope("Layer29_ZeroPadding") as scope:
        padded_29=tf.pad(conv_28, np.array([[0,0],[3,3],[1,0],[0,0]]))
#==============================================================================
# Layer 30:
#     Maxpool 3x3 -s-3
#==============================================================================
    with tf.name_scope("Layer30_maxpool") as scope:
        mpool_30 = tf.nn.max_pool(padded_29,ksize=[1,3,3,1],strides=[1,3,3,1], padding="SAME", name="maxpool_20")
#==============================================================================
# Layer 31:
#     Fully connected Layer
#==============================================================================
    with tf.name_scope("Layer31_full") as scope:
        input_31 = tf.reshape(tensor=mpool_30,shape=[100,7*7*1024])
        W31 = tf.Variable(tf.truncated_normal(shape=[7*7*1024,4096],dtype=tf.float32),name="W31")
        b31 = tf.Variable(tf.truncated_normal(shape=[4096],dtype=tf.float32),name="b31")
        fully_31 = tf.nn.relu(tf.matmul(input_31,W31)+b31,name="fully_31")
#==============================================================================
# Layer 32:
#     Fully connected Layer
#==============================================================================
    with tf.name_scope("Layer32_full") as scope:
        W32 = tf.Variable(tf.truncated_normal(shape=[4096,15*15*6],dtype=tf.float32),name="W32")
        b32 = tf.Variable(tf.truncated_normal(shape=[15*15*6
        ],dtype=tf.float32),name="b32")
        fully_32 = tf.nn.relu(tf.matmul(fully_31,W32)+b32,name="fully_32")
        output_32 = tf.reshape(tensor=fully_32, shape=[100,15,15,6])
        
    with tf.name_scope("cost_function") as scope:
         cost_function = -tf.reduce_sum(y*tf.log(output_32))





























   
    
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        
        writer=tf.summary.FileWriter("summary") 
        writer.add_graph(sess.graph)     
        
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        
        for i in range(3):
            img, xcor, ycor = sess.run([images,x_coords,y_coords])
            
            print(img.shape)
            print(img[0,:,:,:].shape)
            print(xcor.shape)
            print(ycor.shape)

#==============================================================================
#             plt.figure(i*3)
#             plt.imshow(img[0, :, :, :])
#             plt.figure(i*3+1)
#             plt.imshow(img[1, :, :, :])
#             plt.figure(i*3+2)
#             plt.imshow(img[2, :, :, :])
#==============================================================================
        
        coord.request_stop()
        coord.join(threads)
    

    
    plt.show()
    print("Hello")
    


if __name__ == "__main__":
    main()
