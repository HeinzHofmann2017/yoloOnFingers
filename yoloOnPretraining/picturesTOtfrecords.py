#==============================================================================
# 
# Here from all Camera Folders we collect the data together and store
#  it in the trainData-directory as tensorflow-record
#
#==============================================================================

from __future__ import print_function
import numpy as np
import random
import os
import csv
import cv2
import tensorflow as tf
import mailer


def main():

#==============================================================================
#     #training-Data    
#     origin_path = "/media/hhofmann/deeplearning//ilsvrc2012/ILSVRC2012_img_train_t12/"
#     target_path = "/media/hhofmann/deeplearning//ilsvrc2012/train_tf_records/"
#     
# 
#     
#     print("pictures to tfrecord started")
#     list_of_pictures = os.listdir(origin_path)
#     i=0
#     k=0
#     random.seed(448)
#     random.shuffle(list_of_pictures)
#     for j in list_of_pictures:#[0:10234]:#remove this brackets when you want to make all pictures to tf-records
#     #create a record-writer    
#         if(i%10000==0):        
#             path = target_path + "imgNet"+str(k)+".tfrecords"    
#             writer=tf.python_io.TFRecordWriter(path)
#             k+=1#increases if a new record is built
#             print(str(i) + " Pictures done")
# 
#         #Load Picture and convert
#         path=origin_path + str(j)
#         img = np.array(cv2.imread(path))
#         #print(j)
#         shape_of_image = img.shape
#         if len(shape_of_image)==3:
#             #print("worked")
#             height, width, channels = shape_of_image
#             img_raw = img.tostring()
#         
#             example = tf.train.Example(features=tf.train.Features(feature={
#                         'picName':      tf.train.Feature(bytes_list=tf.train.BytesList(value=[j])),        
#                         'height':       tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
#                         'width':        tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
#                         'channels':     tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
#                         'image_raw':    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#                                                                     }))  
#             writer.write(example.SerializeToString())
#         else:
#             mailer.mailto("This Picture failed through the test: " + j + "\n please investigate, why there aren't three dimensions")
# 
#         if(i%100000==0):
#             mailer.mailto("finished with " + str(i) + " Pictures to tfrecords")            
#         i+=1#increases with every picture
#         if(i%10000==0):
#             writer.close()
# 
# 
#             
#     writer.close()
#     mailer.mailto("finished with creating tfrecord from Picture-Dataset")
#     print("finished with creating record")
#==============================================================================
    
    
        #validation-Data
    origin_path = "/media/hhofmann/deeplearning//ilsvrc2012/ILSVRC2012_img_val/"
    target_path = "/media/hhofmann/deeplearning//ilsvrc2012/valid_tf_records/"
    
    print("pictures to tfrecord started")
    list_of_pictures = os.listdir(origin_path)
    i=0
    random.seed(448)
    random.shuffle(list_of_pictures)
    path = target_path + "imgNet_valid.tfrecords"    
    writer=tf.python_io.TFRecordWriter(path)
    for j in list_of_pictures:#[0:10234]:#remove this brackets when you want to make all pictures to tf-records
    #create a record-writer    


        #Load Picture and convert
        path=origin_path + str(j)
        img = np.array(cv2.imread(path))
        #print(j)
        shape_of_image = img.shape
        if j.endswith(".JPEG"):
            #print("worked")
            height, width, channels = shape_of_image
            img_raw = img.tostring()
        
            example = tf.train.Example(features=tf.train.Features(feature={
                        'picName':      tf.train.Feature(bytes_list=tf.train.BytesList(value=[j])),        
                        'height':       tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        'width':        tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        'channels':     tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
                        'image_raw':    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                                                                    }))  
            writer.write(example.SerializeToString())
        else:
            mailer.mailto("This Picture failed through the test: " + j + "\n please investigate, why it isn't a jpeg")
            print("Problem with" + j)
        if(i%10000==0):
            print(str(i) + " Pictures done")
        i+=1

    print("try to close writer")      
    writer.close()
    print("finished with creating record")

if __name__ == '__main__':
    main()