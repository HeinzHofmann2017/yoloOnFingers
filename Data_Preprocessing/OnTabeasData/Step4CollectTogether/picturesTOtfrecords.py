#==============================================================================
# 
# Here from all Camera Folders we collect the data together and store
#  it in the trainData-directory as tensorflow-record
#
#==============================================================================

from __future__ import print_function
import numpy as np
import os
import csv
import cv2
import tensorflow as tf

def main():
    origin_path = "/home/hhofmann/Schreibtisch/Daten/mini_Dataset/"
    number_of_Cameras=4
    print("this Program needs about 1 Minute per 1000 Pictures...")
    #make a new folder if it doesn't allready exist
    path = origin_path + "trainData"
    if not os.path.exists(path):
        os.makedirs(path)
    global_index=0
    #create a record-writer    
    path = origin_path + "trainData/trainData.tfrecords"    
    writer=tf.python_io.TFRecordWriter(path)
    for camera_nr in range(number_of_Cameras):
        #read Data from csv-file
        path = origin_path + "Camera_" + str(camera_nr) + "/UV_Bin/fingers.csv"
        with open(path) as csvfile_read:
            reader = csv.DictReader(csvfile_read,fieldnames=["picName","x_coord","y_coord","width","height","C","P"])
            for row in reader:
                #create a new picname for pic
                global_picName = "pic"+str(global_index)+".png"
                #Load Picture and convert
                path=origin_path +  "Camera_" + str(camera_nr) + "/WHITE/" + row["picName"]
                img = np.array(cv2.imread(path))
                height, width, channels = img.shape
                img_raw = img.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                                #'picName':      tf.train.Feature(int64_list=tf.train.Int64List(value=[global_picName])),        
                                'height':       tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                                'width':        tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                                'channels':     tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
                                'image_raw':    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                                'x_coord':      tf.train.Feature(float_list=tf.train.FloatList(value=[float(row["x_coord"])])),
                                'y_coord':      tf.train.Feature(float_list=tf.train.FloatList(value=[float(row["y_coord"])])),
                                'box_width':    tf.train.Feature(float_list=tf.train.FloatList(value=[float(row["width"])])),
                                'box_height':   tf.train.Feature(float_list=tf.train.FloatList(value=[float(row["height"])])),
                                'C':            tf.train.Feature(float_list=tf.train.FloatList(value=[float(row["C"])])),
                                'P':            tf.train.Feature(float_list=tf.train.FloatList(value=[float(row["P"])]))
                                                                            }))  
                global_index+=1
                writer.write(example.SerializeToString())
    writer.close()
    print("finished with creating record")


if __name__ == '__main__':
    main()