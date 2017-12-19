#
#This Function collects the white-Pictures from all cameras together,
#gives them a new id and put it in the Folder train-Data aside the camera Folders
#(the depending csv's are collected together in the same way.)
#
#
#
from __future__ import print_function

#%matplotlib inline

import matplotlib as mtplt
import numpy as np
import os
import sys
import csv
import cv2
import tensorflow as tf

origin_path = "/home/hhofmann/Schreibtisch/Daten/indexfinger_right/3000_readyTOlearn/"
number_of_Cameras = 4

def main():
    
    #make a new folder if it doesn't allready exist
    path = origin_path + "trainData"
    if not os.path.exists(path):
        os.makedirs(path)
    #delete old csv-file if exist
    path =  origin_path + "trainData/fingers.csv"
    if os.path.isfile(path):
        os.remove(path)
    #make global index for naming new files
    global_index = 0
        
    for camera_nr in range(number_of_Cameras):
        #read Data from csv-file
        path = origin_path + "Camera_"+str(camera_nr)+"/UV_Bin/fingers.csv" 
        with open(path) as csvfile_read: 
            #make a new csv-file
            path = origin_path + "trainData/fingers.csv" 
            with open(path,'a') as csvfile_write:        
                reader = csv.DictReader(csvfile_read,fieldnames=["picName","x_coord","y_coord","width","height","C","P"])
                writer = csv.DictWriter(csvfile_write, fieldnames=["picName","x_coord","y_coord","width","height","C","P"])            
                for row in reader:
                    #print(row["picName"], row["x_coord"], row["y_coord"], row["width"], row["height"], row["C"], row["P"])
                               
                    #create new picname for pic                
                    global_picName= "pic"+str(global_index)+".png" 
                    #write the information from old csv to new csv
                    writer.writerow({'picName'  :global_picName, 
                                     'x_coord'  :row["x_coord"], 
                                     'y_coord'  :row["y_coord"],
                                     'width'    :row["width"],
                                     'height'   :row["height"],
                                     'C'        :row["C"],
                                     'P'        :row["P"]})
                                     
                    #read image and write it with new index
                    path = origin_path + "Camera_"+str(camera_nr)+"/WHITE/" + row["picName"]
                    img = cv2.imread(path)
                    path = origin_path + "trainData/" + global_picName
                    cv2.imwrite(path,img)
                    global_index+=1
                    #if(global_index %200 == 0):
                     #   print(str(global_index))
    print("finished")


if __name__ == '__main__':
    main()