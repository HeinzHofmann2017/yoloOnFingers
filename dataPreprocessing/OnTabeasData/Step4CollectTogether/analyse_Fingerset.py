# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:20:51 2017

@author: hhofmann
"""

from __future__ import print_function
#from PIL import Image
import numpy as np
import os
import sys
import csv
import pickle
import random
#import cv2
this_folder =  os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_folder+ '/../../../helperfunctions/')
import mailer


class Dataset_Heinz(object):
    def __init__(self):

        
        self.all_picture_names = []
        self.train_picnames = []
        self.train_labels = []
        self.valid_picnames = []
        self.valid_labels = []
        
        
#==============================================================================
#         #This path btw. this folder "trainData" has to be in the same folder in which the Four Camera_Folders are
#     def make_lists(self,origin_path="/media/hhofmann/dgx/data_hhofmann/Data/mit Tabea gesammelt/test/trainData/",nrOfCams=4):
#         self.all_picture_names = 12
# 
#         #has to be commented out, because there is no cv2 in the docker for learning
#         #if path doesn't allready exist, create it.
#         if not os.path.exists(origin_path):
#             os.makedirs(origin_path + "../trainData")
# 
#         for dataSet in ["Valid","Train"]:
#                     
#             #get number of Pictures
#             nrOfElements = 0
#             for camera_nr in range(nrOfCams):
#                 with open(origin_path + "../Camera_"+str(camera_nr)+"/UV_Bin/fingers"+dataSet+".csv") as csvfile_read:
#                     reader = csv.DictReader(csvfile_read,fieldnames=["picName","x_coord","y_coord","width","height","C","P"])
#                     for row in reader:
#                         nrOfElements+=1
#             if(dataSet == "Train"):
#                 global_index = 0
#                 label_tensor    = np.zeros((nrOfElements,7,7,5))
#             elif(dataSet == "Valid"):
#                 global_index = 100000                            
#                 label_tensor    = np.zeros((nrOfElements+100000,7,7,5))
#             self.data = [[]for i in range(nrOfElements)]        
# 
#             
#             #save picture information in Array and store pictures with new names in new folder.
#             print("copy pictures will need about 1.5 minute per 1000 pictures")            
#     
#             for camera_nr in range(nrOfCams):
#                 #read Data from csv-file
#                 print("start with Camera " + str(camera_nr))
#                 with open(origin_path + "../Camera_"+str(camera_nr)+"/UV_Bin/fingers"+dataSet+".csv") as csvfile_read:
#                     reader = csv.DictReader(csvfile_read,fieldnames=["picName","x_coord","y_coord","width","height","C","P"])
#                     #for every picture do
#                                                     
#                     for row in reader:
#                         global_picName = "pic"+str(global_index)+".png"
#                         for cellHeight in range(7):#counts from cell 0 to cell 6
#                             for cellWidth in range(7):#counts from cell 0 to cell 6                               
#                                 for i in range(5):
#                                     label_tensor[global_index,cellHeight,cellWidth,i]=0
#                         if float(row["P"]) > 0:
#                             x_coord = (float(row["x_coord"])/1280)*7#This coordinates are between 0 and 7 like the Grid on the picture has 7*7 fields
#                             y_coord = (float(row["y_coord"])/960)*7#This coordinates are between 0 and 7 like the Grid on the picture has 7*7 fields
#                             x_box_and_offset = int(x_coord - 0.0000001) #round down the coordinates to the next lower integer                             
#                             y_box_and_offset = int(y_coord - 0.0000001) #round down the coordinates to the next lower integer
#                             x_fine = x_coord - x_box_and_offset#get the coordinates from the minibox, which is between 0 and 1
#                             y_fine = y_coord -y_box_and_offset#get the coordinates from the minibox, which is between 0 and 1
#                             if(x_coord < 0 or x_coord>7):
#                                 print("Error, x_coord is "+str(x_coord)+" in "+str(row["picName"]))
#                                 x_coord=7
#                             elif(y_coord < 0 or y_coord>7):
#                                 print("Error, y_coord is "+str(y_coord)+" in "+str(row["picName"]))
#                                 y_coord=7
#                             elif(x_box_and_offset < 0 or x_box_and_offset>=7):
#                                 print("Error, x_box_and_offset is "+str(x_box_and_offset)+" in "+str(row["picName"]))
#                                 x_box_and_offset = 6
#                             elif(y_box_and_offset < 0 or y_box_and_offset>=7):
#                                 print("Error, y_box_and_offset is "+str(y_box_and_offset)+" in "+str(row["picName"]))
#                                 y_box_and_offset = 6
#                             elif(x_fine < 0 or x_fine>1.00001):
#                                 print("Error, x_fine is "+str(x_fine)+" in "+str(row["picName"]))
#                                 x_fine = 1
#                             elif(y_fine < 0 or y_fine>1.00001):
#                                 print("Error, y_fine is "+str(y_fine)+" in "+str(row["picName"]))
#                                 x_fine = 1
#                             else:
#                                 #Write the x-coords and y-coords in the correct Grid-Cell.
#                                 # x-dimension
#                                 label_tensor[global_index,y_box_and_offset,x_box_and_offset,0]=x_fine
#                                 # y dimension
#                                 label_tensor[global_index,y_box_and_offset,x_box_and_offset,1]=y_fine
#                                 # height
#                                 label_tensor[global_index,y_box_and_offset,x_box_and_offset,2]=float(row["height"])/1280
#                                 # width
#                                 label_tensor[global_index,y_box_and_offset,x_box_and_offset,3]=float(row["width"])/960
#                                 # Probability, that there is a finger
#                                 label_tensor[global_index,y_box_and_offset,x_box_and_offset,4]=1  
#                         img = cv2.imread(origin_path + "../Camera_"+str(camera_nr)+"/WHITE/"+row["picName"])
#                         if(dataSet == "Train"):
#                             self.data[global_index].append(global_picName)
#                             self.data[global_index].append(label_tensor[global_index,:,:,:])
#                         elif(dataSet == "Valid"):
#                             self.data[global_index-100000].append(global_picName)
#                             self.data[global_index-100000].append(label_tensor[global_index,:,:,:])
#     
#     
#                         cv2.imwrite(origin_path+global_picName,img)            
#                         global_index += 1
#                         
#     
#             print("global index until now is: " + str(global_index))
#             print("start shuffling")
#             random.seed(448)
#             np.random.shuffle(self.data)
#             random.seed(543)
#             np.random.shuffle(self.data)
#             
#             print("store lists with pickle")
#             if(dataSet == "Train"):
#                 pickle.dump(self.data, open(origin_path + "traindata.pkl", "wb"))
#             elif(dataSet == "Valid"):
#                 pickle.dump(self.data, open(origin_path + "validdata.pkl", "wb"))
# 
#         
#==============================================================================
        
        
        
              
          
    def get_train_data(self, origin_path = "/media/hhofmann/dgx/data_hhofmann/Data/indexfinger_right/6000_readyTOlearn/trainData/"):
        '''
        Returns 2-Dimensional Array
        1.Dim:  all training-Pictures with its names
        2.Dim:  0 = Picture-Name
                1 = x_coord of the indexfinger
                2 = y_coord of the indexfinger
                3 = probability that there is a Indexfinger in the picture.                
        '''
        return pickle.load(open(origin_path + "traindata.pkl","rb"))
        
    def get_valid_data(self, origin_path = "/media/hhofmann/dgx/data_hhofmann/Data/indexfinger_right/6000_readyTOlearn/trainData/"):
        '''
        Returns 2-Dimensional Array
        1.Dim:  all validation-Pictures with its names
        2.Dim:  0 = Picture-Name
                1 = x_coord of the indexfinger
                2 = y_coord of the indexfinger
                3 = probability that there is a Indexfinger in the picture.                
        '''
        return pickle.load(open(origin_path + "validdata.pkl","rb"))
        

    
    
    
if __name__ == '__main__':

    
#==============================================================================
#     ImageNetData=Dataset_Heinz()
#     ImageNetData.make_lists()
#     del ImageNetData
#==============================================================================

    ReadData=Dataset_Heinz() 

        
    train_pics = ReadData.get_train_data()
    
    print("\n\nTrain-Name = " + train_pics[11][0])
    label_tensor_train = train_pics[11][1]    
    print("Train_x    = " + str(label_tensor_train[:,:,0]))    
    print("Train_y    = " + str(label_tensor_train[:,:,1]))    
    print("Train_h    = " + str(label_tensor_train[:,:,2]))
    print("Train_w    = " + str(label_tensor_train[:,:,3]))
    print("Train_prob = " + str(label_tensor_train[:,:,4]))
    valid_pics = ReadData.get_valid_data()
    print("\n\nValid-Name = " + valid_pics[11][0])
    label_tensor_valid = valid_pics[11][1]    
    print("Valid_x    = " + str(label_tensor_valid[:,:,0]))    
    print("Valid_y    = " + str(label_tensor_valid[:,:,1]))    
    print("Valid_h    = " + str(label_tensor_valid[:,:,2]))
    print("Valid_w    = " + str(label_tensor_valid[:,:,3]))
    print("Valid_prob = " + str(label_tensor_valid[:,:,4]))

