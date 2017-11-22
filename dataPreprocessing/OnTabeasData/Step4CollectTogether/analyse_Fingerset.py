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
import cv2
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
        
        
        
    def make_lists(self,origin_path="/media/hhofmann/deeplearning/getfingers_heinz/Data/indexfinger_right/3000_readyTOlearn/trainData/",nrOfCams=4):
        #if path doesn't allready exist, create it.
        if not os.path.exists(origin_path):
            os.makedirs(origin_path + "../trainData")
                
        #get number of Pictures
        nrOfElements = 0
        for camera_nr in range(nrOfCams):
            with open(origin_path + "../Camera_"+str(camera_nr)+"/UV_Bin/fingers.csv") as csvfile_read:
                reader = csv.DictReader(csvfile_read,fieldnames=["picName","x_coord","y_coord","width","height","C","P"])
                for row in reader:
                    nrOfElements+=1
                    
        nrOfTestElements = int(nrOfElements/10)            
        self.data       = [[] for i in range(nrOfElements)]        
        self.validdata  = [[] for i in range(nrOfTestElements)]
        self.traindata  = [[] for i in range(nrOfElements-nrOfTestElements)]
        
        #save picture information in Array and store pictures with new names in new folder.
        global_index = 0
        print("copy pictures will need about 1.5ls minute per 1000 pictures")            
        for camera_nr in range(nrOfCams):
            #read Data from csv-file
            print("start with Camera " + str(camera_nr))
            with open(origin_path + "../Camera_"+str(camera_nr)+"/UV_Bin/fingers.csv") as csvfile_read:
                reader = csv.DictReader(csvfile_read,fieldnames=["picName","x_coord","y_coord","width","height","C","P"])
                #for every picture do
                for row in reader:
                    global_picName = "pic"+str(global_index)+".png"
                    self.data[global_index].append(global_picName)
                    self.data[global_index].append(float(row["x_coord"])/1280)#save and normalize
                    self.data[global_index].append(float(row["y_coord"])/960)#save and normalize
                    self.data[global_index].append(float(row["P"]))
                    img = cv2.imread(origin_path + "../Camera_"+str(camera_nr)+"/WHITE/"+row["picName"])
                    cv2.imwrite(origin_path+global_picName,img)
                    global_index += 1
                    

      
        print("start shuffling")
        random.seed(448)
        np.random.shuffle(self.data)
        random.seed(543)
        np.random.shuffle(self.data)
    
        self.validdata = self.data[0:nrOfTestElements][:]
        self.traindata = self.data[nrOfTestElements+1:nrOfElements-1][:]

      
        print("store lists with pickle")
        pickle.dump(self.data,      open(origin_path + "data.pkl",      "wb"))
        pickle.dump(self.validdata, open(origin_path + "validdata.pkl", "wb"))
        pickle.dump(self.traindata, open(origin_path + "traindata.pkl", "wb"))
              
          
    def get_train_data(self, origin_path = "/media/hhofmann/deeplearning/getfingers_heinz/Data/indexfinger_right/3000_readyTOlearn/trainData/"):
        '''
        Returns 2-Dimensional Array
        1.Dim:  all training-Pictures with its names
        2.Dim:  0 = Picture-Name
                1 = x_coord of the indexfinger
                2 = y_coord of the indexfinger
                3 = probability that there is a Indexfinger in the picture.                
        '''
        return pickle.load(open(origin_path + "traindata.pkl","rb"))
        
    def get_valid_data(self, origin_path = "/media/hhofmann/deeplearning/getfingers_heinz/Data/indexfinger_right/3000_readyTOlearn/trainData/"):
        '''
        Returns 2-Dimensional Array
        1.Dim:  all validation-Pictures with its names
        2.Dim:  0 = Picture-Name
                1 = x_coord of the indexfinger
                2 = y_coord of the indexfinger
                3 = probability that there is a Indexfinger in the picture.                
        '''
        return pickle.load(open(origin_path + "validdata.pkl","rb"))
        
    def get_all_data(self, origin_path = "/media/hhofmann/deeplearning/getfingers_heinz/Data/indexfinger_right/3000_readyTOlearn/trainData/"):
        '''
        Returns 2-Dimensional Array
        1.Dim:  all Pictures with its names
        2.Dim:  0 = Picture-Name
                1 = x_coord of the indexfinger
                2 = y_coord of the indexfinger
                3 = probability that there is a Indexfinger in the picture.                
        '''
        return pickle.load(open(origin_path + "data.pkl", "rb"))

    
    
    
if __name__ == '__main__':

    
#==============================================================================
#     ImageNetData=Dataset_Heinz()
#     ImageNetData.make_lists()
#     del ImageNetData
#==============================================================================

    ReadData=Dataset_Heinz() 

        
    train_pics = ReadData.get_train_data()
    print("Train-Name = " +     train_pics[2][0])
    print("Train_x    = " + str(train_pics[2][1]))    
    print("Train_y    = " + str(train_pics[2][2]))
    print("Train_prob = " + str(train_pics[2][3]))
    valid_pics = ReadData.get_valid_data()
    print("\n\nValid-Name = " + valid_pics[2][0])
    print("Valid_x    = " + str(valid_pics[2][1]))    
    print("Valid_y    = " + str(valid_pics[2][2]))
    print("Valid_prob = " + str(valid_pics[2][3]))
    all_pics= ReadData.get_all_data()
    print("\n\nAll-Name   = " + valid_pics[2][0])
    print("All_x      = " + str(valid_pics[2][1]))    
    print("All_y      = " + str(valid_pics[2][2]))
    print("All_prob   = " + str(valid_pics[2][3]))

