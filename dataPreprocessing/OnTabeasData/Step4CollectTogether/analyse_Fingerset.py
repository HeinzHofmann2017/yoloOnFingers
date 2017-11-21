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
        self.data = [[]]
        self.traindata = [[]]
        self.validdata = [[]]
        
        self.all_picture_names = []
        self.train_picnames = []
        self.train_labels = []
        self.valid_picnames = []
        self.valid_labels = []
        
        
        
    def make_lists(self,origin_path="/media/hhofmann/deeplearning/getfingers_heinz/Data/indexfinger_right/3000_readyTOlearn/trainData",nrOfCams=4):
        #if path doesn't allready exist, create it.
        if not os.path.exists(origin_path):
            os.makedirs(origin_path + "../trainData")
        
        global_index = 0
        for camera_nr in range(nrOfCams):
            #read Data from csv-file
            with open(origin_path + "/../Camera_"+str(camera_nr)+"/UV_Bin/fingers.csv") as csvfile_read:
                reader = csv.DictReader(csvfile_read,fieldnames=["picName","x_coord","y_coord","width","height","C","P"])
                #for every picture do
                for row in reader:
                    print("round "+str(global_index))
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
    
    
        nrOfElements = len(self.data[:][None])
        nrOfTestElements = int(nrOfElements/10)
        self.validdata = self.data[0:nrOfTestElements][:]
        self.traindata = self.data[nrOfTestElements+1:nrOfElements-1][:]

      
        print("store lists with pickle")
        pickle.dump(self.validdata,open(origin_path + "validdata.pkl","wb"))
        pickle.dump(self.traindata,open(origin_path + "traindata.pkl","wb"))
              
          
    def get_train_picnames(self, origin_path = "/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
        return pickle.load(open(origin_path + "train_picnames.pkl","rb"))
    
    def get_train_labels(self, origin_path = "/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
        return pickle.load(open(origin_path + "train_labels.pkl", "rb"))
        
    def get_valid_picnames(self, origin_path = "/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
        return pickle.load(open(origin_path + "valid_picnames.pkl","rb"))
    
    def get_valid_labels(self, origin_path = "/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
        return pickle.load(open(origin_path + "valid_labels.pkl", "rb"))
        
    def get_all_picture_names(self, origin_path = "/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
        return pickle.load(open(origin_path + "all_picture_names.pkl", "rb"))
#==============================================================================
# should be Commented out, because the library PIL isn't supportet in our docker container on the dgx and this Function is only used on the desktop
#==============================================================================
    def save_pics_as_grayscale(self, origin_path="/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
        '''
        1. gets Pictures from origin-directory(training and label-pictures)
        2. converts Pictures to grayscale
        3. resize Images so the smaller of both dimensions is 300
        4. Stores Pictures to target-directory
        '''
        train_picnames = self.get_train_picnames(origin_path=origin_path)        
        valid_picnames = self.get_valid_picnames(origin_path=origin_path)
        target_size = 300
#==============================================================================
#         for i in range(len(train_picnames)):
#             img = Image.open(origin_path+ "../ILSVRC2012_img_train_t12/" + train_picnames[i]).convert('L')#Load and convert to grayscale
#             old_dimension_sizes=img.size
#             smallest_size = min(old_dimension_sizes[0],old_dimension_sizes[1])#get the size of the smaller dimension, so we can adapt the size of this dimension to the target-size.
#             ratio = float(target_size)/float(smallest_size)
#             new_dimension_sizes = int(old_dimension_sizes[0]*ratio),int(old_dimension_sizes[1]*ratio)
#             img = img.resize(new_dimension_sizes,Image.ANTIALIAS)   
#             img.save(origin_path+"../ILSVRC2012_img_train_t12_grayscale/"+train_picnames[i])
#         mailer.mailto("finished with creating grayscale on train pictures")
#         for i in range(len(valid_picnames)):
#             img = Image.open(origin_path+ "../ILSVRC2012_img_train_t12/" + valid_picnames[i]).convert('L')#Load and convert to grayscale
#             old_dimension_sizes=img.size
#             smallest_size = min(old_dimension_sizes[0],old_dimension_sizes[1])#get the size of the smaller dimension, so we can adapt the size of this dimension to the target-size.
#             ratio = float(target_size)/float(smallest_size)
#             new_dimension_sizes = int(old_dimension_sizes[0]*ratio),int(old_dimension_sizes[1]*ratio)
#             img = img.resize(new_dimension_sizes,Image.ANTIALIAS)   
#             img.save(origin_path+"../ILSVRC2012_img_train_t12_grayscale/"+valid_picnames[i])
#         mailer.mailto("finished with creating grayscale on validation pictures")
#==============================================================================
         
    
    
    
if __name__ == '__main__':

    
    ImageNetData=Dataset_Heinz()
    ImageNetData.make_lists()
    del ImageNetData

#==============================================================================
#     ReadData=Dataset_Heinz() 
# 
#     ReadData.save_pics_as_grayscale()    
#     
#     train_picnames = ReadData.get_train_picnames() 
#     train_labels = ReadData.get_train_labels()
#     valid_picnames = ReadData.get_valid_picnames() 
#     valid_labels = ReadData.get_valid_labels()
#     all_picture_names= ReadData.get_all_picture_names()
#     print("train_picname = " + train_picnames[1])
#     print("train_labels = " + str(train_labels[1]))
#     print("does it match?: " + all_picture_names[train_labels[1]])
#     print("valid_picname = " + valid_picnames[1])
#     print("valid_labels = " + str(valid_labels[1]))
#     print("does it match?: " + all_picture_names[valid_labels[1]])
#==============================================================================
