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
this_folder =  os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_folder+ '/../../helperfunctions/')
import mailer


class Dataset_Heinz(object):
    def __init__(self):
        self.picnames = []
        self.labels = []
        self.all_picture_names = []
        self.train_picnames = []
        self.train_labels = []
        self.valid_picnames = []
        self.valid_labels = []
        
        
        
    def make_lists(self,origin_path="/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
          print("started read pictures (duration=2min)")
          self.picnames = os.listdir(origin_path + "../ILSVRC2012_img_train_t12")

          print(self.picnames[1])
                  
          index_to_delete=0
          for j in range(len(self.picnames)):
              if self.picnames[j].endswith(".JPEG"):
                  there_was_a_match = 0
                  #print(j[0:9])
                  for k in range(len(self.all_picture_names)-1,-1,-1):#Diese for-schleife läuft rückwärts(ist ein bisschen schneller)
                      if self.picnames[j][0:9] == self.all_picture_names[k]:
                          there_was_a_match = 1
                  if there_was_a_match == 0:
                      self.all_picture_names.append(str(self.picnames[j][0:9]))
                      #print(str(len(self.all_picture_names)))
              else:
                  index_to_delete = j
          self.picnames.pop(index_to_delete)#delete file from list, which isn't a picture
          
          print("start shuffling")
          random.seed(448)
          random.shuffle(self.picnames)
          random.seed(543)
          random.shuffle(self.picnames)


          print("make index-list(duration= 2,5 min)")    
          for j in range(len(self.picnames)):
                for k in range(len(self.all_picture_names)):
                    if self.picnames[j][0:9] == self.all_picture_names[k]:
                        self.labels.append(k)
          self.train_picnames = self.picnames[0:1200000]
          self.train_labels = self.labels[0:1200000]
          self.valid_picnames = self.picnames[1200002:1281166]
          self.valid_labels = self.labels[1200002:1281166]
          
          print("store lists with pickle")
          pickle.dump(self.all_picture_names,open(origin_path + "all_picture_names.pkl","wb"))
          pickle.dump(self.train_picnames,open(origin_path + "train_picnames.pkl","wb"))
          pickle.dump(self.train_labels,open(origin_path + "train_labels.pkl","wb"))
          pickle.dump(self.valid_picnames,open(origin_path + "valid_picnames.pkl","wb"))
          pickle.dump(self.valid_labels,open(origin_path + "valid_labels.pkl","wb"))
              
          
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

    
#==============================================================================
#     ImageNetData=Dataset_Heinz()
#     ImageNetData.make_lists()
#     del ImageNetData
#==============================================================================

    ReadData=Dataset_Heinz() 

    ReadData.save_pics_as_grayscale()    
    
    train_picnames = ReadData.get_train_picnames() 
    train_labels = ReadData.get_train_labels()
    valid_picnames = ReadData.get_valid_picnames() 
    valid_labels = ReadData.get_valid_labels()
    all_picture_names= ReadData.get_all_picture_names()
    print("train_picname = " + train_picnames[1])
    print("train_labels = " + str(train_labels[1]))
    print("does it match?: " + all_picture_names[train_labels[1]])
    print("valid_picname = " + valid_picnames[1])
    print("valid_labels = " + str(valid_labels[1]))
    print("does it match?: " + all_picture_names[valid_labels[1]])
