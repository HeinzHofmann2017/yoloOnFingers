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
# Commented out, because the library PIL isn't supportet in our docker container on the dgx and this Function is only used on the desktop
#==============================================================================
#==============================================================================
#     def save_pics_as_grayscale(self, origin_path="/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/"):
#         train_picnames = self.get_train_picnames(origin_path=origin_path)        
#         valid_picnames = self.get_valid_picnames(origin_path=origin_path)
#         for i in range(len(train_picnames)):
#             img = Image.open(origin_path+ "../ILSVRC2012_img_train_t12/" + train_picnames[i]).convert('L')
#             old_size=img.size
#             if old_size[0]<244 or old_size[1]<244:
#                 if old_size[0]>= 244:
#                     #Todo: eine Seite zeropadden
#                     new_size=(old_size[0],244)
#                 elif old_size[1]>= 244:
#                     new_size=(244,old_size[1])                    
#                     #Todo: andere Seite zeropadden
#                 else:
#                     new_size=(244,244)
#                     #Todo: beide Seiten Zeropadden
#                 new_img = Image.new("L",new_size)
#                 new_img.paste(img,((new_size[0]-old_size[0])/2,
#                                    (new_size[1]-old_size[1])/2))
#                 new_img.save(origin_path+"../ILSVRC2012_img_train_t12_grayscale/"+train_picnames[i])
#             else:
#                 img.save(origin_path+"../ILSVRC2012_img_train_t12_grayscale/"+train_picnames[i])
#             if(i%100000==0):
#                 mailer.mailto("Trainingdata "+str(i)+" done")
#       
#         for i in range(len(valid_picnames)):
#             img = Image.open(origin_path+ "../ILSVRC2012_img_train_t12/" + valid_picnames[i]).convert('L')
#             old_size=img.size
#             if old_size[0]<244 or old_size[1]<244:
#                 if old_size[0]>= 244:
#                     #Todo: eine Seite zeropadden
#                     new_size=(old_size[0],244)
#                 elif old_size[1]>= 244:
#                     new_size=(244,old_size[1])                    
#                     #Todo: andere Seite zeropadden
#                 else:
#                     new_size=(244,244)
#                     #Todo: beide Seiten Zeropadden
#                 new_img = Image.new("L",new_size)
#                 new_img.paste(img,((new_size[0]-old_size[0])/2,
#                                    (new_size[1]-old_size[1])/2))
#                 new_img.save(origin_path+"../ILSVRC2012_img_train_t12_grayscale/"+valid_picnames[i])
#             else:
#                 img.save(origin_path+"../ILSVRC2012_img_train_t12_grayscale/"+valid_picnames[i])
#             if(i%10000==0):
#                 mailer.mailto("Validationdata "+str(i)+" done")
#         mailer.mailto("finished with creating grayscale pictures")
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
