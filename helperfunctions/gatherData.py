# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 09:59:16 2018

@author: hhofmann
"""
import os
from os import listdir
import cv2

make4500=False

def gatherData(origin_path="/home/hhofmann/Schreibtisch/Data/", nrOfCams=4):
    print("start")
    if(make4500==True):
        if not os.path.exists(origin_path+"4500/Camera_0/UV"):
            os.makedirs(origin_path+"4500/Camera_0/UV")
        if not os.path.exists(origin_path+"4500/Camera_1/UV"):
            os.makedirs(origin_path+"4500/Camera_1/UV")
        if not os.path.exists(origin_path+"4500/Camera_2/UV"):
            os.makedirs(origin_path+"4500/Camera_2/UV")
        if not os.path.exists(origin_path+"4500/Camera_3/UV"):
            os.makedirs(origin_path+"4500/Camera_3/UV")
        if not os.path.exists(origin_path+"4500/Camera_0/WHITE"):
            os.makedirs(origin_path+"4500/Camera_0/WHITE")
        if not os.path.exists(origin_path+"4500/Camera_1/WHITE"):
            os.makedirs(origin_path+"4500/Camera_1/WHITE")
        if not os.path.exists(origin_path+"4500/Camera_2/WHITE"):
            os.makedirs(origin_path+"4500/Camera_2/WHITE")
        if not os.path.exists(origin_path+"4500/Camera_3/WHITE"):
            os.makedirs(origin_path+"4500/Camera_3/WHITE")
        for cam_nr in range(nrOfCams):
            print("cam nr "+str(cam_nr))
            global_index = 0
            for pic in listdir(origin_path+"6000/Camera_"+str(cam_nr)+"/UV/"):
                if(global_index<4500):                
                    img = cv2.imread(origin_path+"6000/Camera_"+str(cam_nr)+"/UV/"+pic,0)
                    cv2.imwrite(origin_path+"4500/Camera_"+str(cam_nr)+"/UV/pic"+str(global_index)+".png",img)
                    img = cv2.imread(origin_path+"6000/Camera_"+str(cam_nr)+"/WHITE/"+pic,0)
                    cv2.imwrite(origin_path+"4500/Camera_"+str(cam_nr)+"/WHITE/pic"+str(global_index)+".png",img)                
                    global_index+=1
                
    else:
        if not os.path.exists(origin_path+"9000/Camera_0/UV"):
            os.makedirs(origin_path+"9000/Camera_0/UV")
        if not os.path.exists(origin_path+"9000/Camera_1/UV"):
            os.makedirs(origin_path+"9000/Camera_1/UV")
        if not os.path.exists(origin_path+"9000/Camera_2/UV"):
            os.makedirs(origin_path+"9000/Camera_2/UV")
        if not os.path.exists(origin_path+"9000/Camera_3/UV"):
            os.makedirs(origin_path+"9000/Camera_3/UV")
        if not os.path.exists(origin_path+"9000/Camera_0/WHITE"):
            os.makedirs(origin_path+"9000/Camera_0/WHITE")
        if not os.path.exists(origin_path+"9000/Camera_1/WHITE"):
            os.makedirs(origin_path+"9000/Camera_1/WHITE")
        if not os.path.exists(origin_path+"9000/Camera_2/WHITE"):
            os.makedirs(origin_path+"9000/Camera_2/WHITE")
        if not os.path.exists(origin_path+"9000/Camera_3/WHITE"):
            os.makedirs(origin_path+"9000/Camera_3/WHITE")  
        for cam_nr in range(nrOfCams):
            print("cam nr "+str(cam_nr))
            global_index = 0
            for pic in listdir(origin_path+"6000/Camera_"+str(cam_nr)+"/UV/"):               
                img = cv2.imread(origin_path+"6000/Camera_"+str(cam_nr)+"/UV/"+pic,0)
                cv2.imwrite(origin_path+"9000/Camera_"+str(cam_nr)+"/UV/pic"+str(global_index)+".png",img)
                img = cv2.imread(origin_path+"6000/Camera_"+str(cam_nr)+"/WHITE/"+pic,0)
                cv2.imwrite(origin_path+"9000/Camera_"+str(cam_nr)+"/WHITE/pic"+str(global_index)+".png",img)                
                global_index+=1
            for pic in listdir(origin_path+"3000/Camera_"+str(cam_nr)+"/UV/"):
                img = cv2.imread(origin_path+"3000/Camera_"+str(cam_nr)+"/UV/"+pic,0)
                cv2.imwrite(origin_path+"9000/Camera_"+str(cam_nr)+"/UV/pic"+str(global_index)+".png",img)
                img = cv2.imread(origin_path+"3000/Camera_"+str(cam_nr)+"/WHITE/"+pic,0)
                cv2.imwrite(origin_path+"9000/Camera_"+str(cam_nr)+"/WHITE/pic"+str(global_index)+".png",img)            
                global_index+=1
if __name__ == '__main__':
    gatherData()