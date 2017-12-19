# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:05:14 2017

@author: hhofmann
"""
from __future__ import print_function
import numpy as np
import cv2

origin_path = "/home/hhofmann/Schreibtisch/Daten/indexfinger_right/3000_readyTOlearn/"
path=origin_path +  "Camera_3/WHITE/pic156.png"
img = np.array(cv2.imread(path))
height, width, channels = img.shape
print("Hoehe   = ", height)
print("Breite  = ", width)
print("channels= ", channels)