# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:20:05 2017

@author: hhofmann
"""

def main():
    # my code here
    weights=0
    weights += (7*7*64)#1
    weights += (3*3*192)#3
    weights += (1*1*128)#5
    weights += (3*3*256)#6
    weights += (1*1*256)#7
    weights += (3*3*512)#8
    weights += (1*1*256)#10
    weights += (3*3*512)#11
    weights += (1*1*256)#12
    weights += (3*3*512)#13
    weights += (1*1*256)#14
    weights += (3*3*512)#15
    weights += (1*1*256)#16
    weights += (3*3*512)#17
    weights += (1*1*512)#18
    weights += (3*3*1024)#19
    weights += (1*1*512)#21
    weights += (3*3*1024)#22
    weights += (1*1*512)#23
    weights += (3*3*1024)#24
    weights += (3*3*1024)#25
    weights += (3*3*1024)#26
    weights += (3*3*1024)#27
    weights += (3*3*1024)#28
    weights += (3*3*1024)#30
    weights += (7*7*1024*4096)#31
    weights += (4096*7*7*6)#32
    print("nr of weights = "+str(weights))
    nr_of_bytes = 4*weights
    nr_of_GBytes = float(nr_of_bytes) / 1000000000
    print("nr of Weights in GBytes(with float32, 4*Nr_of_weights/1G) = "+str(nr_of_GBytes))
    
    nr_of_pixels = 0
    nr_of_pixels += (960*1280*1)#input
    nr_of_pixels += (480*640*64)#1
    nr_of_pixels += (240*320*64)#2    
    nr_of_pixels += (240*240*64)#3  
    nr_of_pixels += (120*160*192)#4
    nr_of_pixels += (120*160*128)#5
    nr_of_pixels += (120*160*256)#6
    nr_of_pixels += (120*160*256)#7
    nr_of_pixels += (120*160*512)#8
    nr_of_pixels += (60*80*512)#9
    nr_of_pixels += (60*80*256)#10
    nr_of_pixels += (60*80*512)#11
    nr_of_pixels += (60*80*256)#12
    nr_of_pixels += (60*80*512)#13
    nr_of_pixels += (60*80*256)#14
    nr_of_pixels += (60*80*512)#15
    nr_of_pixels += (60*80*256)#16
    nr_of_pixels += (60*80*512)#17
    nr_of_pixels += (60*80*512)#18
    nr_of_pixels += (60*80*1024)#19
    nr_of_pixels += (30*40*1024)#20
    nr_of_pixels += (30*40*512)#21
    nr_of_pixels += (30*40*1024)#22
    nr_of_pixels += (30*40*512)#23
    nr_of_pixels += (30*40*1024)#24
    nr_of_pixels += (30*40*1024)#25
    nr_of_pixels += (15*20*1024)#26
    nr_of_pixels += (15*20*1024)#27
    nr_of_pixels += (15*20*1024)#28
    nr_of_pixels += (21*21*1024)#29 spezial, je nachdem
    nr_of_pixels += (7*7*1024)#30 ab hier fix egal was der input macht.
    nr_of_pixels += (4096)#31
    nr_of_pixels += (7*7*30)#32
    
    print("Nr of saved Pixels with full size images = " + str(nr_of_pixels))
    nr_of_Bytes = 4*nr_of_pixels
    nr_of_GBytes = float(nr_of_Bytes)/1000000000
    print("Nr of GBytes with full size images (with float32, 4*Nr_of_weights/1G)= " + str(nr_of_GBytes))
    
if __name__ == "__main__":
    main()