#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 22:28:37 2021

@author: marco
"""

import cv2
import os
import numpy as np 

# comp4_det_test_shoe.txt info: confidence xmin, ymin, xmax, ymax

with open('results/comp4_det_test_shoe.txt', 'r') as f:
    lines = f.readlines()

splitlines = [x.strip().split(' ') for x in lines]
image_names = [x[0]+'.jpg' for x in splitlines]
confidence = np.array([float(x[1]) for x in splitlines])
BB = np.array([[float(z) for z in x[2:]] for x in splitlines])   # xmin, ymin, xmax, ymax

with open('2007_ShoesStatesTest.txt','r') as f2:
    lines2=f2.readlines()

splitlines2 = [x2.strip().split('/') for x2 in lines2]
image_full_list = [x2[-1] for x2 in splitlines2]
    
## sort by confidence
#sorted_ind = np.argsort(-confidence) # sort (from large to small) and return the index
## sorted_scores = np.sort(-confidence)
#BB = BB[sorted_ind, :]
#image_names = [image_names[x] for x in sorted_ind]

img_dir = '/home/marco/catkin_workspace/src/darknet_ros/darknet/RPdevkit/RP2007/JPEGImagesTest/'
dest_dir= '/home/marco/catkin_workspace/src/darknet_ros/darknet/RPdevkit/RP2007/CroppedShoesStatesTesting/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
cropped_img_ids = []     # list of cropped images ids
cropped_img_list=[]      # list of cropped images names
cropped_confidence = []  # confidence list for cropped images

# Non-maximum suppression
for i in range(len(image_names)):
    if image_names[i] not in cropped_img_list:
        cropped_img_list.append(image_names[i]) # name list
        cropped_img_ids.append(i) # index list
        cropped_confidence.append(confidence[i])  
    else: # image_names[i] in cropped_img_list
        if confidence[i]>confidence[cropped_img_ids[-1]]:
            cropped_img_ids[-1]=i # replace the smaller one
            cropped_img_list[-1] = image_names[i] #
            cropped_confidence[-1]=confidence[i]
       
for ids in cropped_img_ids:
    img =  cv2.imread(img_dir+image_names[ids])
    [xmin, ymin, xmax, ymax] = BB[ids]
    # 0.9-1.1 to enlarge the bbx, round() would be more accurate than int()
    cropped_img = img[int(ymin*0.98):int(ymax*1.02), int(xmin*0.98):int(xmax*1.02)] # [ymin:ymax, xmin:xmax]
#    cv2.imshow('cropped_img', cropped_img)
    cv2.imwrite(dest_dir+image_names[ids], cropped_img)
    
sorted_cropped_confidence = cropped_confidence
sorted_cropped_confidence.sort()
print('10 minimal confidence after non-maximum suppression/threshold filtering:')
print(sorted_cropped_confidence[0:10])


print('No objects were detected in these images below:')      
for ids2 in image_full_list:
    if ids2 not in image_names:
        print(ids2)
