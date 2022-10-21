# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:55:52 2022
@author: Marco
Project: YOLO
"""

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
#import pickle
import os
#from os import listdir, getcwd
#from os.path import join
#import random
import shutil

# the original characters 'VOC' and 'voc' was replaced by 'RP'
# Method 1: Predefine classes by hand
# classes=["table_tennis","paper","shoe"]
# Method 2: Predefine classes according to predefined_classes.txt used in Labelimg
classes_file = open('predefined_classes.txt')
classes =  classes_file.read().splitlines() # & classes_file.readlines() read without '\n'


def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def convert(size, box):  # normalize and convert to YOLO inputting format
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):   # convert .xml annotations to .txt ones
    # in_file_path = os.path.join(work_sapce_dir, 'RPdevkit/RP2007/Annotations/%s.xml' %image_id)
    in_file_path = 'RPdevkit/RP2007/Annotations/%s.xml' %image_id
    out_file = open('RPdevkit/RP2007/labels/%s.txt' %image_id, 'w')  # create an empty file 
    if os.path.isfile(in_file_path):  # with .xml annotations, convert and write the content
        in_file =  open('RPdevkit/RP2007/Annotations/%s.xml' %image_id) # use imagename as imageid
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            in_file.close()
    out_file.close()

# make empty folders and files
# wd = os.getcwd()
wd = os.getcwd()
work_sapce_dir = os.path.join(wd, "RPdevkit/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
work_sapce_dir = os.path.join(work_sapce_dir, "RP2007/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
        os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")  # training/data set
if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
clear_hidden_files(image_dir)
test_image_dir = os.path.join(work_sapce_dir, "JPEGImagesTest/")  # testing set, for YOLOv4 AlexyAB's framework
if not os.path.isdir(test_image_dir):
        os.mkdir(test_image_dir)
clear_hidden_files(test_image_dir)
RP_file_dir = os.path.join(work_sapce_dir, "ImageSets/")
if not os.path.isdir(RP_file_dir):
        os.mkdir(RP_file_dir)
RP_file_dir = os.path.join(RP_file_dir, "Main/")
if not os.path.isdir(RP_file_dir):
        os.mkdir(RP_file_dir)

train_file = open(os.path.join(wd, "2007_train.txt"), 'w')
test_file = open(os.path.join(wd, "2007_test.txt"), 'w')
train_file.close()
test_file.close()
RP_train_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/train.txt"), 'w')
RP_test_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/test.txt"), 'w')
RP_train_file.close()
RP_test_file.close()
if not os.path.exists('RPdevkit/RP2007/labels'):
    os.makedirs('RPdevkit/RP2007/labels')
else:
    shutil.rmtree('RPdevkit/RP2007/labels')
    os.makedirs('RPdevkit/RP2007/labels')
    
train_file = open(os.path.join(wd, "2007_train.txt"), 'a')  # record absolute paths of images with extension
test_file = open(os.path.join(wd, "2007_test.txt"), 'a')
RP_train_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/train.txt"), 'a') # record image names without extension
RP_test_file = open(os.path.join(work_sapce_dir, "ImageSets/Main/test.txt"), 'a')
list = os.listdir(image_dir) # list the names of images/training images
test_list = os.listdir(test_image_dir) # list testing images


# Images for training and testing is stored in different folders (JPEGImages/ and JPEGImagesTest/) for YOLOv4 AlexyAB's framework
# If the dataset was devided into training set and testing set previously, get the name/path list
## training images list
for i in range(0,len(list)):
    path = os.path.join(image_dir,list[i])
    if os.path.isfile(path):
        image_path = image_dir + list[i]                                 # absolute paths with extension
        RP_path = list[i]                                                # image names without extension
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))  # split extensions
        (RP_nameWithoutExtention, RP_extention) = os.path.splitext(os.path.basename(RP_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        
    #if os.path.exists(annotation_path):
        train_file.write(image_path + '\n')
        RP_train_file.write(RP_nameWithoutExtention + '\n')
        convert_annotation(nameWithoutExtention)  # convert .xml into .txt

## background images list


## testing images list
for i in range(0,len(test_list)):
    test_path = os.path.join(test_image_dir,test_list[i])
    if os.path.isfile(test_path):
        test_image_path = test_image_dir + test_list[i]
        test_RP_path = test_list[i]
        (test_nameWithoutExtention, test_extention) = os.path.splitext(os.path.basename(test_image_path))
        (test_RP_nameWithoutExtention, test_RP_extention) = os.path.splitext(os.path.basename(test_RP_path))
        test_annotation_name = test_nameWithoutExtention + '.xml'
        test_annotation_path = os.path.join(annotation_dir, test_annotation_name)
        
    if os.path.exists(test_annotation_path):
        test_file.write(test_image_path + '\n')
        RP_test_file.write(test_RP_nameWithoutExtention + '\n')
        convert_annotation(test_nameWithoutExtention)

# Images for training and testing is stored in the same folder (JPEGImages/) for YOLOv3 framework
##else if the dataset wasn't devided at first, divide the dataset by train-test ratio (probo)
#probo = random.randint(1, 100)
#print("Probobility: %d" % probo)
#for i in range(0,len(list)):
#    path = os.path.join(image_dir,list[i])
#    if os.path.isfile(path):
#        image_path = image_dir + list[i]
#        RP_path = list[i]
#        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
#        (RP_nameWithoutExtention, RP_extention) = os.path.splitext(os.path.basename(RP_path))
#        annotation_name = nameWithoutExtention + '.xml'
#        annotation_path = os.path.join(annotation_dir, annotation_name)
#    probo = random.randint(1, 100)
#    print("Probobility: %d" % probo)
#    if(probo < 75):
#        if os.path.exists(annotation_path):
#            train_file.write(image_path + '\n')
#            RP_train_file.write(RP_nameWithoutExtention + '\n')
#            convert_annotation(nameWithoutExtention)
#    else:
#        if os.path.exists(annotation_path):
#            test_file.write(image_path + '\n')
#            RP_test_file.write(RP_nameWithoutExtention + '\n')
#            convert_annotation(nameWithoutExtention)
            
# Closing the files is always a good habit          
train_file.close()
test_file.close()
RP_train_file.close()
RP_test_file.close()
