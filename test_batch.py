#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:51:21 2022

@author: marco
"""

import os
import sys
import shutil

arg = sys.argv

txt_file = arg[1]
with open(txt_file, mode="r") as f:
    data = f.readlines()
    f.close()
data = [i.split("\n")[0] for i in data]
# print(data)

i = 0
while True:
    if i < len(data):
        new_name = "results/{}.jpg".format(data[i].split("/")[-1][:-4])
        try:
            shutil.move("predictions.jpg", new_name)
            print(i, end="\r")
            i += 1
        except Exception as e:
            pass

