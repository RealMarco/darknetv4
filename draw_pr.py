#!/usr/bin/python3 
# -*- coding: utf-8 -*-
#import cPickle
import _pickle as cPickle
import matplotlib.pyplot as plt
fr = open('testRP/shoe_pr.pkl','rb')  # testRP/box_pr.pkl
inf = cPickle.load(fr)
fr.close()

x=inf['rec']
y=inf['prec']
plt.figure()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR curve of shoe class')
plt.plot(x,y)
plt.show()
print('AP:',inf['ap'])
