# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:33:58 2020

@author: markb
"""
import numpy as np
#from sklearn.datasets import load_iris
#from sklearn.linear_model import LogisticRegression
#from scipy import integrate
import view_images_script as ld

Y,lbl=ld.load_images('train',1)
Z=Y[0].reshape(28,28)

#d={'points':[0,0]}

#a, b = -1, 1
#g = lambda x: 0
#h = lambda x: 1
#print(integrate.dblquad(f_real,a,b,g,h,args=(2,1)))