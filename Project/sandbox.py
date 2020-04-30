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
import itertools
Y,lbl=ld.load_images('train',2)
Z=Y[0].reshape(28,28)
W=Y[1].reshape(28,28)
a=np.zeros(4)
b=np.ones(4)

b*=a
print(b)

lst=['a','b','c','d','e','f']
lst2=list(map(lambda x: x+'z',lst))

x=np.arange(10)
y=list(map(lambda x:x+2,x))

x=np.zeros((100,100))
y=np.ones(100)

def func(a,n):
    a=n*
for n in range(100):
    x[n]=n*y
    