# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:33:58 2020

@author: markb
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from scipy import integrate

class stupidObj(object):
    def __init__(self,data,deg):
        self.data=data
        self.d=deg
        
        
def stupid(x,y,z=4):
    print(x,y,z)

def f_real(x,y,p,q,c,d):
    return np.real(((x-c)+1j*(y-d))**p*((x-c)-1j*(y-d))**q)

def f_imag(x,y,p,q,c,d):
    return np.imag(((x-c)+1j*(y-d))**p*((x-c)-1j*(y-d))**q)
a,c=1,1
b,d=2,2
p=q=3
cent=[1.5,1.5]
x=integrate.nquad(f_real,[[a,b],[c,d]],args=(p,q,cent[0],cent[1]),opts=None)[0]
y=integrate.nquad(f_imag,[[a,b],[c,d]],args=(p,q,cent[0],cent[1]),opts=None)[0]
print(x,y)

#d={'points':[0,0]}

#a, b = -1, 1
#g = lambda x: 0
#h = lambda x: 1
#print(integrate.dblquad(f_real,a,b,g,h,args=(2,1)))