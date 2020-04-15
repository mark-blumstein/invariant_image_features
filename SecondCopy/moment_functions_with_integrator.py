# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:53:43 2020

@author: blums
"""

import numpy as np
from scipy.integrate import nquad
from scipy.ndimage.measurements import center_of_mass



    
    
def rmomentForm(x,y,p,q,c,d):
    return np.real(((x-c)+1j*(y-d))**p*((x-c)-1j*(y-d))**q)

def imomentForm(x,y,p,q,c,d):
    return np.imag(((x-c)+1j*(y-d))**p*((x-c)-1j*(y-d))**q)

def indexList(string,dMax):
    if string=="cmp":
        numMoms=1
        for d in range(1,dMax+1):
            for q in range(int(np.floor(d/2)+1)):
                numMoms+=1
        dct={}
        c=0
        for d in range(dMax+1):
           for q in range(int(np.floor(d/2)+1)):
               p=d-q
               dct[(p,q)]=c
               dct[c]=(p,q)
               c+=1
        return dct,numMoms

        
def complexMoments(X,dMax,imgSz=28):
    cent=center_of_mass(X)
    idx,numMoms=indexList("cmp",dMax)
    cmpMoments=np.zeros(numMoms,dtype=np.complex)
    cnt=0
    for deg in range(dMax+1):
        for q in range(int(np.floor(deg/2)+1)):
            p=deg-q
            momSum=0
            for i in range(imgSz):
                for j in range(imgSz):
                    pxlVal=X[i,j]
                    if (pxlVal>1):
                        a=(i-cent[0])-1/2
                        b=(i-cent[0])+1/2
                        c=(j-cent[1])-1/2
                        d=(j-cent[1])+1/2
                        x=nquad(rmomentForm,[[a,b],[c,d]],args=(p,q,cent[0],cent[1]),opts=None)[0]
                        y=nquad(imomentForm,[[a,b],[c,d]],args=(p,q,cent[0],cent[1]),opts=None)[0]
                        z=x+y*(1j)
                        momSum+=z*pxlVal
            cmpMoments[cnt]=momSum
            cnt+=1
    return cmpMoments

class MomentObj(object):
    def __init__(self,Data,dMax):
        self.Data=Data
        self.dMax=dMax
        imgSz=28
        numImgs=Data.shape[0]
        numMoms=1
        for d in range(1,dMax+1):
            for q in range(int(np.floor(d/2)+1)):
                numMoms+=1   
        self.cmpMoments=np.zeros([numImgs,numMoms],dtype=np.complex)
        print("-"*20,"Computing Complex Moments","-"*20)
        for n in range(numImgs):
                X=Data[n].reshape(imgSz,imgSz)
                self.cmpMoments[n]=complexMoments(X,dMax,imgSz)
                print("Finished complex moments of image ",n)
        return 
    pass

