# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:33:50 2020

@author: markb
"""

import numpy as np
from scipy.ndimage.measurements import center_of_mass

def numMoms(dMax):
    numGeo=int((dMax+1)*(dMax+2)/2)
    numCmp=0
    for d in range(dMax+1):
       for q in range(int(np.floor(d/2)+1)):
           numCmp+=1
    return numGeo, numCmp


def geoMoments(X,dMax,numPts=100):
    #X is image in square grid of size numPts by numPts
    Box=np.ones((numPts,numPts)) #compute this outside the function
    ImgMesh=np.kron(X,Box)
    sz=ImgMesh.shape[0]
    cent=(1/sz)*np.array(center_of_mass(ImgMesh)) 
    
    #0th moment
    dA=1/((sz-1)**2)
    imPower=dA*ImgMesh.sum()
    
    #create coordinate array scaled to be in unit square and centralized
    #add extra axis-to be used for broadcasting when computing moments
    coordX=np.linspace(-1*cent[0],-1*cent[0]+1,sz).reshape(sz,1)
    coordY=np.linspace(-1*cent[1],-1*cent[1]+1,sz).reshape(1,sz)
    
    
    
    #Iterate over (p,q) plane to compute (p,q)th moment
        
    #Initialize array to store complex moments
    numGeo,numCmp=numMoms(dMax)
    geoMoments=np.zeros(numGeo)
    
    #Build Up and Right Array
    z=np.zeros(sz)
    UpArr=coordY+z.reshape(sz,1)
    RtArr=coordX+z.reshape(1,sz)
    #do first column minus origin (q>0) outside of main loop
    CurrArr=np.copy(coordY)
    ctr=0
    for q in range(1,dMax+1):
        geoMoments[ctr]=(CurrArr*ImgMesh).sum()
        ctr+=1
        CurrArr=CurrArr*coordY
        
    CurrArr=coordX+(1j)*coordY
    NextArr=UpArr*CurrArr #Move up i.e. q++1
    ctr=0    
    for p in range(1,dMax+1):
        cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
        CurrArr=CurrArr*RtArr #move right p++1
        ctr+=1
        
    #Main loop
    for q in range(1,qMax+1):
        CurrArr=np.copy(NextArr)
        NextArr=CurrArr*UpArr
        for p in range(q,dMax-q+1):
            cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
            CurrArr=CurrArr*RtArr
            ctr+=1
    
    #scale by area form for Riemann sum
    
    geoMmts=dA*geoMoments
    return geoMoments


def computeAll(Data,dMax,numPts=100):
    print("*"*15,"Computing Moments","*"*15)
    numImgs=Data.shape[0]
    Box=np.ones((numPts,numPts))
    numGeo,numCmp=numMoms(dMax)
    cmpMoms=np.zeros((numImgs,numCmp),dtype=np.complex)
    for n in range(numImgs):
        X=Data[n].reshape(28,28)
        cmpMoms[n]=cmpMoments(X,dMax,Box,numPts)
        print("\t","Finished Complex Moments of image number ",n)
    return cmpMoms

def cmpMoments(X,dMax,Box,numPts=100):
    #X is image in square grid of size numPts by numPts
    #Box=np.ones((numPts,numPts)) #compute this outside the function
    ImgMesh=np.kron(X,Box)
    sz=ImgMesh.shape[0]
    cent=(1/sz)*np.array(center_of_mass(ImgMesh)) 
    
    #create coordinate array scaled to be in unit square and centralized
    #add extra axis-to be used for broadcasting when computing moments
    coordX=np.linspace(-1*cent[0],-1*cent[0]+1,sz).reshape(sz,1)
    coordY=np.linspace(-1*cent[1],-1*cent[1]+1,sz).reshape(1,sz)
    
    #Iterate over (p,q) plane to compute (p,q)th moment

    qMax=int(np.floor(dMax/2))
    RtArr=coordX+(1j)*coordY #multiply by this array to move right i.e. p+1
    UpArr=np.conj(RtArr) #multiply by this array to move up  i.e. q++1
    
    #initialize array to store complex moments
    numGeo,numCmp=numMoms(dMax)
    cmpMoments=np.zeros(numCmp,dtype=np.complex)
    
    #Do first row q=0, p=1..dMax outside of main loop
    #avoids checking for p=q=0 case
    CurrArr=coordX+(1j)*coordY
    NextArr=UpArr*CurrArr #Move up i.e. q++1
    ctr=0    
    for p in range(1,dMax+1):
        cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
        CurrArr=CurrArr*RtArr #move right p++1
        ctr+=1
        
    #Main loop
    for q in range(1,qMax+1):
        CurrArr[:]=NextArr
        NextArr=CurrArr*UpArr
        for p in range(q,dMax-q+1):
            cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
            CurrArr=CurrArr*RtArr
            ctr+=1
    
    #scale by area form for Riemann sum
    dA=1/((sz-1)**2)
    cmpMoments*=dA
    return cmpMoments

    
def idxCmp3(dMax):
    ctr=0
    idx={}
    qMax=int(np.floor(dMax/2))
    for p in range(1,dMax+1):
        idx[ctr]=(p,0)
        idx[(p,0)]=ctr
        ctr+=1

    #Main loop
    for q in range(1,qMax+1):
        for p in range(q,dMax-q+1):
            idx[ctr]=(p,q)
            idx[(p,q)]=ctr
            ctr+=1
    return idx

def viewMoment(X,p,q,Box,numPts=100):
    ImgMesh=np.kron(X,Box)
    sz=ImgMesh.shape[0]
    cent=(1/sz)*np.array(center_of_mass(ImgMesh))
    coordX=np.linspace(-1*cent[0],-1*cent[0]+1,sz).reshape(sz,1)
    coordY=np.linspace(-1*cent[1],-1*cent[1]+1,sz).reshape(1,sz)    
    
    RtArr=coordX+(1j)*coordY
    UpArr=np.conj(RtArr)
    
    Arr=((RtArr)**p)*((UpArr)**q)
    plt.imshow(Arr.real)
    plt.figure()
    plt.imshow(Arr.imag)
        
    
            
    