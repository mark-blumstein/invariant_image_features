# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:02:03 2020

@author: blums
"""
import numpy as np
from moment_functions_with_integrator import indexList
import view_images_script as ld

def Ls(A,B):
    numImgs=A.shape[0]
    numKeys=B.shape[0]
    numMoms=A.shape[1]
    scores=np.zeros([numImgs,numKeys])
    for n in range(numImgs):
        a=A[n]
        for m in range(numKeys):
            b=B[m]
            r=a/b
            z=r[:(numMoms-1)]
            w=r[1:(numMoms+1)]
            zNorm=np.abs(z)
            wNorm=np.abs(w)
            inn=np.abs(np.inner(np.conj(z),w))**2
            print(inn)
            print(zNorm,wNorm)
            scores[n,m]=inn/(zNorm*wNorm)
    return scores
            

dMax=6

trainData,trainLbls=ld.load_images('train',200)
testData,testLbls=ld.load_images('test',50)
trainMmts,testMmts=ld.loadMomentData()

#Vector of 0,k moments. Take k0 and then cmp conjugate
idx,n=indexList('cmp',dMax)
v=np.zeros(dMax+1,dtype=int)
for k in range(dMax+1):
    v[k]=int(idx[(k,0)])
    
lTrain=np.conj(trainMmts.cmpMoments[:,v])
lTest=np.conj(testMmts.cmpMoments[:,v])

Keys=lTrain[[0,200,400,600,800,1000,1200,1400,1600,1800]]

scores=Ls(lTest,Keys)
   