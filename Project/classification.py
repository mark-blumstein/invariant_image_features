# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:02:03 2020

@author: blums
"""
import numpy as np
from moment_functions import idxCmp3
import view_images_script as ld

def coherenceScore(A,B):
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
            zNorm=np.linalg.norm(z)**2
            wNorm=np.linalg.norm(w)**2
            inn=np.abs(np.inner(np.conj(z),w))**2
            scores[n,m]=inn/(zNorm*wNorm)
    
    preds=np.zeros(numImgs)
    ctr=0
    for score in scores:
        preds[ctr]=score.argmax()
        ctr+=1
    return scores,preds
            

dMax=10
#trMmts,tstMmts=ld.loadMomentData()

#Vector of 0,k moments. Take k0 and then cmp conjugate
idx=idxCmp3(dMax)
v=np.zeros(dMax,dtype=int)
for k in range(1,dMax):
    print(k,idx[k,0])
    v[k-1]=int(idx[(k,0)])
    
lTrain=np.conj(trCmp[:,v])
lTest=np.conj(trCmp[:,v])

Keys=lTrain[[0,200,400,600,800,1000,1200,1400,1600,1800]]

scores,preds=coherenceScore(lTest,Keys)


   