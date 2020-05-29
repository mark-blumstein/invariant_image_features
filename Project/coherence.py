# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:02:03 2020

@author: blums
"""
import numpy as np
#from moment_functions import idxCmp3
import view_images_script as ld
#from moment_functions import cmp_moment
import moment_functions as mf
#from moment_functions import rotate
import matplotlib.pyplot as plt 
from scipy.ndimage import rotate


def coherence(A,B):
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

deg=ld.deg
num_train=ld.num_train #of each digit
num_test=ld.num_test

tr,tst=ld.loadMomentData()
#lost moment 10,0 need to fix in moment_functions.py
#tr=tr[:,0:35]
#tst=tst[:,0:35]

#reshape array to include class label
tr.clgeo=tr.geo.reshape((10,num_train,tr.geo.shape[1]))
tr.clcmp=tr.cmp.reshape((10,num_train,tr.cmp.shape[1]))

tst.clgeo=tst.geo.reshape((10,num_test,tst.geo.shape[1]))
tst.clcmp=tst.cmp.reshape((10,num_test,tst.cmp.shape[1]))

#Test the coherence classifier
X=tr.data[0].reshape(28,28)
Y=rotate(X,30)
plt.figure()
plt.imshow(X)
plt.figure()
plt.imshow(Y)

xmts=tr.cmp[0][2:(deg+1)]

ymts=[]
for k in range(2,deg+1):
    mt=mf.compute_cmp(Y,k,0)
    ymts.append(mt)
ymts=np.array(ymts)  


ymts=ymts[np.newaxis,:]
xmts=xmts[np.newaxis,:]

sc,pr=coherence(xmts,ymts)






#coherence classification

#Extract vector of p,0 moments 1<= p <= deg
#These happen to be the first ten entries of the moments array
#i.e. indices zero to deg minus one

   
#Take zeroth image from each class of training
#as a key point for coherence classification
keys=tr.clcmp[:,0,:(deg+1)]
scores,preds=coherence(tst.cmp[:,:(deg+1)],keys)

#confusion matrix
c=np.zeros((10,10))
for k in range(10):
    tot=0
    for j in range(10):
        s=(preds[(100*k):(100*k+100)]==j).sum()
        c[k,j]=s
        tot+=s
    c[k]=1/tot*c[k]


   