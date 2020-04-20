# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:07:45 2020

@author: markb
"""
import numpy as np
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression

#################################################################
#####Logistic Regression

#clf=LogisticRegression()
#clf.fit(np.log(trainMom),trainLbls)
#preds=clf.predict(np.log(testMom))
#num_corr=(preds==testLbls).sum()
#pct_corr=num_corr/testLbls.shape[0]


clf=LogisticRegression()
clf.fit(trainMom,trainLbls)
preds=clf.predict(testMom)
num_corr=(preds==testLbls).sum()
pct_corr=num_corr/testLbls.shape[0]
print("Geometric Moments: ",pct_corr,"\n")

###################


clf=LogisticRegression()
s=np.sign(trainStdMom)
t=np.abs(trainStdMom)
clf.fit(s*np.log(t),trainLbls)
s=np.sign(testStdMom)
t=np.abs(testStdMom)
preds=clf.predict(s*np.log(t))
num_corr=(preds==testLbls).sum()
pct_corr=num_corr/testLbls.shape[0]
print("Standardized Moments: ",pct_corr,"\n")

###################


clf=LogisticRegression()
x=np.log(trainCmpMom)
z=np.zeros([x.shape[0],2*x.shape[1]])
c=0
for idx, n in np.ndenumerate(x):
    z[idx[0],c:(c+2)]=x[idx].real,x[idx].imag
    c+=2
    if idx[1]==x.shape[1]-1:
        c=0

clf.fit(z,trainLbls)

x=np.log(testCmpMom)
z=np.zeros([x.shape[0],2*x.shape[1]])
c=0
for idx, n in np.ndenumerate(x):
    z[idx[0],c:(c+2)]=x[idx].real,x[idx].imag
    c+=2
    if idx[1]==x.shape[1]-1:
        c=0

preds=clf.predict(z)

num_corr=(preds==testLbls).sum()
pct_corr=num_corr/x.shape[0]
print("Complex Moments (cmplx log): ",pct_corr,"\n")


###################


clf=LogisticRegression()
x=trainCmpMom
z=np.zeros([x.shape[0],2*x.shape[1]])
c=0
for idx, n in np.ndenumerate(x):
    z[idx[0],c:(c+2)]=x[idx].real,x[idx].imag
    c+=2
    if idx[1]==x.shape[1]-1:
        c=0

clf.fit(z,trainLbls)

x=testCmpMom
z=np.zeros([x.shape[0],2*x.shape[1]])
c=0
for idx, n in np.ndenumerate(x):
    z[idx[0],c:(c+2)]=x[idx].real,x[idx].imag
    c+=2
    if idx[1]==x.shape[1]-1:
        c=0

preds=clf.predict(z)

num_corr=(preds==testLbls).sum()
pct_corr=num_corr/x.shape[0]
print("Complex Moments (no log): ",pct_corr,"\n")



###################


#
#clf=LogisticRegression()
#x=np.log(trainFlusMom)
#z=np.zeros([x.shape[0],2*x.shape[1]])
#c=0
#for idx, n in np.ndenumerate(x):
#    z[idx[0],c:(c+2)]=x[idx].real,x[idx].imag
#    c+=2
#    if idx[1]==x.shape[1]-1:
#        c=0
#
#clf.fit(z,trainLbls)
#
#x=np.log(testFlusMom)
#z=np.zeros([x.shape[0],2*x.shape[1]])
#c=0
#for idx, n in np.ndenumerate(x):
#    z[idx[0],c:(c+2)]=x[idx].real,x[idx].imag
#    c+=2
#    if idx[1]==x.shape[1]-1:
#        c=0
#
#preds=clf.predict(z)
#
#num_corr=(preds==testLbls).sum()
#pct_corr=num_corr/x.shape[0]
#print("Invariant Moments (cmplx log): ",pct_corr,"\n")