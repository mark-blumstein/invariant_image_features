# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:07:45 2020

@author: markb
"""
import numpy as np
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
import view_images_script as ld
import moment_functions as mf
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import svm


deg=ld.deg
num_train=ld.num_train #of each digit
num_test=ld.num_test
tr,tst=ld.loadMomentData()
tr.lbls=np.kron(np.arange(10),np.ones(num_train)).ravel()
tst.lbls=np.kron(np.arange(10),np.ones(num_test)).ravel()

tr.geo_std=preprocessing.scale(tr.geo)
tst.geo_std=preprocessing.scale(tst.geo)
#################################################################
#####Logistic Regression
#clf=LogisticRegression()
reg=LogisticRegression(max_iter=1000)
parameters={'C':np.linspace(0,1,10)}
clf=GridSearchCV(reg,parameters)
#clf.fit(tr.geo,tr.lbls)
clf.fit(tr.geo_std,tr.lbls)
#preds=clf.predict(tst.geo)
preds=clf.predict(tst.geo_std)
num_corr=(preds==tst.lbls).sum()
pct_corr=num_corr/tst.lbls.shape[0]
print(pct_corr)
 #scored 81% total accuracy


clf_svm=svm.SVC()
clf_svm.fit(tr.geo_std,tr.lbls)
preds=clf_svm.predict(tst.geo_std)
num_corr=(preds==tst.lbls).sum()
pct_corr=num_corr/tst.lbls.shape[0]
print(pct_corr)
#####################################

idx=tr.idx
        
tr.flus=np.zeros(tr.cmp.shape)
for ctr in range(tr.flus.shape[0]):
    tr.flus[ctr]=mf.flus(tr.cmp[ctr],idx)

tr.cmp_flus_split=np.array([tr.cmp_flus.real,tr.cmp_flus.imag])    
tr.cmp_flus_std=preprocessing.scale(tr.cmp_flus)



tst.cmp_flus=np.zeros(tst.cmp.shape)
for ctr in range(tst.cmp_flus.shape[0]):
    tst.cmp_flus[ctr]=mf.flus(tst.cmp_flus,idx)
    
tst.cmp_flus_std=preprocessing.scale(tst.cmp_flus)

clf_flus=LogisticRegression()
clf.fit(tr.geo_std,tr.lbls)
preds=clf.predict(tst.geo_std)

num_corr=(preds==tst.lbls).sum()
pct_corr=num_corr/tst.lbls.shape[0]