# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:14:24 2020

@author: markb
"""
import view_images_script as ld
import moment_functions as mf
import numpy as np
import matplotlib.pyplot as plt
import pickle

num_train=500 #of each digit
num_test=100
deg=8 #total degree of moments to be computed

#Load the image data and labels
Xtrain,trainLbls=ld.load_images('train',num_train)
Xtest,testLbls=ld.load_images('test',num_test)

#Y=Xtrain[0].reshape(28,28)
#plt.imshow(Y,cmap='gray',vmin=0,vmax=255,interpolation='none')

#construct moment objects which store the feature vectors
tr_mts=mf.Moments(Xtrain)
#tr_mts=mf.Moments()
#tr_mts.fit(Xtrain)
tr_mts.compute(deg)

te_mts=mf.Moments(Xtest)
#te_mts.fit(Xtest)
te_mts.compute(deg)
        

fileNm="train_mts_july2020.data"
fileObj=open(fileNm,'wb') 
pickle.dump(tr_mts,fileObj)
fileObj.close() 

fileNm="test_mts_july2020.data"
fileObj=open(fileNm,'wb') 
pickle.dump(te_mts,fileObj)
fileObj.close() 




#testMom=mf.MomentObj(Xtest,deg)



