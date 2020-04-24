# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:14:24 2020

@author: markb
"""
import view_images_script as ld
import moment_functions as mf
import matplotlib.pyplot as plt
import pickle

num_train=10 #of each digit
num_test=50
deg=10 #total degree of moments to be computed

#Load the image data and labels
Xtrain,trainLbls=ld.load_images('train',num_train)
Xtest,testLbls=ld.load_images('test',num_test)

#Y=Xtrain[0].reshape(28,28)
#plt.imshow(Y,cmap='gray',vmin=0,vmax=255,interpolation='none')

#construct moment objects which store the feature vectors
trCmp=mf.computeAll(Xtrain,deg)

fileNm="trainMoms_vectorized.data"
fileObj=open(fileNm,'wb') 
pickle.dump(trCmp,fileObj)
fileObj.close() 

#fileNm="testMoms.data"
#fileObj=open(fileNm,'wb') 
#pickle.dump(testMom,fileObj)
#fileObj.close() 
# fileObj=open("pickled.data",'rb')
# b=pickle.load(fileObj)
# fileObj.close()


#testMom=mf.MomentObj(Xtest,deg)



