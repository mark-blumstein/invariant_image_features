# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:14:24 2020

@author: markb
"""
import view_images_script as ld
import moment_functions as mf


num_train=100 #of each digit
num_test=50
deg=4 #total degree of moments to be computed

#Load the image data and labels
Xtrain,trainLbls=ld.load_images('train',num_train)
Xtest,testLbls=ld.load_images('test',num_test)

#construct moment objects which store the feature vectors
trainMom=mf.momentClass(Xtrain,deg)
trainMom.compute_all()

testMom=mf.momentClass(Xtest,deg)
testMom.compute_all()


