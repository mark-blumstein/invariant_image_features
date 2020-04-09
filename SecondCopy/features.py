# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:00:19 2020

@author: markb
"""


#####################################################
#### make vector of labels
num_pxls=784

trainLbls=np.zeros(tot_train)
testLbls=np.zeros(tot_test)

c=0
d=0
for i in range(tot_train):
    trainLbls[i]=d
    c+=1
    if (c%num_train) ==0:
        d+=1
        
c=0
d=0
for i in range(tot_test):
    testLbls[i]=d
    c+=1
    if (c%num_test) ==0:
        d+=1
        
        
###############################
##### Store feature vecs 
        
a=train.MomentsVec[0,0,:].shape[0]
trainMom=train.MomentsVec.reshape(tot_train,a)
testMom=test.MomentsVec.reshape(tot_test,a)

a=train.stdMoments[0,0,:].shape[0]
trainStdMom=train.stdMoments.reshape(tot_train,a)
testStdMom=test.stdMoments.reshape(tot_test,a)


a=train.cmpMoments[0,0,:].shape[0]
trainCmpMom=train.cmpMoments.reshape(tot_train,a)
testCmpMom=test.cmpMoments.reshape(tot_test,a)


a=train.flusMoments[0,0,:].shape[0]
trainFlusMom=train.flusMoments.reshape(tot_train,a)
testFlusMom=test.flusMoments.reshape(tot_test,a)



