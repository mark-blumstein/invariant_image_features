# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:28:51 2020

@author: markb
"""

import numpy as np
import moment_functions as mf
import matplotlib.pyplot as plt
import view_images_script as ld

####Moved to moment_functions.py
    
#def buildMoment(X,dMax)
#    dMax=10 #number of moments to compute
#    
#    #extract moment features for training and testing
#    
#    train=mf.momentClass(Xtrain,dMax)
#    print("-"*30,"Computing Training Data Moment Object","-"*30)
#    train.compute_all()
#    print("-"*30,"Finished Training Data Moment Object","-"*30)
#    print("\n")
#    test=mf.momentClass(Xtest,dMax)
#    print("-"*30,"Computing Testing Data Moment Object","-"*30)
#    test.compute_all()
#    print("-"*30,"Finished Training Data Moment Object","-"*30)
#
#    return train,test





























#chr_num=X.shape[0] #character number i.e. 0,1,2,3..,9
#img_num=X.shape[1] #which sample of a given character
#M=np.zeros([chr_num,img_num,dMax+1,dMax+1])
#
#
#for n in range(chr_num):
#    for k in range(img_num):
#        for d in range(dMax+1):
#            for p in range(d+1):
#                q=d-p
#                M[n,k,p,q]=mf.Standard_Moment(X[n,k,:].reshape(28,28),p,q)
#    print(n)
#
##Vectorize the moments
#num_moms=int((dMax+1)*(dMax+2)/2)
#
#Mvec=np.zeros([chr_num,img_num,num_moms])
#vec_idx=np.empty(num_moms,dtype=tuple)
#c=0
#for d in range(dMax+1):
#    for q in range(d+1):
#        p=d-q
#        vec_idx[c]=(p,q)
#        c+=1
#        
#c=0
#for n in range(chr_num):
#    for k in range(img_num):
#        c=0
#        for d in range(dMax+1):
#            for q in range(d+1):
#                p=d-q
#                Mvec[n,k,c]=M[n,k,p,q]
#                c+=1
#
##Compute the complex moments
#C=np.zeros(M.shape,dtype=np.complex)
#for n in range(chr_num):
#    for k in range(img_num):
#        C[n,k,:,:]=mf.Complexify_Moments(M[n,k,:,:],dMax)
#
#
##Vectorized complex moments (take only p>=q)
#num_cmoms=1
#for d in range(1,dMax+1):
#    for q in range(int(np.floor(d/2)+1)):
#        num_cmoms+=1
#
#cvec_idx=np.empty(num_cmoms,dtype=tuple)
#Cvec=np.zeros([chr_num,img_num,num_cmoms],dtype=complex)
#
#for n in range(chr_num):
#    for k in range(img_num):
#        c=0
#        for d in range(dMax+1):
#            for q in range(int(np.floor(d/2)+1)):
#                p=d-q
#                Cvec[n,k,c]=C[n,k,p,q]
#                cvec_idx[c]=(p,q)
#                c+=1
#                
#
#
#
##M=np.zeros([dMax+1,dMax+1])
##
##
##for d in range(dMax+1):
##    for p in range(d+1):
##        q=d-p
##        M[p,q]=mf.Standard_Moment(Y,p,q)
##
##C=mf.Complexify_Moments(M,dMax)
#
##Compute the Flusser Invariant Basis
#B=np.zeros(M.shape,dtype=np.complex)
#
#for n in range(chr_num):
#    for k in range(img_num):
#        B[n,k,:,:]=mf.Flus_Moment(C[n,k,:,:])
#
#
##Log scale on B
#L=np.zeros(B.shape,dtype=np.complex)
#for n in range(chr_num):
#    for k in range(img_num):
#        for idx, x in np.ndenumerate(B[n,k,:,:]):
#            p=idx[0]
#            q=idx[1]
#            if p>=q:
#                if x==0+0j:
#                    L[n,k,p,q]=-100
#                else:
#                    L[n,k,p,q]=np.log(B[n,k,p,q])
#
#        
#B=mf.Flus_Moment(C)
#
#
#
#
#L=np.zeros(B.shape,dtype=np.complex)
#
#for idx, x in np.ndenumerate(B):
#    p=idx[0]
#    q=idx[1]
#    if p>= q:
#        if x==0+0j:
#            L[idx]=-100
#        else: 
#            L[idx]=np.log(B[idx])
#            
#
#        
#        
#        
#
#        
#
