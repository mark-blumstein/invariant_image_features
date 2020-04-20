# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:39:41 2020

@author: markb
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def load_images(opt,k):
    if (opt in ['train','test']) == False:
        return print("enter train or test")
    
    pth='../MNIST_Data/'
    
    dct={'train': ['train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz'],
        'test':['t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz'] }
    #Load first k images of each type into data
    num_images=k*10
    image_size=28
    num_pxls=image_size**2
    
    #Open the data
    file1=dct[opt][0]
    file2=dct[opt][1]

    f = gzip.open(pth+file1,'r') #labels
    g = gzip.open(pth+file2,'r') #data
    
    #Data has zeros in initial positions so advance buffers 
    f.read(8)
    g.read(16) 
    
    #Load data from the gzip file
    t=True
    ctr=np.zeros(10)
    
    #Store image tensor in X. 
    #Zeroth axis is the drawn number in the image
    #First axis is sample number
    #Second axis is the image stored in a vec
    X=np.zeros([10,k,num_pxls])
    
    while t==True:
        buff=f.read(1) #advance labels buffer
        n=np.frombuffer(buff,dtype=np.uint8) #current label
        n=int(n[0])
        bufg=g.read(num_pxls)#advance images buffer
        if ctr[n]<k:
            j=int(ctr[n])
            X[n,j,:num_pxls]=np.frombuffer(bufg,dtype=np.uint8).astype(np.float32)
            ctr[n]+=1
        if (ctr==k).sum()==10:
            t=False
    
    X=X.reshape(num_images,num_pxls)
    
    # labels=np.zeros([num_images,num_pxls])
    
    labels=np.zeros(num_images)
    ctr=0
    for l in range(10):
        for c in range(k):
            labels[ctr]=l
            ctr+=1
    
    # for l in range(10):
    #     for c in range(k):
    #         labels[c:c+k]=l
            
    return X,labels




def loadMomentData():
    fileNm="trainMoms.data"
    fileObj=open(fileNm,'rb') 
    x=pickle.load(fileObj)
    fileObj.close() 
    
    fileNm="testMoms.data"
    fileObj=open(fileNm,'rb') 
    y=pickle.load(fileObj)
    fileObj.close() 
    
    return x,y

#Plot some images
#for j in range(90,100):
#    image=X[7,j,:num_pxls].reshape(image_size,image_size)
#    plt.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='none')
#    plt.figure()
#
#plt.close('all')










#Old Code

#Load training data from gzip file into ndarray called data
#num_images = 1000
#image_size = 28
#
#
#f = gzip.open(pth+'train-images-idx3-ubyte.gz','r')
#
#f.read(16) #first two bytes are 0, this advances buffer
#buf = f.read(image_size * image_size * num_images)
#data = np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
#
##data = data.reshape(num_images, image_size, image_size, 1)
#
#
#
#
#
#
#
#
##Load labels from gzip file into ndarray called labels
#f = gzip.open(pth+'train-labels-idx1-ubyte.gz','r')
#f.read(8)
#buf=f.read(num_images)
#labels=np.frombuffer(buf,dtype=np.uint8).astype(np.int64)


    

#image = np.asarray(data[3]).squeeze()
#plt.imshow(image)
#plt.show()

#for i in range(0,50):   
#    buf = f.read(1)
#    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#    print(labels)

#X=np.zeros(num_images*image_size*image_size)
#while t==True:    
#    buff=f.read(1) #advance labels buffer
#    n=np.frombuffer(buff,dtype=np.uint8) #current label
#    n=n[0]
#    bufg=g.read(num_pxls)#advance images buffer
#    if ctr[n]<100:
#        labels[c]=n
#        X[c*num_pxls:(c+1)*num_pxls]=np.frombuffer(bufg,dtype=np.uint8).astype(np.float32)
#        ctr[n]+=1
#        c+=1
#    if (ctr==k).sum()==10:
#        t==False
#    
        
##Display the jth image. use gray color map
#j=3
#image=X[(j*image_size**2):((j+1)*image_size**2)].reshape(image_size,image_size)
##set interpolation to none for showing image
#plt.imshow(image,cmap='gray', vmin=0, vmax=255, interpolation='none')
    