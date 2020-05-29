# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:03:56 2020

@author: markb
"""

import numpy as np
import view_images_script as ld
import moment_functions as mf
import matplotlib.pyplot as plt 
from scipy.ndimage import convolve

from scipy.fft import fft2


deg=ld.deg
num_train=ld.num_train #of each digit
num_test=ld.num_test
tr,tst=ld.loadMomentData()

#Sample image
X=tr.data[0].reshape(28,28)
mts=tr.cmp[0]
amts=np.abs(mts)
plt.imshow(X); plt.show()


#convolution
x_lbs=np.arange(-28,28).reshape(1,56)
y_lbs=np.arange(-28,28).reshape(56,1)
lbs=x_lbs*y_lbs
c=convolve(X,lbs,mode='constant',cval=0.0)
plt.imshow(c)
plt.show()


for p in range(5):
    for q in range(5):
        lbs=(x_lbs)**p*(y_lbs)**q
        c=convolve(X,lbs,mode='constant',cval=0.0)
        plt.figure()
        plt.imshow(c)
        plt.title(str(p)+str(q))

#compute first and second moment "about zero" (non-normalized in any way)

    
idx={}
ctr=0
for q in range(deg+1):
    for p in range(deg+1-q):
        idx[(p,q)]=ctr
        idx[ctr]=(p,q)
        ctr+=1
        
#get indices in "diagonal" order
didx=[idx.get(x) for x in mf.gen_diag_all(deg)]
mts[]