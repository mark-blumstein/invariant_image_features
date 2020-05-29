# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:33:58 2020

@author: markb
"""
import numpy as np
#from sklearn.datasets import load_iris
#from sklearn.linear_model import LogisticRegression
#from scipy import integrate
import view_images_script as ld
import itertools
import matplotlib.pyplot as plt
from scipy.ndimage import geometric_transform, affine_transform, rotate
a = np.arange(12.).reshape((4, 3))
def shift_func(output_coords):
    return (output_coords[0] - 0.5, output_coords[1] - 0.5)

Y,lbl=ld.load_images('train',2)
Z=Y[0].reshape(28,28)

theta=30
g=rotate(Z,theta)
plt.figure()
plt.imshow(g)
plt.show()


theta=0
c=np.cos(theta)
s=np.sin(theta)
R=np.array([[c,-s],[s,c]])

g=affine_transform(Z,R)
plt.figure()
plt.imshow(g)
plt.show()


theta=0
def rotate_func(output_coords):
    x=output_coords[0]
    y=output_coords[1]
    c=np.cos(theta)
    s=np.sin(theta)
    return (c*x+-1*s*y,c*x+s*y)
g=geometric_transform(Z, rotate_func)
plt.figure()
plt.imshow(g)
plt.show()


W=Y[1].reshape(28,28)
a=np.zeros(4)
b=np.ones(4)

b*=a
print(b)

lst=['a','b','c','d','e','f']
lst2=list(map(lambda x: x+'z',lst))

x=np.arange(10)
y=list(map(lambda x:x+2,x))


    